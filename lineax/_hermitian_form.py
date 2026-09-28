# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, Float, PyTree

from ._custom_types import sentinel
from ._misc import inexact_asarray
from ._norm import tree_dot
from ._operator import (
    AbstractLinearOperator,
    is_full_rank,
    is_hermitian,
    linearise,
    TangentLinearOperator,
)
from ._solve import (
    AbstractLinearSolver,
    check_rank_compat,
    linear_solve,
    partial_isometry_fast_path,
)
from ._solver import AutoLinearSolver


def _is_none(x):
    return x is None


@eqx.filter_custom_jvp
def _dual_hermitian_form(vector, operator, state, options, solver, throw):
    w = linear_solve(
        operator, vector, solver, state=state, options=options, throw=throw
    ).value
    return tree_dot(vector, w).real


@_dual_hermitian_form.def_jvp
def _dual_hermitian_form_jvp(primals, tangents):
    vector, operator, state, options, solver, throw = primals
    t_vector, t_operator, _, _, _, _ = tangents

    # With `w = A⁺x` and `A` Hermitian (so `A⁺ᴴ = A⁺`),
    #
    # d(xᴴA⁺x) = xᴴ (A⁺)' x + 2 Re(wᴴ x'),
    #
    # and the derivative of the pseudoinverse (as in `_linear_solve_jvp`) is
    #
    # (A⁺)' = -A⁺A'A⁺ + A⁺A⁺ᴴA'ᴴ(I - AA⁺) + (I - A⁺A)A'ᴴA⁺ᴴA⁺.
    #
    # The first term gives `-wᴴA'w`, a single matvec against the solution we already
    # have. With `r = (I - AA⁺)x = x - Aw` and `y = A⁺w`, the other two give
    # `yᴴA'ᴴr + rᴴA'ᴴy`, which vanish when `A` is nonsingular. A tangent operator
    # carries the tags of its primal, so `A'` is Hermitian too, and these two terms are
    # complex conjugates: together `2 Re(rᴴA'y)`.
    #
    # The primal solve respects `throw`; the tangent solve uses `throw=True`, as there
    # is nowhere to pipe its error (see `_linear_solve_jvp`).
    w = linear_solve(
        operator, vector, solver, state=state, options=options, throw=throw
    ).value
    out = tree_dot(vector, w).real
    t_out = jnp.zeros_like(out)
    if any(t is not None for t in jtu.tree_leaves(t_vector, is_leaf=_is_none)):
        t_vector = jtu.tree_map(
            eqxi.materialise_zeros, vector, t_vector, is_leaf=_is_none
        )
        t_out = t_out + 2 * tree_dot(w, t_vector).real
    if any(t is not None for t in jtu.tree_leaves(t_operator, is_leaf=_is_none)):
        t_operator = linearise(TangentLinearOperator(operator, t_operator))
        t_out = t_out - tree_dot(w, t_operator.mv(w)).real
        # As in `_linear_solve_jvp`, the correction is exactly zero if the operator is
        # declared full rank, whether or not the solver assumes it.
        if not (solver.assume_full_rank() or is_full_rank(operator)):
            r = (vector**ω - operator.mv(w) ** ω).ω
            y = linear_solve(
                operator, w, solver, state=state, options=options, throw=True
            ).value
            t_out = t_out + 2 * tree_dot(r, t_operator.mv(y)).real
    return out, t_out


def dual_hermitian_form(
    vector: PyTree[ArrayLike],
    operator: AbstractLinearOperator,
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=True),
    *,
    options: dict[str, Any] | None = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> Float[Array, ""]:
    r"""Computes the dual Hermitian form $x^\mathrm{H} A^\dagger x$ of a Hermitian
    operator $A$, where $A^\dagger$ is its (pseudo)inverse.

    This is the quadratic term of a Gaussian log-likelihood (the squared Mahalanobis
    distance). The operator must be Hermitian (real symmetric included); if it is
    singular then the pseudoinverse is used, as in [`lineax.linear_solve`][] with the
    same solver. Its derivatives are cheaper than those of
    `tree_dot(x, linear_solve(A, x).value)`: differentiating with respect to the
    operator costs a matvec rather than a second linear solve.

    **Arguments:**

    - `vector`: the vector $x$. Should have the structure of `operator.in_structure()`.
    - `operator`: the Hermitian linear operator $A$.
    - `solver`: the linear solver to use. Defaults to
        `AutoLinearSolver(well_posed=True)`.
    - `options`: additional options passed to the solver. Defaults to `None`.
    - `state`: if passed, this should be the state of the solver, as initialised by
        `solver.init(operator, options)`. If not passed then it will be initialised,
        with gradients stopped through the operator (as in [`lineax.linear_solve`][]).
    - `throw`: as [`lineax.linear_solve`][]. Defaults to `True`.

    **Returns:**

    A real scalar array.
    """
    if eqx.is_array(operator):
        raise ValueError(
            "`lineax.dual_hermitian_form(operator=...)` should be an "
            "`AbstractLinearOperator`, not a raw JAX array. If you are trying to pass "
            "a matrix then this should be passed as "
            "`lineax.MatrixLinearOperator(matrix)`."
        )
    if not is_hermitian(operator):
        raise ValueError(
            "`lineax.dual_hermitian_form` requires a Hermitian operator. For a "
            "non-Hermitian operator, take the inner product of `vector` with "
            "`lineax.linear_solve(operator, vector).value` instead."
        )
    if options is None:
        options = {}
    vector = jtu.tree_map(inexact_asarray, vector)
    check_rank_compat(solver, operator)
    # For a partial isometry `linear_solve` never uses the state, so don't factorise.
    if state is sentinel and not partial_isometry_fast_path(solver, operator, options):
        dynamic_operator, static_operator = eqx.partition(operator, eqx.is_array)
        stopped_operator = eqx.combine(
            lax.stop_gradient(dynamic_operator), static_operator
        )
        state = solver.init(stopped_operator, options)
    dynamic_state, static_state = eqx.partition(state, eqx.is_array)
    state = eqx.combine(lax.stop_gradient(dynamic_state), static_state)
    options = eqxi.nondifferentiable(
        options, name="`lineax.dual_hermitian_form(..., options=...)`"
    )
    solver = eqxi.nondifferentiable(
        solver, name="`lineax.dual_hermitian_form(..., solver=...)`"
    )
    return _dual_hermitian_form(vector, operator, state, options, solver, throw)
