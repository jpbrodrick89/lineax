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
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from ._custom_types import sentinel
from ._operator import AbstractLinearOperator, TangentLinearOperator
from ._solve import AbstractDirectLinearSolver, linear_solve
from ._solver.normal import Normal


def _is_none(x):
    return x is None


def _det_sign_error_msg(
    solver: "AbstractDirectLinearSolver | Normal",
    operator: AbstractLinearOperator,
) -> str:
    if not solver.assume_full_rank():
        return (
            f"`lx.determinant` with `{type(solver).__name__}`: sign of the "
            "determinant is not available from this solver's factorisation. "
            "Use `lx.LU()` for full-rank square matrices, or `lx.QR()` for "
            "full-rank rectangular matrices."
        )
    if isinstance(solver, Normal):
        return (
            "`lx.determinant` with `Normal`: sign of the determinant is not "
            "recoverable — the gram matrix construction destroys sign information. "
            "To recover the sign of a full-rank rectangular matrix's "
            "pseudodeterminant, use `lx.QR()`."
        )
    return (
        f"`lx.determinant` with `{type(solver).__name__}`: sign of the determinant "
        "is not available from this solver's factorisation. "
        "Use `lx.LU()` for full-rank square matrices, or `lx.QR()` for "
        "full-rank rectangular matrices."
    )



@eqx.filter_custom_jvp
def _slogdet(operator, solver, options, state):
    state = eqxi.nondifferentiable(state, name="`lx.slogdet` state")
    return solver.slogdet(state, options)


@_slogdet.def_jvp
def _slogdet_jvp(primals, tangents):
    operator, solver, options, state = primals
    t_operator, _, _, _ = tangents

    # Primal — state is already stop-gradiented by the caller
    sign, lad = solver.slogdet(state, options)

    # Tangent: d(lad)/dA = trace(A† dA), where A† is the pseudoinverse
    has_t_op = any(
        t is not None for t in jtu.tree_leaves(t_operator, is_leaf=_is_none)
    )
    if has_t_op:
        dA = TangentLinearOperator(operator, t_operator).as_matrix()  # (m, n)

        def solve_col(col):
            # Reuse the stopped state for efficiency
            return linear_solve(operator, col, solver, state=state, throw=False).value

        # vmap over the n columns of dA; each solve gives an n-vector
        # X[i] = A† dA[:,i], so trace(A† dA) = trace(X)
        X = jax.vmap(solve_col)(dA.T)  # (n, n)
        lad_dot = jnp.trace(X)

        if jnp.issubdtype(dA.dtype, jnp.complexfloating):
            # For complex A: sign carries the imaginary part of the trace
            sign_dot = (lad_dot - jnp.real(lad_dot).astype(lad_dot.dtype)) * sign
            lad_dot = jnp.real(lad_dot)
        else:
            sign_dot = jnp.zeros_like(sign)
    else:
        sign_dot = jnp.zeros_like(sign)
        lad_dot = jnp.zeros_like(lad)

    return (sign, lad), (sign_dot, lad_dot)


def slogdet(
    operator: AbstractLinearOperator,
    solver: "AbstractDirectLinearSolver | Normal",
    *,
    options: dict[str, Any] | None = None,
    state: Any = sentinel,
) -> tuple[Array, Array]:
    """Compute `(sign, log|det(operator)|)` using the given direct solver.

    Follows the same convention as `numpy.linalg.slogdet`.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: an [`lineax.AbstractDirectLinearSolver`][] or [`lineax.Normal`][].
    - `options`: any extra options to pass to the solver.
    - `state`: if provided, use this pre-computed factorised state instead of
        calling `solver.init`. Allows multiple determinant computations to share
        the same factorisation.

        !!! warning

            Do **not** apply `lax.stop_gradient` to this state. `slogdet` applies
            it internally (like [`lineax.linear_solve`][]) and uses an analytic JVP
            rule for differentiation. Manually stopping gradients before passing the
            state will break the JVP.

    **Returns:**

    A 2-tuple of `(sign, logabsdet)`. `sign` is `nan` when the solver cannot
    recover it cheaply (e.g. [`lineax.SVD`][] on a full-rank square matrix, or
    [`lineax.Normal`][]).
    """
    if options is None:
        options = {}
    if state is sentinel:
        dynamic_op, static_op = eqx.partition(operator, eqx.is_array)
        stopped_op = eqx.combine(lax.stop_gradient(dynamic_op), static_op)
        state = solver.init(stopped_op, options)
    dynamic_state, static_state = eqx.partition(state, eqx.is_array)
    state = eqx.combine(lax.stop_gradient(dynamic_state), static_state)
    return _slogdet(operator, solver, options, state)


def determinant(
    operator: AbstractLinearOperator,
    solver: "AbstractDirectLinearSolver | Normal",
    *,
    options: dict[str, Any] | None = None,
    state: Any = sentinel,
    throw: bool = True,
) -> Array:
    """Compute det(operator) using the given direct solver.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: an [`lineax.AbstractDirectLinearSolver`][] or [`lineax.Normal`][].
    - `options`: any extra options to pass to the solver.
    - `state`: if provided, use this pre-computed factorised state instead of
        calling `solver.init`. Allows multiple determinant computations to share
        the same factorisation.

        !!! warning

            Do **not** apply `lax.stop_gradient` to this state. `determinant`
            applies it internally (like [`lineax.linear_solve`][]) and uses an
            analytic JVP rule for differentiation. Manually stopping gradients
            before passing the state will break the JVP.

    - `throw`: if `True` (the default), raise an error when the sign of the
        determinant is not available (e.g. when using [`lineax.Normal`][] or
        [`lineax.SVD`][] on a full-rank square matrix). If `False`, a `nan`
        result is returned silently.

    **Returns:**

    A scalar array equal to the determinant.
    """
    if options is None:
        options = {}
    sign, lad = slogdet(operator, solver, options=options, state=state)
    det = sign * jnp.exp(lad)
    if throw:
        msg = _det_sign_error_msg(solver, operator)
        det = eqx.error_if(det, jnp.isnan(sign), msg)
    return det
