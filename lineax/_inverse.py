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
import jax
import jax.lax as lax
import jax.numpy as jnp

from ._operator import (
    AbstractLinearOperator,
    conj,
    diagonal,
    FunctionLinearOperator,
    has_unit_diagonal,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    materialise,
    MatrixLinearOperator,
    tridiagonal,
)
from ._solve import AbstractLinearSolver, AutoLinearSolver, linear_solve
from ._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)


class InverseLinearOperator(AbstractLinearOperator):
    """Represents the inverse of a linear operator.

    `InverseLinearOperator(A).mv(v)` computes `A^{-1} v` by solving `A x = v`
    with `linear_solve`. This is useful for obtaining a matrix representation
    of the inverse with a specific solver, composing inverses with other operators
    (such as is required in the Woodbury matrix identity). It also enables an
    intuitive way to cache the state without breaking AD through `linearise`.
    Currently, `materialise` will always return a single concretised operator
    (never a `ComposedLinearOperator`), if you need direct access to factorisations
    see [`solver.init`](../api/solvers.md).

    **Arguments:**

    - `operator`: the linear operator to invert. Must be square.
    - `solver`: the linear solver to use. Defaults to
        `AutoLinearSolver(well_posed=True)`.
    - `options`: additional options passed to the solver. Defaults to `None`.
    """

    operator: AbstractLinearOperator
    solver: AbstractLinearSolver = eqx.field(
        static=True, default=AutoLinearSolver(well_posed=True)
    )
    options: dict[str, Any] | None = None

    def __check_init__(self):
        if self.operator.in_size() != self.operator.out_size():
            raise ValueError(
                "InverseLinearOperator requires a square operator, but got "
                f"in_size={self.operator.in_size()} and "
                f"out_size={self.operator.out_size()}."
            )
        well_posed = getattr(self.solver, "well_posed", True)
        if not well_posed:
            raise ValueError(
                "InverseLinearOperator requires a well-posed solver, but got "
                f"`solver.well_posed={well_posed}`. Use a well-posed solver "
                "(e.g. `AutoLinearSolver(well_posed=True)`)."
            )

    def mv(self, vector):
        options = self.options if self.options is not None else {}
        return linear_solve(self.operator, vector, self.solver, options=options).value

    def as_matrix(self):
        return materialise(self).as_matrix()

    def transpose(self):
        return InverseLinearOperator(
            self.operator.transpose(), self.solver, self.options
        )

    def in_structure(self):
        return self.operator.out_structure()

    def out_structure(self):
        return self.operator.in_structure()


def _inverse_tags(inv_operator):
    tags = set()
    for check, tag in [
        (is_symmetric, symmetric_tag),
        (is_diagonal, diagonal_tag),
        (is_tridiagonal, tridiagonal_tag),
        (is_lower_triangular, lower_triangular_tag),
        (is_upper_triangular, upper_triangular_tag),
        (is_positive_semidefinite, positive_semidefinite_tag),
        (is_negative_semidefinite, negative_semidefinite_tag),
        (has_unit_diagonal, unit_diagonal_tag),
    ]:
        if check(inv_operator):
            tags.add(tag)
    return frozenset(tags)


@linearise.register(InverseLinearOperator)
def _(operator):
    options = operator.options if operator.options is not None else {}
    state = operator.solver.init(operator.operator, options)
    dynamic_state, static_state = eqx.partition(state, eqx.is_array)
    dynamic_state = lax.stop_gradient(dynamic_state)
    state = eqx.combine(dynamic_state, static_state)

    def solve_fn(vector):
        return linear_solve(
            operator.operator,
            vector,
            operator.solver,
            state=state,
            options=options,
        ).value

    tags = _inverse_tags(operator)
    return FunctionLinearOperator(solve_fn, operator.in_structure(), tags)


@materialise.register(InverseLinearOperator)
def _(operator):
    n = operator.in_size()
    eye = jnp.eye(
        n, dtype=jnp.result_type(*jax.tree_util.tree_leaves(operator.in_structure()))
    )
    matrix = jax.vmap(operator.mv, in_axes=1, out_axes=1)(eye)
    return MatrixLinearOperator(matrix, _inverse_tags(operator))


@diagonal.register(InverseLinearOperator)
def _(operator):
    if is_diagonal(operator.operator):
        return 1.0 / diagonal(operator.operator)
    return jnp.diag(operator.as_matrix())


@tridiagonal.register(InverseLinearOperator)
def _(operator):
    matrix = operator.as_matrix()
    diag = jnp.diag(matrix)
    lower = jnp.diag(matrix, k=-1)
    upper = jnp.diag(matrix, k=1)
    return (diag, lower, upper)


# Inverse preserves these structural properties.
# (PSD/NSD is included because InverseLinearOperator enforces invertibility
# via the well-posed solver requirement, so the operator is strictly
# positive/negative definite — for which inversion preserves definiteness.)
for check in (
    is_symmetric,
    is_diagonal,
    is_lower_triangular,
    is_upper_triangular,
    is_positive_semidefinite,
    is_negative_semidefinite,
):

    @check.register(InverseLinearOperator)
    def _(operator, check=check):
        return check(operator.operator)


@is_tridiagonal.register(InverseLinearOperator)
def _(operator):
    return is_diagonal(operator.operator)


@has_unit_diagonal.register(InverseLinearOperator)
def _(operator):
    return has_unit_diagonal(operator.operator) and (
        is_lower_triangular(operator.operator) or is_upper_triangular(operator.operator)
    )


@conj.register(InverseLinearOperator)
def _(operator):
    return InverseLinearOperator(
        conj(operator.operator), operator.solver, operator.options
    )
