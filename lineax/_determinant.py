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

from typing import Any, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ._operator import AbstractLinearOperator
from ._solve import AbstractDirectLinearSolver
from ._solver.normal import Normal


MaybeDirectLinearSolver: TypeAlias = AbstractDirectLinearSolver | Normal


def _det_sign_error_msg(solver: MaybeDirectLinearSolver, operator: AbstractLinearOperator) -> str:
    if isinstance(solver, Normal):
        return (
            "`lx.determinant` with `Normal`: sign of the determinant is not "
            "recoverable — the gram matrix construction destroys sign information. "
            "Use `lx.LU()` instead."
        )
    if operator.in_size() != operator.out_size():
        return (
            "`lx.determinant`: sign of the determinant is undefined for non-square "
            "operators. Use `lx.logabsdet` for the log absolute value, or a square "
            "operator."
        )
    if not solver.assume_full_rank():
        return (
            f"`lx.determinant` with `{type(solver).__name__}`: sign of the "
            "determinant is not available from this solver's factorisation. "
            "Use `lx.LU()` instead."
        )
    return (
        f"`lx.determinant` with `{type(solver).__name__}`: sign of the determinant "
        "is not available from this solver's factorisation. Use `lx.LU()` instead."
    )


def slogdet(
    operator: AbstractLinearOperator,
    solver: MaybeDirectLinearSolver,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[Array, Array]:
    """Compute `(sign, log|det(operator)|)` using the given direct solver.

    Follows the same convention as `numpy.linalg.slogdet`.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: a [`lineax.MaybeDirectLinearSolver`][].
    - `options`: any extra options to pass to the solver.

    **Returns:**

    A 2-tuple of `(sign, logabsdet)`. `sign` is `nan` when the solver cannot
    recover it (e.g. [`lineax.Normal`][] or [`lineax.SVD`][]).
    """
    if options is None:
        options = {}
    state = solver.init(operator, options)
    return solver.slogdet(state, options)


def logabsdet(
    operator: AbstractLinearOperator,
    solver: MaybeDirectLinearSolver,
    *,
    options: dict[str, Any] | None = None,
) -> Array:
    """Compute log|det(operator)| using the given direct solver.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: a [`lineax.MaybeDirectLinearSolver`][].
    - `options`: any extra options to pass to the solver.

    **Returns:**

    A scalar array equal to the log of the absolute value of the determinant.
    """
    if options is None:
        options = {}
    state = solver.init(operator, options)
    _, lad = solver.slogdet(state, options)
    return lad


def determinant(
    operator: AbstractLinearOperator,
    solver: MaybeDirectLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    throw: bool = True,
) -> Array:
    """Compute det(operator) using the given direct solver.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: a [`lineax.MaybeDirectLinearSolver`][].
    - `options`: any extra options to pass to the solver.
    - `throw`: if `True` (the default), raise an error when the sign of the
        determinant is not available (e.g. when using [`lineax.Normal`][] or
        [`lineax.SVD`][]). If `False`, a `nan` result is returned silently.

    **Returns:**

    A scalar array equal to the determinant.
    """
    if options is None:
        options = {}
    state = solver.init(operator, options)
    sign, lad = solver.slogdet(state, options)
    det = sign * jnp.exp(lad)
    if throw:
        msg = _det_sign_error_msg(solver, operator)
        det = eqx.error_if(det, jnp.isnan(sign), msg)
    return det
