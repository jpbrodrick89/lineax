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
import jax.numpy as jnp
from jaxtyping import Array

from ._operator import AbstractLinearOperator
from ._solve import AbstractDirectLinearSolver
from ._solver.normal import Normal


def _det_sign_error_msg(
    solver: AbstractDirectLinearSolver | Normal,
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


def slogdet(
    operator: AbstractLinearOperator,
    solver: AbstractDirectLinearSolver | Normal,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
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

            Do **not** apply `lax.stop_gradient` to this state. The sign and
            log-absolute-determinant are computed directly from the factorisation,
            so gradients must flow through the state for `slogdet` to be
            differentiable with respect to the operator.

    **Returns:**

    A 2-tuple of `(sign, logabsdet)`. `sign` is `nan` when the solver cannot
    recover it cheaply (e.g. [`lineax.SVD`][] on a full-rank square matrix, or
    [`lineax.Normal`][]).
    """
    if options is None:
        options = {}
    if state is None:
        state = solver.init(operator, options)
    return solver.slogdet(state, options)


def determinant(
    operator: AbstractLinearOperator,
    solver: AbstractDirectLinearSolver | Normal,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
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

            Do **not** apply `lax.stop_gradient` to this state. The determinant
            is computed directly from the factorisation, so gradients must flow
            through the state for `determinant` to be differentiable with respect
            to the operator.

    - `throw`: if `True` (the default), raise an error when the sign of the
        determinant is not available (e.g. when using [`lineax.Normal`][] or
        [`lineax.SVD`][] on a full-rank square matrix). If `False`, a `nan`
        result is returned silently.

    **Returns:**

    A scalar array equal to the determinant.
    """
    if options is None:
        options = {}
    if state is None:
        state = solver.init(operator, options)
    sign, lad = solver.slogdet(state, options)
    det = sign * jnp.exp(lad)
    if throw:
        msg = _det_sign_error_msg(solver, operator)
        det = eqx.error_if(det, jnp.isnan(sign), msg)
    return det
