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

import jax.numpy as jnp
from jaxtyping import Array

from ._operator import AbstractLinearOperator
from ._solve import AbstractDirectLinearSolver
from ._solver.normal import Normal


MaybeDirectLinearSolver: TypeAlias = AbstractDirectLinearSolver | Normal


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

    A 2-tuple of `(sign, logabsdet)`.
    """
    if options is None:
        options = {}
    state = solver.init(operator, options)
    return solver.det_sign(state, options), solver.logabsdet(state, options)


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
    return solver.logabsdet(state, options)


def determinant(
    operator: AbstractLinearOperator,
    solver: MaybeDirectLinearSolver,
    *,
    options: dict[str, Any] | None = None,
) -> Array:
    """Compute det(operator) using the given direct solver.

    **Arguments:**

    - `operator`: a linear operator.
    - `solver`: a [`lineax.MaybeDirectLinearSolver`][].
    - `options`: any extra options to pass to the solver.

    **Returns:**

    A scalar array equal to the determinant.
    """
    if options is None:
        options = {}
    state = solver.init(operator, options)
    sign = solver.det_sign(state, options)
    logabs = solver.logabsdet(state, options)
    return sign * jnp.exp(logabs)
