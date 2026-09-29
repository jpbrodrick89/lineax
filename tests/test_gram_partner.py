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

"""Tests for the gram solve used by `_linear_solve_jvp`.

For a tall operator, the JVP of a least-squares solve has a term `(AᴴA)⁺ w`. A tall QR
or `Normal` computes it with a single solve against the gram matrix `AᴴA`, reusing its
factorisation, rather than the generic adjoint solve. The standard JVP suites are
square-only, so they never reach it: these tests check the tall case directly, against
the pseudoinverse.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import tree_allclose


_tall_solvers = (lx.QR(), lx.Normal(lx.Cholesky()), lx.Normal(lx.SVD()))


def _solve(solver, vector):
    def fn(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        return lx.linear_solve(operator, vector, solver).value

    return fn


def _reference(vector):
    return lambda matrix: jnp.linalg.pinv(matrix) @ vector


@pytest.mark.parametrize("solver", _tall_solvers)
@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_gram_solve_jvp(solver, dtype, getkey):
    matrix = jr.normal(getkey(), (6, 3), dtype=dtype)
    vector = jr.normal(getkey(), (6,), dtype=dtype)
    t_matrix = jr.normal(getkey(), (6, 3), dtype=dtype)
    _, t_out = jax.jvp(_solve(solver, vector), (matrix,), (t_matrix,))
    _, t_expected = jax.jvp(_reference(vector), (matrix,), (t_matrix,))
    assert tree_allclose(t_out, t_expected, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("solver", _tall_solvers)
def test_gram_solve_second_order(solver, getkey):
    matrix = jr.normal(getkey(), (6, 3))
    vector = jr.normal(getkey(), (6,))
    t_matrix = jr.normal(getkey(), (6, 3))

    def second(fn):
        jvp = lambda m: jax.jvp(fn, (m,), (t_matrix,))[1]
        return jax.jvp(jvp, (matrix,), (t_matrix,))[1]

    expected = second(_reference(vector))
    assert tree_allclose(second(_solve(solver, vector)), expected, rtol=1e-6)


def test_gram_solve_grad_of_grad(getkey):
    # The QR gram solve's `Cholesky` state once held `is_nsd` as a scalar array. Under
    # grad-of-grad, that is inlined into the transposed jaxpr as a literal, so it
    # reached the `linear_solve` transpose rule as a Python bool, and a different
    # number of dynamic inputs than were bound.
    matrix = jr.normal(getkey(), (6, 3))
    vector = jr.normal(getkey(), (6,))
    weights = jr.normal(getkey(), (6, 3))

    def grad_of_grad(fn):
        loss = lambda m: jnp.sum(fn(m) ** 3)
        return jax.grad(lambda m: jnp.sum(jax.grad(loss)(m) * weights))(matrix)

    expected = grad_of_grad(_reference(vector))
    assert tree_allclose(grad_of_grad(_solve(lx.QR(), vector)), expected, rtol=1e-6)
