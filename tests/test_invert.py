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

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from .helpers import tree_allclose


def _well_conditioned_matrix(getkey, size=3, dtype=jnp.float64):
    """Generate a well-conditioned random matrix."""
    while True:
        matrix = jr.normal(getkey(), (size, size), dtype=dtype)
        if jnp.linalg.cond(matrix) < 100:
            return matrix


def _well_conditioned_psd_matrix(getkey, size=3, dtype=jnp.float64):
    """Generate a well-conditioned PSD matrix."""
    matrix = _well_conditioned_matrix(getkey, size, dtype)
    return matrix @ matrix.T.conj()


# -- Validation --


def test_square_only_validation():
    """invert rejects non-square operators."""
    matrix = jnp.ones((3, 4))
    op = lx.FunctionLinearOperator(
        lambda x: matrix @ x,
        jax.ShapeDtypeStruct((4,), jnp.float64),
    )
    with pytest.raises(ValueError, match="square"):
        lx.invert(op)


def test_well_posed_false_validation():
    """invert rejects well_posed=False."""
    op = lx.MatrixLinearOperator(jnp.eye(3))
    with pytest.raises(ValueError, match="well-posed"):
        lx.invert(op, solver=lx.AutoLinearSolver(well_posed=False))


def test_well_posed_none_validation():
    """invert rejects well_posed=None."""
    op = lx.MatrixLinearOperator(jnp.eye(3))
    with pytest.raises(ValueError, match="well-posed"):
        lx.invert(op, solver=lx.AutoLinearSolver(well_posed=None))


def test_assume_full_rank_validation():
    """invert rejects solvers that don't assume full rank."""
    op = lx.MatrixLinearOperator(jnp.eye(3))
    with pytest.raises(ValueError, match="full rank"):
        lx.invert(op, solver=lx.SVD())


# -- Core behaviour --


def test_mv(getkey):
    """invert(A).mv(v) solves A x = v."""
    matrix = _well_conditioned_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix)
    inv_op = lx.invert(op)
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    result = inv_op.mv(vec)
    expected = jnp.linalg.solve(matrix, vec)
    assert tree_allclose(result, expected, atol=1e-10)


def test_composition_identity(getkey):
    """(invert(A) @ A).mv(v) ~ v."""
    matrix = _well_conditioned_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix)
    inv_op = lx.invert(op)
    composed = inv_op @ op
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    result = composed.mv(vec)
    assert tree_allclose(result, vec, atol=1e-10)


def test_double_inverse(getkey):
    """invert(invert(A)).mv(v) ~ A.mv(v)."""
    matrix = _well_conditioned_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix)
    double_inv = lx.invert(lx.invert(op))
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    result = double_inv.mv(vec)
    expected = matrix @ vec
    assert tree_allclose(result, expected, atol=1e-8)


# -- Explicit solver tests --


def test_solver_cholesky(getkey):
    """Works with Cholesky solver for PSD matrices."""
    matrix = _well_conditioned_psd_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
    inv_op = lx.invert(op, solver=lx.Cholesky())
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    result = inv_op.mv(vec)
    expected = jnp.linalg.solve(matrix, vec)
    assert tree_allclose(result, expected, atol=1e-10)


def test_solver_cg(getkey):
    """Works with CG (iterative) solver for PSD matrices."""
    matrix = _well_conditioned_psd_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix, lx.positive_semidefinite_tag)
    inv_op = lx.invert(op, solver=lx.CG(rtol=1e-12, atol=1e-12))
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    result = inv_op.mv(vec)
    expected = jnp.linalg.solve(matrix, vec)
    assert tree_allclose(result, expected, atol=1e-8)


# -- Diagonal optimisation --


def test_diagonal_optimisation(getkey):
    """diagonal(invert(op)) uses O(n) path for diagonal inner operators."""
    diag = jr.normal(getkey(), (5,), dtype=jnp.float64)
    diag = jnp.where(jnp.abs(diag) < 0.1, 1.0, diag)
    op = lx.DiagonalLinearOperator(diag)
    inv_op = lx.invert(op)
    vec = jr.normal(getkey(), (5,), dtype=jnp.float64)
    result = inv_op.mv(vec)
    expected = vec / diag
    assert tree_allclose(result, expected, atol=1e-10)


# -- vmap --


def test_vmap(getkey):
    """vmap over invert(A).mv works correctly."""
    matrix = _well_conditioned_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix)
    inv_op = lx.invert(op)
    vecs = jr.normal(getkey(), (5, 3), dtype=jnp.float64)
    result = jax.vmap(inv_op.mv)(vecs)
    expected = jax.vmap(lambda v: jnp.linalg.solve(matrix, v))(vecs)
    assert tree_allclose(result, expected, atol=1e-10)


# -- AD --


def test_grad_wrt_vector(getkey):
    """VJP through invert(A).mv(v) wrt vector."""
    matrix = _well_conditioned_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix)
    inv_op = lx.invert(op)

    def f(vec):
        return jnp.sum(inv_op.mv(vec))

    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    grad = jax.grad(f)(vec)
    expected = jnp.linalg.solve(matrix.T, jnp.ones(3, dtype=jnp.float64))
    assert tree_allclose(grad, expected, atol=1e-10)


def test_jvp_wrt_vector(getkey):
    """JVP through invert(A).mv(v) wrt vector."""
    matrix = _well_conditioned_matrix(getkey)
    op = lx.MatrixLinearOperator(matrix)
    inv_op = lx.invert(op)

    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)
    t_vec = jr.normal(getkey(), (3,), dtype=jnp.float64)

    primals, tangents = eqx.filter_jvp(inv_op.mv, (vec,), (t_vec,))
    expected_primals = jnp.linalg.solve(matrix, vec)
    expected_tangents = jnp.linalg.solve(matrix, t_vec)
    assert tree_allclose(primals, expected_primals, atol=1e-10)
    assert tree_allclose(tangents, expected_tangents, atol=1e-10)


def test_grad_wrt_operator(getkey):
    """VJP through invert(A).mv(v) wrt the inner matrix."""
    matrix = _well_conditioned_matrix(getkey)
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)

    def f_inv(mat):
        op = lx.MatrixLinearOperator(mat)
        inv_op = lx.invert(op)
        return jnp.sum(inv_op.mv(vec))

    def f_jnp(mat):
        return jnp.sum(jnp.linalg.solve(mat, vec))

    grad_inv = jax.grad(f_inv)(matrix)
    grad_jnp = jax.grad(f_jnp)(matrix)
    assert tree_allclose(grad_inv, grad_jnp, atol=1e-8)


def test_jvp_wrt_operator(getkey):
    """JVP through invert(A).mv(v) wrt the inner matrix."""
    matrix = _well_conditioned_matrix(getkey)
    t_matrix = jr.normal(getkey(), (3, 3), dtype=jnp.float64)
    vec = jr.normal(getkey(), (3,), dtype=jnp.float64)

    def f_inv(mat):
        op = lx.MatrixLinearOperator(mat)
        inv_op = lx.invert(op)
        return inv_op.mv(vec)

    def f_jnp(mat):
        return jnp.linalg.solve(mat, vec)

    out, t_out = eqx.filter_jvp(f_inv, (matrix,), (t_matrix,))
    expected_out, expected_t_out = eqx.filter_jvp(f_jnp, (matrix,), (t_matrix,))
    assert tree_allclose(out, expected_out, atol=1e-10)
    assert tree_allclose(t_out, expected_t_out, atol=1e-8)
