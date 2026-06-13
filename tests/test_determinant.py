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

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest

from .helpers import construct_matrix


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _make_op(matrix, tags=()):
    return lx.MatrixLinearOperator(matrix, tags)


# ----------------------------------------------------------------------------
# Square determinant correctness
# ----------------------------------------------------------------------------

SQUARE_DET_CASES = [
    (lx.LU(), ()),
    (lx.QR(), ()),
    (lx.Cholesky(), lx.positive_semidefinite_tag),
    (lx.Cholesky(), lx.negative_semidefinite_tag),
    (lx.Triangular(), lx.lower_triangular_tag),
    (lx.Triangular(), lx.upper_triangular_tag),
    (lx.Diagonal(), lx.diagonal_tag),
    (lx.Tridiagonal(), lx.tridiagonal_tag),
]


@pytest.mark.parametrize("solver,tags", SQUARE_DET_CASES)
def test_determinant_square(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = _make_op(matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    expected = jnp.linalg.det(matrix)
    assert jnp.allclose(det, expected, atol=1e-10), f"got {det}, expected {expected}"


@pytest.mark.parametrize("solver,tags", SQUARE_DET_CASES)
def test_slogdet_square(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = _make_op(matrix, tags)
    sign, lad = lx.slogdet(op, solver)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"
    # sign may be nan for Normal; only check for solvers that return it
    if not jnp.isnan(sign):
        assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"


# ----------------------------------------------------------------------------
# sign=nan for solvers that cannot recover it
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("solver,tags", [
    (lx.SVD(), ()),
    (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
])
def test_slogdet_sign_is_nan(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = _make_op(matrix, tags)
    sign, _ = lx.slogdet(op, solver)
    assert jnp.isnan(sign)


@pytest.mark.parametrize("solver,tags", [
    (lx.SVD(), ()),
    (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
])
def test_determinant_throw_true_raises(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = _make_op(matrix, tags)
    with pytest.raises(Exception):
        lx.determinant(op, solver, throw=True)


@pytest.mark.parametrize("solver,tags", [
    (lx.SVD(), ()),
    (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
])
def test_determinant_throw_false_nan(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = _make_op(matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    assert jnp.isnan(det)


# ----------------------------------------------------------------------------
# QR rectangular: lad correct, sign is ±1
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_lad(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    _, lad = lx.slogdet(op, lx.QR())
    s = jnp.linalg.svd(A, compute_uv=False)
    lad_svd = jnp.sum(jnp.log(s))
    assert jnp.allclose(lad, lad_svd, atol=1e-10), f"lad {lad} vs svd {lad_svd}"


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_sign_is_pm1(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, _ = lx.slogdet(op, lx.QR())
    assert not jnp.isnan(sign)
    assert jnp.allclose(jnp.abs(sign), 1.0, atol=1e-10)


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_sign_matches_explicit_qr(shape, getkey):
    """sign should match sign(det(Q_full)) * prod(sign(diag(R)))."""
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, _ = lx.slogdet(op, lx.QR())

    # For tall A: QR of A; for wide A: QR of A^T (matching lineax init)
    B = A if A.shape[0] >= A.shape[1] else A.T
    Q_full, R_full = jnp.linalg.qr(B, mode="complete")
    n = min(B.shape)
    R_sq = R_full[:n, :n]
    sign_ref = jnp.sign(jnp.linalg.det(Q_full)) * jnp.prod(jnp.sign(jnp.diag(R_sq)))
    sign_ref = sign_ref.astype(jnp.float64)
    assert jnp.allclose(sign, sign_ref, atol=1e-10), f"sign {sign} vs ref {sign_ref}"


# ----------------------------------------------------------------------------
# SVD slogdet: lad matches log-pseudodeterminant
# ----------------------------------------------------------------------------

def test_svd_slogdet_lad_fullrank(getkey):
    (matrix,) = construct_matrix(getkey, lx.SVD(), ())
    op = _make_op(matrix)
    _, lad = lx.slogdet(op, lx.SVD())
    s = jnp.linalg.svd(matrix, compute_uv=False)
    assert jnp.allclose(lad, jnp.sum(jnp.log(s)), atol=1e-10)


def test_svd_slogdet_lad_rankdeficient(getkey):
    """Rank-deficient matrix: lad = sum of log(nonzero singular values)."""
    matrix = jr.normal(getkey(), (3, 3), dtype=jnp.float64)
    matrix = matrix.at[0, :].set(0)  # rank 2
    op = _make_op(matrix)
    _, lad = lx.slogdet(op, lx.SVD())
    s = jnp.linalg.svd(matrix, compute_uv=False)
    s_nonzero = s[s > 1e-10]
    expected_lad = jnp.sum(jnp.log(s_nonzero))
    assert jnp.allclose(lad, expected_lad, atol=1e-8)


# ----------------------------------------------------------------------------
# state= kwarg: pre-computed state is reused correctly
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("solver,tags", [
    (lx.LU(), ()),
    (lx.QR(), ()),
])
def test_slogdet_state_kwarg(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = _make_op(matrix, tags)
    state = solver.init(op, {})
    sign1, lad1 = lx.slogdet(op, solver)
    sign2, lad2 = lx.slogdet(op, solver, state=state)
    assert jnp.allclose(sign1, sign2)
    assert jnp.allclose(lad1, lad2)


# ----------------------------------------------------------------------------
# JVP of slogdet: d(lad)/dA = trace(A^{-1} dA)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("solver,tags", [
    (lx.LU(), ()),
    (lx.QR(), ()),
])
def test_slogdet_jvp_lad(solver, tags, getkey):
    (matrix, t_matrix) = construct_matrix(getkey, solver, tags, num=2)

    def lad_fn(mat):
        op = lx.MatrixLinearOperator(mat)
        _, lad = lx.slogdet(op, solver)
        return lad

    _, lad_dot = jax.jvp(lad_fn, (matrix,), (t_matrix,))
    expected = jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    assert jnp.allclose(lad_dot, expected, atol=1e-8), (
        f"lad_dot {lad_dot} vs expected {expected}"
    )


def test_slogdet_jvp_matches_finite_difference(getkey):
    """JVP lad_dot matches finite-difference approximation."""
    (matrix, t_matrix) = construct_matrix(getkey, lx.LU(), (), num=2)

    def lad_fn(mat):
        op = lx.MatrixLinearOperator(mat)
        _, lad = lx.slogdet(op, lx.LU())
        return lad

    _, lad_dot = jax.jvp(lad_fn, (matrix,), (t_matrix,))

    eps = np.sqrt(np.finfo(np.float64).eps)
    lad_plus = lad_fn(matrix + eps * t_matrix)
    lad_minus = lad_fn(matrix - eps * t_matrix)
    fd_dot = (lad_plus - lad_minus) / (2 * eps)

    assert jnp.allclose(lad_dot, fd_dot, atol=1e-6), (
        f"jvp {lad_dot} vs fd {fd_dot}"
    )


def test_determinant_jvp(getkey):
    """JVP through determinant."""
    (matrix, t_matrix) = construct_matrix(getkey, lx.LU(), (), num=2)

    def det_fn(mat):
        op = lx.MatrixLinearOperator(mat)
        return lx.determinant(op, lx.LU())

    det, det_dot = jax.jvp(det_fn, (matrix,), (t_matrix,))

    # d(det(A))/dA_ij = det(A) * (A^{-T})_ij
    # det_dot = det(A) * trace(A^{-1} dA)
    expected_dot = det * jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    assert jnp.allclose(det_dot, expected_dot, atol=1e-8), (
        f"det_dot {det_dot} vs expected {expected_dot}"
    )
