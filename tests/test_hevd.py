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


tol = 1e-10


def _hermitian(getkey, size, dtype):
    matrix = jr.normal(getkey(), (size, size), dtype=dtype)
    return matrix + matrix.conj().T


def _hermitian_with_spectrum(getkey, eigvals, dtype):
    """A Hermitian matrix `Q diag(eigvals) Q^H` with `Q` (real/complex) unitary.

    Lets us place exact zero eigenvalues *between* large eigenvalues of both signs,
    which (after eigh's ascending sort) is the case that stresses HEVD's
    magnitude-based masking and truncation -- the eigenpairs to discard are in the
    interior of the spectrum, not at a contiguous tail.
    """
    n = len(eigvals)
    q, _ = jnp.linalg.qr(jr.normal(getkey(), (n, n), dtype=dtype))
    d = jnp.asarray(eigvals, dtype=dtype)
    return (q * d[None, :]) @ q.conj().T


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_hevd_indefinite_rank_deficient(getkey, dtype):
    # Indefinite (both signs) AND rank-deficient (interior zeros).
    eigvals = [4.0, -3.0, 0.0, 0.0, 2.0]
    matrix = _hermitian_with_spectrum(getkey, eigvals, dtype)
    operator = lx.MatrixLinearOperator(matrix, lx.hermitian_tag)
    b = matrix @ jr.normal(getkey(), (5,), dtype=dtype)

    hevd = lx.linear_solve(operator, b, solver=lx.HEVD(), throw=False)
    lstsq, *_ = jnp.linalg.lstsq(matrix, b)
    assert tree_allclose(hevd.value, lstsq, atol=tol, rtol=tol)
    # Two interior eigenvalues are (numerically) zero -> effective rank 3.
    assert int(hevd.stats["rank"]) == 3


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_hevd_indefinite_max_rank_truncation(getkey, dtype):
    # The r=3 largest-magnitude eigenvalues are {4, -3, 2}: of *both* signs and not
    # contiguous in eigh's ascending order, so truncation must select by magnitude.
    eigvals = [4.0, -3.0, 0.0, 0.0, 2.0]
    matrix = _hermitian_with_spectrum(getkey, eigvals, dtype)
    b = matrix @ jr.normal(getkey(), (5,), dtype=dtype)

    operator = lx.MatrixLinearOperator(matrix, (lx.hermitian_tag, lx.MaxRankTag(3)))
    truncated = lx.linear_solve(operator, b, solver=lx.HEVD(), throw=False).value
    lstsq, *_ = jnp.linalg.lstsq(matrix, b)
    assert tree_allclose(truncated, lstsq, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_hevd_matches_svd_indefinite(getkey, dtype):
    # An indefinite Hermitian matrix (eigenvalues of both signs) is the case where
    # eigh's ascending ordering differs most from SVD's descending-magnitude one.
    matrix = _hermitian(getkey, 5, dtype)
    operator = lx.MatrixLinearOperator(matrix, lx.hermitian_tag)
    b = jr.normal(getkey(), (5,), dtype=dtype)

    hevd = lx.linear_solve(operator, b, solver=lx.HEVD()).value
    svd = lx.linear_solve(operator, b, solver=lx.SVD()).value
    true_x = jnp.linalg.solve(matrix, b)
    assert tree_allclose(hevd, true_x, atol=tol, rtol=tol)
    assert tree_allclose(hevd, svd, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_hevd_singular_pseudoinverse(getkey, dtype):
    # Rank-2 (in a 5x5) Hermitian operator: HEVD should return the pseudoinverse
    # (minimum-norm least-squares) solution, matching `jnp.linalg.lstsq`.
    factor = jr.normal(getkey(), (5, 2), dtype=dtype)
    matrix = factor @ factor.conj().T  # PSD, rank <= 2, Hermitian
    operator = lx.MatrixLinearOperator(matrix, lx.hermitian_tag)
    b = matrix @ jr.normal(getkey(), (5,), dtype=dtype)

    hevd = lx.linear_solve(operator, b, solver=lx.HEVD(), throw=False).value
    lstsq, *_ = jnp.linalg.lstsq(matrix, b)
    assert tree_allclose(hevd, lstsq, atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_hevd_rejects_non_hermitian(getkey, dtype):
    matrix = jr.normal(getkey(), (4, 4), dtype=dtype)
    operator = lx.MatrixLinearOperator(matrix)  # untagged, not Hermitian
    b = jr.normal(getkey(), (4,), dtype=dtype)
    with pytest.raises(ValueError, match="Hermitian"):
        lx.linear_solve(operator, b, solver=lx.HEVD())


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_hevd_max_rank_truncation(getkey, dtype):
    # Tagging `MaxRankTag(r)` statically truncates to the r largest-magnitude
    # eigenvalues, which (because eigh sorts ascending) live at *both* ends of the
    # spectrum, not contiguously -- the truncation must select by magnitude.
    factor = jr.normal(getkey(), (6, 3), dtype=dtype)
    matrix = factor @ factor.conj().T  # PSD rank <= 3
    b = matrix @ jr.normal(getkey(), (6,), dtype=dtype)

    operator = lx.MatrixLinearOperator(matrix, (lx.hermitian_tag, lx.MaxRankTag(3)))
    assert lx.max_rank(operator) == 3
    truncated = lx.linear_solve(operator, b, solver=lx.HEVD(), throw=False).value
    lstsq, *_ = jnp.linalg.lstsq(matrix, b)
    assert tree_allclose(truncated, lstsq, atol=tol, rtol=tol)


def test_hevd_max_rank_violation_errors(getkey):
    # A genuinely full-rank operator wrongly tagged low-rank should be caught.
    matrix = _hermitian(getkey, 4, jnp.float64)
    operator = lx.MatrixLinearOperator(matrix, (lx.hermitian_tag, lx.MaxRankTag(2)))
    b = jr.normal(getkey(), (4,), dtype=jnp.float64)
    with pytest.raises(Exception):
        sol = lx.linear_solve(operator, b, solver=lx.HEVD(), throw=True)
        jax.block_until_ready(sol.value)


def test_hevd_jit_and_grad(getkey):
    matrix = _hermitian(getkey, 4, jnp.float64)
    b = jr.normal(getkey(), (4,), dtype=jnp.float64)

    @eqx.filter_jit
    def solve(m, v):
        op = lx.MatrixLinearOperator(m, lx.hermitian_tag)
        return lx.linear_solve(op, v, solver=lx.HEVD()).value

    x = solve(matrix, b)
    assert tree_allclose(matrix @ x, b, atol=tol, rtol=tol)

    def loss(m):
        return jnp.sum(solve(m + m.T, b) ** 2)

    grad = jax.grad(loss)(matrix)
    assert jnp.all(jnp.isfinite(grad))


# --- is_hermitian / hermitian_tag semantics ---


def test_is_hermitian_real_vs_complex_symmetric(getkey):
    real = jr.normal(getkey(), (3, 3), dtype=jnp.float64)
    real = real + real.T
    real_op = lx.MatrixLinearOperator(real, lx.symmetric_tag)
    # For real operators, symmetric == Hermitian.
    assert lx.is_symmetric(real_op)
    assert lx.is_hermitian(real_op)

    comp = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)
    comp = comp + comp.T  # complex symmetric (A = A^T) but NOT Hermitian
    comp_op = lx.MatrixLinearOperator(comp, lx.symmetric_tag)
    assert lx.is_symmetric(comp_op)
    assert not lx.is_hermitian(comp_op)

    # A genuine complex Hermitian is Hermitian but not symmetric.
    herm = comp + comp.conj().T
    herm_op = lx.MatrixLinearOperator(herm, lx.hermitian_tag)
    assert lx.is_hermitian(herm_op)
    assert not lx.is_symmetric(herm_op)


def test_is_hermitian_diagonal():
    real = lx.DiagonalLinearOperator(jnp.array([1.0, 2.0, 3.0]))
    assert lx.is_symmetric(real)
    assert lx.is_hermitian(real)

    comp = lx.DiagonalLinearOperator(jnp.array([1.0 + 1j, 2.0]))
    # A complex diagonal is symmetric (A = A^T) but Hermitian only if real-valued.
    assert lx.is_symmetric(comp)
    assert not lx.is_hermitian(comp)


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.complex128))
def test_is_hermitian_semidefinite(getkey, dtype):
    m = jr.normal(getkey(), (3, 3), dtype=dtype)
    psd = lx.MatrixLinearOperator(m @ m.conj().T, lx.positive_semidefinite_tag)
    # PSD/NSD are Hermitian by definition for both real and complex dtypes.
    assert lx.is_hermitian(psd)


def test_hermitian_tag_preserved_under_transpose_and_invert():
    tags = frozenset({lx.hermitian_tag})
    assert lx.hermitian_tag in lx.transpose_tags(tags)
    assert lx.hermitian_tag in lx.invert_tags(tags)


def test_hermitian_preserved_under_real_scaling_only(getkey):
    herm = _hermitian(getkey, 3, jnp.complex128)
    op = lx.MatrixLinearOperator(herm, lx.hermitian_tag)
    # Negation and real scaling preserve Hermitian-ness...
    assert lx.is_hermitian(-op)
    assert lx.is_hermitian(op * 2.0)
    assert lx.is_hermitian(op / 2.0)
    # ...but a complex scalar does not, since (cA)^H = conj(c) A != cA.
    assert not lx.is_hermitian(op * (1.0 + 1j))


def test_hermitian_preserved_under_addition(getkey):
    a = lx.MatrixLinearOperator(_hermitian(getkey, 3, jnp.complex128), lx.hermitian_tag)
    b = lx.MatrixLinearOperator(_hermitian(getkey, 3, jnp.complex128), lx.hermitian_tag)
    assert lx.is_hermitian(a + b)
