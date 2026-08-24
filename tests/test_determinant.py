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
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest
from lineax._solver.tridiagonal import _slogdet_scan, _slogdet_tree_reduce

from .helpers import (
    construct_matrix,
    make_jac_operator,
    make_matrix_operator,
)


# ----------------------------------------------------------------------------
# Square determinant and slogdet: correctness vs jnp.linalg
# Parametrised over operator type to exercise both direct matrix storage
# and the as_matrix() materialisation path (JacobianLinearOperator).
# ----------------------------------------------------------------------------

SQUARE_DET_CASES = [
    (lx.LU(), ()),
    (lx.QR(), ()),
    (lx.Cholesky(), lx.positive_semidefinite_tag),
    (lx.Cholesky(), lx.negative_semidefinite_tag),
    (lx.Triangular(), lx.lower_triangular_tag),
    (lx.Triangular(), lx.upper_triangular_tag),
    (lx.Diagonal(well_posed=True), lx.diagonal_tag),
    (lx.Diagonal(well_posed=False), lx.diagonal_tag),
    (lx.Tridiagonal(), lx.tridiagonal_tag),
    (lx.HEVD(), lx.symmetric_tag),
    (lx.Circulant(well_posed=True), lx.circulant_tag),
    (lx.Circulant(well_posed=False), lx.circulant_tag),
    (lx.AutoLinearSolver(well_posed=True), ()),
    (lx.AutoLinearSolver(well_posed=None), ()),
]

# Complex analogue. `symmetric_tag` would build a complex-*symmetric* (non-Hermitian)
# matrix, which `HEVD` rejects, so the Hermitian solvers use `hermitian_tag` /
# `positive_semidefinite_tag` instead. These exercise the complex sign paths -- most
# notably the `QR` complex-Householder sign and the `Circulant` FFT determinant -- which
# the real cases cannot reach.
COMPLEX_DET_CASES = [
    (lx.LU(), ()),
    (lx.QR(), ()),
    (lx.Cholesky(), lx.positive_semidefinite_tag),
    (lx.Triangular(), lx.lower_triangular_tag),
    (lx.Triangular(), lx.upper_triangular_tag),
    (lx.Diagonal(well_posed=True), lx.diagonal_tag),
    (lx.Diagonal(well_posed=False), lx.diagonal_tag),
    (lx.Tridiagonal(), lx.tridiagonal_tag),
    (lx.HEVD(), lx.hermitian_tag),
    (lx.Circulant(well_posed=True), lx.circulant_tag),
    (lx.Circulant(well_posed=False), lx.circulant_tag),
    (lx.AutoLinearSolver(well_posed=True), ()),
    (lx.AutoLinearSolver(well_posed=None), ()),
]


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", SQUARE_DET_CASES)
def test_determinant_square(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = make_operator(getkey, matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    expected = jnp.linalg.det(matrix)
    assert jnp.allclose(det, expected, atol=1e-10), f"got {det}, expected {expected}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", SQUARE_DET_CASES)
def test_slogdet_square(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = make_operator(getkey, matrix, tags)
    sign, lad = lx.slogdet(op, solver)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"
    if not jnp.isnan(sign):
        assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"


def test_default_solver(getkey):
    # `determinant`/`slogdet` default to `AutoLinearSolver(well_posed=True)`, matching
    # `linear_solve`, so a solver need not be passed explicitly.
    (matrix,) = construct_matrix(getkey, lx.LU(), ())
    op = lx.MatrixLinearOperator(matrix)
    assert jnp.allclose(lx.determinant(op), jnp.linalg.det(matrix), atol=1e-10)
    sign, lad = lx.slogdet(op)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, atol=1e-10)


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", COMPLEX_DET_CASES)
def test_determinant_square_complex(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=jnp.complex128)
    op = make_operator(getkey, matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    expected = jnp.linalg.det(matrix)
    assert jnp.allclose(det, expected, atol=1e-10), f"got {det}, expected {expected}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize("solver,tags", COMPLEX_DET_CASES)
def test_slogdet_square_complex(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags, dtype=jnp.complex128)
    op = make_operator(getkey, matrix, tags)
    sign, lad = lx.slogdet(op, solver)
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"
    if not jnp.isnan(sign):
        assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"


# ----------------------------------------------------------------------------
# Normal(Cholesky): sign=nan, lad = sum(log(singular values)) for rectangular A
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_normal_cholesky_slogdet_rectangular(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, lad = lx.slogdet(op, lx.Normal(lx.Cholesky()))
    assert jnp.isnan(sign)
    s = jnp.linalg.svd(A, compute_uv=False)
    assert jnp.allclose(lad, jnp.sum(jnp.log(s)), atol=1e-8)


# ----------------------------------------------------------------------------
# sign=nan: throw kwarg
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.SVD(), ()),
        (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
    ],
)
def test_slogdet_sign_is_nan(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = lx.MatrixLinearOperator(matrix, tags)
    sign, _ = lx.slogdet(op, solver)
    assert jnp.isnan(sign)


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.SVD(), ()),
        (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
    ],
)
def test_determinant_throw_true_raises(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = lx.MatrixLinearOperator(matrix, tags)
    with pytest.raises(Exception):
        lx.determinant(op, solver, throw=True)


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.SVD(), ()),
        (lx.Normal(lx.Cholesky()), lx.positive_semidefinite_tag),
    ],
)
def test_determinant_throw_false_nan(solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)
    op = lx.MatrixLinearOperator(matrix, tags)
    det = lx.determinant(op, solver, throw=False)
    assert jnp.isnan(det)


# ----------------------------------------------------------------------------
# SVD slogdet: log-pseudodeterminant
# ----------------------------------------------------------------------------


def test_svd_slogdet_lad_fullrank(getkey):
    (matrix,) = construct_matrix(getkey, lx.SVD(), ())
    op = lx.MatrixLinearOperator(matrix)
    _, lad = lx.slogdet(op, lx.SVD())
    _, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(lad, ref_lad, atol=1e-10)


def test_diagonal_slogdet_rankdeficient(getkey):
    """Diagonal(well_posed=False): zero entry excluded from pseudodeterminant."""
    diag = jr.normal(getkey(), (4,), dtype=jnp.float64)
    diag = diag.at[1].set(0.0)
    op = lx.DiagonalLinearOperator(diag)
    sign, lad = lx.slogdet(op, lx.Diagonal(well_posed=False))
    nonzero = diag[jnp.abs(diag) > 1e-10]
    assert jnp.allclose(sign, jnp.prod(jnp.sign(nonzero)).real, atol=1e-10)
    assert jnp.allclose(lad, jnp.sum(jnp.log(jnp.abs(nonzero))), atol=1e-10)


def test_svd_slogdet_lad_rankdeficient(getkey):
    """Rank-deficient: lad = sum of log(nonzero singular values)."""
    matrix = jr.normal(getkey(), (3, 3), dtype=jnp.float64)
    matrix = matrix.at[0, :].set(0)
    op = lx.MatrixLinearOperator(matrix)
    _, lad = lx.slogdet(op, lx.SVD())
    s = jnp.linalg.svd(matrix, compute_uv=False)
    s_nonzero = s[s > 1e-10]
    assert jnp.allclose(lad, jnp.sum(jnp.log(s_nonzero)), atol=1e-8)


# ----------------------------------------------------------------------------
# Tridiagonal slogdet: the division-free minor recurrence
# ----------------------------------------------------------------------------


def test_tridiagonal_slogdet_singular_leading_minor():
    """An invertible operator whose leading principal minor is singular.

    `d[0] == 0` makes the LU-pivot recurrence divide by zero on the very first
    step; the minor recurrence is division-free and handles it.
    """
    matrix = jnp.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]], dtype=jnp.float64
    )
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10), f"sign: {sign} vs {ref_sign}"
    assert jnp.allclose(lad, ref_lad, atol=1e-10), f"lad: {lad} vs {ref_lad}"


@pytest.mark.parametrize("n", [1, 2, 3, 16, 17, 33, 256])
def test_tridiagonal_slogdet_block_boundaries(n, getkey):
    """Sizes either side of the renormalisation block length, plus n=1."""
    key = getkey()
    diagonal = jr.normal(key, (n,), dtype=jnp.float64) + 4.0
    off = jr.normal(getkey(), (2, max(n - 1, 0)), dtype=jnp.float64) * 0.5
    matrix = (
        jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
        if n > 1
        else jnp.diag(diagonal)
    )
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)


def test_tridiagonal_slogdet_no_overflow():
    """A large well-conditioned operator: the raw minors would overflow float64.

    det of this operator is ~exp(1400), so an unscaled three-term recurrence
    returns `inf`. The per-block renormalisation keeps `lad` finite and exact.
    """
    n = 1024
    diagonal = jnp.full((n,), 4.0, dtype=jnp.float64)
    off = jnp.full((n - 1,), 0.5, dtype=jnp.float64)
    matrix = jnp.diag(diagonal) + jnp.diag(off, -1) + jnp.diag(off, 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    _, lad = lx.slogdet(op, lx.Tridiagonal())
    _, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.isfinite(lad)
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)


@pytest.mark.parametrize("scale", [1e-60, 1e-20, 1e20, 1e60])
def test_tridiagonal_slogdet_badly_scaled(scale, getkey):
    """Uniformly scaled operators: the up-front power-of-two prescale handles these.

    Without it the minors leave float range inside a single block, since they grow
    or decay by roughly one entry-magnitude per step.
    """
    n = 128
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0) * scale
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float64) * 0.5 * scale
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    # `jnp.linalg.slogdet` is itself fine here; only `det` would overflow.
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.isfinite(lad)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-12)


@pytest.mark.parametrize("span", [12.0, 40.0])
def test_tridiagonal_slogdet_graded(span, getkey):
    """Entries spanning 10**-span within one operator.

    The renormalisation block length caps this: a block underflows once the minors
    decay past float range within it. At `_SLOGDET_BLOCK = 4` the measured limit is
    `span ~ 63`, so 40 has comfortable margin -- but raising the block length to 8
    would drop the limit to 38 and fail this case. See the block comment beside
    `_SLOGDET_BLOCK`, which also records the silently-wrong band just above the limit.
    """
    n = 128
    grade = 10.0 ** (-span * jnp.arange(n, dtype=jnp.float64) / n)
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0) * grade
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float64) * 0.5 * grade[None, :-1]
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-12)


def test_tridiagonal_slogdet_graded_float32(getkey):
    """As `test_tridiagonal_slogdet_graded`, in float32.

    The limit on tolerated grading is set by the exponent range, so float32 divides it
    by about eight: 8 decades rather than 63 at `_SLOGDET_BLOCK = 4`. 5 leaves margin.
    """
    n = 128
    span = 5.0
    grade = (10.0 ** (-span * jnp.arange(n, dtype=jnp.float32) / n)).astype(jnp.float32)
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float32) + 4.0) * grade
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float32) * 0.5 * grade[None, :-1]
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    _, lad = lx.slogdet(op, lx.Tridiagonal())
    _, ref_lad = jnp.linalg.slogdet(matrix.astype(jnp.float64))
    assert jnp.allclose(lad, ref_lad.astype(jnp.float32), rtol=1e-5)


@pytest.mark.parametrize("power", [0, 40, 77, 150])
def test_tridiagonal_slogdet_asymmetric_off_diagonals(power):
    """Off-diagonals of wildly different sizes but a coupling of exactly 1.

    The minor recurrence only ever sees `lower * upper`, so this operator is a
    permuted identity: its determinant is exactly 1 for every `power`. The prescale
    therefore has to measure the coupling as `sqrt(|l u|)` -- taking `max(|l|, |u|)`
    instead rescales by 10**power too much and underflows the whole recurrence.
    """
    n = 6
    diagonal = jnp.ones(n, dtype=jnp.float64)
    lower = jnp.full(n - 1, 10.0**power, dtype=jnp.float64)
    upper = jnp.full(n - 1, 10.0**-power, dtype=jnp.float64)
    op = lx.TridiagonalLinearOperator(diagonal, lower, upper)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    assert sign == 1
    assert jnp.allclose(lad, 0.0, atol=1e-12)


@pytest.mark.parametrize("span", [0.0, 40.0, 50.0])
def test_tridiagonal_slogdet_graded_jvp(span, getkey):
    """The gradient has to tolerate the same grading as the primal.

    The renormalisation scale cancels out of `lad` exactly, so its tangent is zero --
    but only if we say so. Letting AD differentiate it costs a `1/scale**2`, which
    overflows once a block's scale falls below sqrt(smallest normal), i.e. at half
    the grading the primal handles. That put the cliff at `span = 40`, which is
    exactly what `test_tridiagonal_slogdet_graded` pins.
    """
    n = 64
    grade = 10.0 ** (-span * jnp.arange(n, dtype=jnp.float64) / n)
    diagonal = (jr.normal(getkey(), (n,), dtype=jnp.float64) + 4.0) * grade
    off = jr.normal(getkey(), (2, n - 1), dtype=jnp.float64) * 0.5 * grade[None, :-1]
    matrix = jnp.diag(diagonal) + jnp.diag(off[0], -1) + jnp.diag(off[1], 1)

    def lad_of(m):
        return lx.slogdet(lx.MatrixLinearOperator(m, lx.tridiagonal_tag))[1]

    grad = jax.grad(lad_of)(matrix)
    assert jnp.all(jnp.isfinite(grad))
    # d(log|det A|)/dA = (A^-1)^T, of which only the band is meaningful here.
    expected = jnp.linalg.inv(matrix).T
    for k in (-1, 0, 1):
        assert jnp.allclose(jnp.diag(grad, k), jnp.diag(expected, k), rtol=1e-8)


def test_tridiagonal_slogdet_implementations_agree(getkey):
    """`lax.platform_dependent` runs only one implementation per platform.

    So CI on CPU never exercises `_slogdet_tree_reduce` and CI on GPU never exercises
    `_slogdet_scan`, and nothing otherwise compares them. Call both directly on
    whatever device is to hand: they compute the same recurrence and must agree.
    """
    for n in (1, 2, 3, 5, 16, 17, 64, 129):
        diagonal = jr.normal(getkey(), (n,), dtype=jnp.float64) + 3.0
        lower = jr.normal(getkey(), (n - 1,), dtype=jnp.float64)
        upper = jr.normal(getkey(), (n - 1,), dtype=jnp.float64)
        matrix = jnp.diag(diagonal) + jnp.diag(lower, -1) + jnp.diag(upper, 1)
        _, ref_lad = jnp.linalg.slogdet(matrix)
        op = lx.TridiagonalLinearOperator(diagonal, lower, upper)
        seq = _slogdet_scan(diagonal, lower, upper)
        assert jnp.allclose(seq[1], ref_lad, rtol=1e-10), f"sequential, n={n}"
        if n > 1:
            # `_slogdet_tree_reduce` is only reached for n > 1; `slogdet` special-cases
            # the 1x1 operator before dispatching.
            par = _slogdet_tree_reduce(diagonal, lower, upper)
            assert jnp.allclose(par[1], ref_lad, rtol=1e-10), f"parallel, n={n}"
            assert seq[0] == par[0], f"sign disagreement, n={n}"
        # And the shipped entry point agrees with both on this platform.
        assert jnp.allclose(lx.slogdet(op, lx.Tridiagonal())[1], ref_lad, rtol=1e-10)


def test_tridiagonal_slogdet_singular():
    """A singular operator gives `(0, -inf)`, not `nan`.

    Only for this operator, not in general: the division-free minor recurrence is not
    backward stable when the answer is exactly zero, so a singular operator whose zero
    determinant arises from cancellation across many steps can come out as a small
    finite value instead, and the two implementations need not agree on which. Detecting
    exact singularity is not something this solver promises; use a rank-revealing
    factorisation (`lx.SVD`) if that is what you need.
    """
    matrix = jnp.array(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )
    op = lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)
    sign, lad = lx.slogdet(op, lx.Tridiagonal())
    assert sign == 0
    assert lad == -jnp.inf


# ----------------------------------------------------------------------------
# QR rectangular: lad, sign=±1, sign vs explicit full-QR
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_lad(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    _, lad = lx.slogdet(op, lx.QR())
    _, ref_lad = jnp.linalg.slogdet(A.T @ A if shape[0] >= shape[1] else A @ A.T)
    assert jnp.allclose(lad, 0.5 * ref_lad, atol=1e-10)


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_sign_is_pm1(shape, getkey):
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, _ = lx.slogdet(op, lx.QR())
    assert not jnp.isnan(sign)
    assert jnp.allclose(jnp.abs(sign), 1.0, atol=1e-10)


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)])
def test_qr_rectangular_sign_vs_full_qr(shape, getkey):
    """sign matches sign(det(Q_full)) * prod(sign(diag(R))) via jnp.linalg.qr."""
    A = jr.normal(getkey(), shape, dtype=jnp.float64)
    op = lx.MatrixLinearOperator(A)
    sign, _ = lx.slogdet(op, lx.QR())

    # lineax QR decomposes A directly (tall) or A^T (wide)
    B = A if A.shape[0] >= A.shape[1] else A.T
    Q_full, R_full = jnp.linalg.qr(B, mode="complete")
    n = min(B.shape)
    R_sq = R_full[:n, :n]
    sign_ref = (
        jnp.sign(jnp.linalg.det(Q_full)) * jnp.prod(jnp.sign(jnp.diag(R_sq)))
    ).astype(jnp.float64)
    assert jnp.allclose(sign, sign_ref, atol=1e-10), f"sign {sign} vs ref {sign_ref}"


# ----------------------------------------------------------------------------
# JVP and grad: validated against jax.jvp/grad of jnp.linalg.slogdet.
# Parametrised over operator type: make_jac_operator exercises the AD path
# through TangentLinearOperator.as_matrix() for a JacobianLinearOperator.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize(
    "solver,tags,use_state",
    [
        (lx.LU(), (), False),
        (lx.LU(), (), True),
        (lx.QR(), (), False),
    ],
)
def test_slogdet_jvp_lad(make_operator, solver, tags, use_state, getkey):
    (matrix, t_matrix) = construct_matrix(getkey, solver, tags, num=2)

    def lad_lx(mat):
        op = make_operator(getkey, mat, tags)
        if use_state:
            op_dyn, op_st = eqx.partition(op, eqx.is_inexact_array)
            op_stopped = eqx.combine(lax.stop_gradient(op_dyn), op_st)
            state = solver.init(op_stopped, {})
            _, lad = lx.slogdet(op, solver, state=state)
        else:
            _, lad = lx.slogdet(op, solver)
        return lad

    def lad_jax(mat):
        return jnp.linalg.slogdet(mat)[1]

    _, lad_dot_lx = jax.jvp(lad_lx, (matrix,), (t_matrix,))
    _, lad_dot_jax = jax.jvp(lad_jax, (matrix,), (t_matrix,))
    assert jnp.allclose(lad_dot_lx, lad_dot_jax, atol=1e-8), (
        f"lad_dot {lad_dot_lx} vs jax {lad_dot_jax}"
    )


@pytest.mark.parametrize("solver", (lx.LU(), lx.QR()))
def test_slogdet_jvp_complex(solver, getkey):
    # For complex `A`, `log det A = log|det A| + i*arg(det A)`, so the tangent of the
    # complex `sign = det/|det|` is non-trivial. This exercises the `sign_dot` branch of
    # the custom JVP (dormant for real inputs) against `jnp.linalg.slogdet`.
    matrix = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)
    t_matrix = jr.normal(getkey(), (3, 3), dtype=jnp.complex128)

    def slogdet_lx(mat):
        return lx.slogdet(lx.MatrixLinearOperator(mat), solver)

    def slogdet_jax(mat):
        return jnp.linalg.slogdet(mat)

    (s_lx, l_lx), (sd_lx, ld_lx) = jax.jvp(slogdet_lx, (matrix,), (t_matrix,))
    (s_jax, l_jax), (sd_jax, ld_jax) = jax.jvp(slogdet_jax, (matrix,), (t_matrix,))
    assert jnp.allclose(l_lx, l_jax, atol=1e-8), f"lad {l_lx} vs {l_jax}"
    assert jnp.allclose(ld_lx, ld_jax, atol=1e-8), f"lad_dot {ld_lx} vs {ld_jax}"
    assert jnp.allclose(s_lx, s_jax, atol=1e-8), f"sign {s_lx} vs {s_jax}"
    assert jnp.allclose(sd_lx, sd_jax, atol=1e-8), f"sign_dot {sd_lx} vs {sd_jax}"


@pytest.mark.parametrize("make_operator", (make_matrix_operator, make_jac_operator))
@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.LU(), ()),
        (lx.QR(), ()),
    ],
)
def test_slogdet_grad(make_operator, solver, tags, getkey):
    (matrix,) = construct_matrix(getkey, solver, tags)

    def lad_lx(mat):
        op = make_operator(getkey, mat, tags)
        return lx.slogdet(op, solver)[1]

    def lad_jax(mat):
        return jnp.linalg.slogdet(mat)[1]

    grad_lx = jax.grad(lad_lx)(matrix)
    grad_jax = jax.grad(lad_jax)(matrix)
    assert jnp.allclose(grad_lx, grad_jax, atol=1e-8), (
        f"max diff {jnp.max(jnp.abs(grad_lx - grad_jax))}"
    )


def test_slogdet_grad_singular_pseudodet(getkey):
    # A rank-deficient Hermitian operator: differentiating the log-pseudodeterminant
    # via HEVD (a pseudoinverse solver) must succeed without raising, even though the
    # JVP's internal tangent solves use `throw=True`. The derivative is `trace(A⁺ dA)`,
    # which is finite despite `A` being singular.
    n, r = 5, 3
    factor = jr.normal(getkey(), (n, r), dtype=jnp.float64)
    matrix = factor @ factor.T  # symmetric PSD, rank r < n -> singular

    def lad_lx(mat):
        op = lx.MatrixLinearOperator(mat, lx.symmetric_tag)
        return lx.slogdet(op, lx.HEVD())[1]

    grad = jax.grad(lad_lx)(matrix)
    assert jnp.all(jnp.isfinite(grad)), grad


# ----------------------------------------------------------------------------
# Second-order AD: JVP of JVP, compared against jnp.linalg.slogdet
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver,tags",
    [
        (lx.LU(), ()),
        (lx.QR(), ()),
    ],
)
def test_slogdet_jvp_jvp(solver, tags, getkey):
    (matrix, t1, t2) = construct_matrix(getkey, solver, tags, num=3)

    def lad_lx(mat):
        return lx.slogdet(lx.MatrixLinearOperator(mat), solver)[1]

    def lad_jax(mat):
        return jnp.linalg.slogdet(mat)[1]

    inner_lx = lambda m: jax.jvp(lad_lx, (m,), (t1,))[1]
    inner_jax = lambda m: jax.jvp(lad_jax, (m,), (t1,))[1]

    _, dot2_lx = jax.jvp(inner_lx, (matrix,), (t2,))
    _, dot2_jax = jax.jvp(inner_jax, (matrix,), (t2,))
    assert jnp.allclose(dot2_lx, dot2_jax, atol=1e-6), (
        f"jvp_jvp {dot2_lx} vs jax {dot2_jax}"
    )


def _structured_case(kind, n, key, complex_):
    """(operator, tangent_operator, solver) for each structure with a fast JVP."""
    dtype = jnp.complex128 if complex_ else jnp.float64
    keys = jr.split(key, 2)
    if kind == "diagonal":
        make = lx.DiagonalLinearOperator
        args = lambda k: (jr.normal(k, (n,), dtype=dtype) + 4.0,)  # noqa: E731
        solver = lx.Diagonal(well_posed=True)
    elif kind == "tridiagonal":
        make = lx.TridiagonalLinearOperator
        args = lambda k: (  # noqa: E731
            jr.normal(jr.split(k)[0], (n,), dtype=dtype) + 4.0,
            jr.normal(jr.split(k)[1], (n - 1,), dtype=dtype) * 0.3,
            jr.normal(k, (n - 1,), dtype=dtype) * 0.3,
        )
        solver = lx.Tridiagonal()
    elif kind == "circulant":
        make = lx.CirculantLinearOperator
        args = lambda k: (  # noqa: E731
            jr.normal(k, (n,), dtype=dtype).at[0].add(n),
        )
        solver = lx.Circulant(well_posed=True)
    elif kind == "triangular":

        def make(matrix):
            return lx.MatrixLinearOperator(matrix, lx.lower_triangular_tag)

        args = lambda k: (  # noqa: E731
            jnp.tril(jr.normal(k, (n, n), dtype=dtype)) + 4.0 * jnp.eye(n, dtype=dtype),
        )
        solver = lx.Triangular()
    elif kind == "tridiagonal_tagged":
        # A dense matrix that merely *promises* to be tridiagonal: the fast path has to
        # reach it through `tridiagonal(...)`, including on the tangent operator.
        def make(matrix):
            return lx.MatrixLinearOperator(matrix, lx.tridiagonal_tag)

        def args(k):
            k0, k1, k2 = jr.split(k, 3)
            d = jr.normal(k0, (n,), dtype=dtype) + 4.0
            lo = jr.normal(k1, (n - 1,), dtype=dtype) * 0.3
            up = jr.normal(k2, (n - 1,), dtype=dtype) * 0.3
            return (jnp.diag(d) + jnp.diag(lo, -1) + jnp.diag(up, 1),)

        solver = lx.Tridiagonal()
    else:
        raise AssertionError(kind)
    return make(*args(keys[0])), make(*args(keys[1])), solver


@pytest.mark.parametrize(
    "kind",
    ["diagonal", "triangular", "tridiagonal", "tridiagonal_tagged", "circulant"],
)
@pytest.mark.parametrize("complex_", [False, True])
def test_slogdet_structured_jvp(kind, complex_, getkey):
    """Structured operators take a fast JVP path; it must match `trace(A^-1 dA)`.

    `lx.slogdet`'s generic rule solves once per column of the tangent. For operators
    whose determinant is a cheap pure-JAX function of their structure we differentiate
    that function instead, which is O(n) (O(n log n) for circulant). This checks the
    two agree, on the sign as well as the log-magnitude -- the sign tangent is only
    nonzero in the complex case, and is where `jnp.sign` would silently give zero.
    """
    n = 32
    op, t_op, solver = _structured_case(kind, n, getkey(), complex_)
    (sign, lad), (sign_dot, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, solver), (op,), (t_op,)
    )
    matrix = op.as_matrix()
    t_matrix = t_op.as_matrix()
    ref_sign, ref_lad = jnp.linalg.slogdet(matrix)
    trace = jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    ref_lad_dot = jnp.real(trace)
    if complex_:
        # The sign carries the imaginary part of the trace; cast explicitly, as lineax
        # runs under strict dtype promotion.
        ref_sign_dot = (trace - ref_lad_dot.astype(trace.dtype)) * ref_sign
    else:
        ref_sign_dot = jnp.zeros_like(sign)
    assert jnp.allclose(sign, ref_sign, atol=1e-10)
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)
    assert jnp.allclose(lad_dot, ref_lad_dot, rtol=1e-8)
    assert jnp.allclose(sign_dot, ref_sign_dot, atol=1e-8)


@pytest.mark.parametrize("kind", ["diagonal", "tridiagonal", "circulant"])
def test_slogdet_structured_jvp_is_not_quadratic(kind, getkey):
    """The fast JVP must not materialise the tangent densely.

    The generic rule is O(n**2) in both work and memory -- at n = 65536 it needs a
    34GB tangent -- so a regression here is a memory blow-up rather than a slowdown.
    Differentiating at a size whose dense tangent would not fit checks the fast path
    is really being taken.
    """
    n = 100_000
    op, t_op, solver = _structured_case(kind, n, getkey(), False)
    _, (_, lad_dot) = jax.jvp(lambda o: lx.slogdet(o, solver), (op,), (t_op,))
    assert jnp.isfinite(lad_dot)


def test_slogdet_pseudodeterminant_jvp(getkey):
    """A rank-deficient solver computes a pseudodeterminant, a different function.

    The fast path differentiates whatever the solver computes, so it needs no rank
    guard: `Diagonal(well_posed=False)` masks small entries and its tangent masks the
    same ones. This pins that, since getting it wrong would silently differentiate the
    full determinant of a singular operator.
    """
    diag = jnp.array([2.0, 3.0, 0.0, 5.0])
    t_diag = jnp.array([1.0, 1.0, 1.0, 1.0])
    op = lx.DiagonalLinearOperator(diag)
    t_op = lx.DiagonalLinearOperator(t_diag)
    solver = lx.Diagonal(well_posed=False)
    (_, lad), (_, lad_dot) = jax.jvp(lambda o: lx.slogdet(o, solver), (op,), (t_op,))
    # Pseudodeterminant over the nonzero entries: 2 * 3 * 5, and its tangent
    # sum(t / d) over those same entries.
    assert jnp.allclose(lad, jnp.log(30.0))
    assert jnp.allclose(lad_dot, 1 / 2 + 1 / 3 + 1 / 5)


def test_slogdet_structured_jvp_nonflat_structure(getkey):
    """`is_tridiagonal` does not imply a flat in/out structure, and need not.

    A pytree-structured operator can carry the tag; `tridiagonal(...)` ravels it, so
    the fast path applies there too. (`try_structured_materialise` guards on flatness
    for a different reason: `TridiagonalLinearOperator` cannot represent a pytree.)
    """
    matrix = jnp.diag(jnp.array([3.0, 4.0, 5.0, 6.0]))
    matrix += jnp.diag(jnp.array([0.1, 0.2, 0.3]), -1)
    matrix += jnp.diag(jnp.array([0.4, 0.5, 0.6]), 1)
    t_matrix = jnp.arange(16.0).reshape(4, 4) * jnp.where(
        jnp.abs(jnp.arange(4)[:, None] - jnp.arange(4)[None, :]) <= 1, 1.0, 0.0
    )

    struct = {
        "a": jax.ShapeDtypeStruct((2,), jnp.float64),
        "b": jax.ShapeDtypeStruct((2,), jnp.float64),
    }

    def to_pytree(m):
        return {
            "a": {"a": m[:2, :2], "b": m[:2, 2:]},
            "b": {"a": m[2:, :2], "b": m[2:, 2:]},
        }

    def make(m):
        return lx.PyTreeLinearOperator(to_pytree(m), struct, lx.tridiagonal_tag)

    op, t_op = make(matrix), make(t_matrix)
    assert not isinstance(op.in_structure(), jax.ShapeDtypeStruct)
    (_, lad), (_, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, lx.Tridiagonal()), (op,), (t_op,)
    )
    _, ref_lad = jnp.linalg.slogdet(matrix)
    ref_dot = jnp.trace(jnp.linalg.solve(matrix, t_matrix))
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)
    assert jnp.allclose(lad_dot, ref_dot, rtol=1e-8)


@pytest.mark.parametrize("lower", [True, False])
def test_slogdet_unit_diagonal_jvp(lower):
    """A unit diagonal pins the determinant to 1 *for a triangular operator*.

    The solver honours the promise rather than reading the stored diagonal, so the
    fast path must too -- and must not extend the reasoning to operators that merely
    happen to have ones on the diagonal.
    """
    matrix = jnp.array([[1.0, 0.0], [3.0, 1.0]])
    t_matrix = jnp.array([[5.0, 0.0], [7.0, 11.0]])
    if not lower:
        matrix, t_matrix = matrix.T, t_matrix.T
    tag = (lx.lower_triangular_tag, lx.unit_diagonal_tag)
    if not lower:
        tag = (lx.upper_triangular_tag, lx.unit_diagonal_tag)
    op = lx.MatrixLinearOperator(matrix, tag)
    t_op = lx.MatrixLinearOperator(t_matrix, tag)
    (sign, lad), (sign_dot, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, lx.Triangular()), (op,), (t_op,)
    )
    assert jnp.allclose(sign, 1.0)
    assert jnp.allclose(lad, 0.0)
    assert jnp.allclose(sign_dot, 0.0)
    assert jnp.allclose(lad_dot, 0.0)


def test_slogdet_pseudodeterminant_complex_sign_jvp():
    """The masked pseudodeterminant's *sign* also has a tangent, for complex operators.

    `jnp.sign` reports a zero tangent, which is right for real inputs and wrong for
    complex ones, so this path needs `unit_phase` as much as the full-rank one does.
    """
    diag = jnp.array([2.0 + 1.0j, 3.0 - 2.0j, 0.0 + 0.0j, 5.0 + 4.0j])
    t_diag = jnp.array([1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 0.0j, 1.0 + 2.0j])
    op = lx.DiagonalLinearOperator(diag)
    t_op = lx.DiagonalLinearOperator(t_diag)
    solver = lx.Diagonal(well_posed=False)
    (sign, _), (sign_dot, lad_dot) = jax.jvp(
        lambda o: lx.slogdet(o, solver), (op,), (t_op,)
    )
    # Over the three retained entries, d log(det) = sum(t / d); the real part is the
    # tangent of log|det| and the rest turns the phase.
    kept = jnp.array([0, 1, 3])
    trace = jnp.sum(t_diag[kept] / diag[kept])
    assert jnp.allclose(lad_dot, jnp.real(trace))
    assert jnp.allclose(sign_dot, (trace - jnp.real(trace).astype(trace.dtype)) * sign)


@pytest.mark.parametrize("kind", ["function", "jacobian"])
def test_slogdet_structured_jvp_opaque_operator(kind):
    """An operator carrying a structure tag need not be made of arrays alone.

    `FunctionLinearOperator` and `JacobianLinearOperator` hold a callable, so the fast
    path has to differentiate them with `eqx.filter_jvp`; plain `jax.jvp` rejects the
    tangent, whose callable leaf is `None`. These reach the fast path like any other
    tagged operator, and used to reach the generic one, so a regression here is a
    crash on operators that previously worked.
    """
    n = 8
    key = jr.PRNGKey(0)
    k0, k1, k2, k3 = jr.split(key, 4)
    matrix = (
        jnp.diag(jr.normal(k0, (n,)) + 4.0)
        + jnp.diag(jr.normal(k1, (n - 1,)) * 0.3, -1)
        + jnp.diag(jr.normal(k2, (n - 1,)) * 0.3, 1)
    )
    tangent = jnp.diag(jr.normal(k3, (n,)))
    structure = jax.ShapeDtypeStruct((n,), matrix.dtype)

    def make(m):
        if kind == "function":
            return lx.FunctionLinearOperator(
                lambda v: m @ v, structure, lx.tridiagonal_tag
            )
        return lx.JacobianLinearOperator(
            lambda x, args: m @ x,
            jnp.zeros(n, matrix.dtype),
            None,
            tags=lx.tridiagonal_tag,
        )

    (_, lad), (_, lad_dot) = eqx.filter_jvp(
        lambda m: lx.slogdet(make(m), lx.Tridiagonal()), (matrix,), (tangent,)
    )
    _, ref_lad = jnp.linalg.slogdet(matrix)
    ref_dot = jnp.trace(jnp.linalg.solve(matrix, tangent))
    assert jnp.allclose(lad, ref_lad, rtol=1e-10)
    assert jnp.allclose(lad_dot, ref_dot, rtol=1e-8)
