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
import lineax as lx
import pytest


# ---------------------------------------------------------------------------
# RankRangeTag semantics
# ---------------------------------------------------------------------------


def test_rank_tag_constructors():
    assert lx.MaxRankTag(5) == lx.RankRangeTag(lo=0, hi=5)
    assert lx.MinRankTag(5) == lx.RankRangeTag(lo=5, hi=None)
    assert lx.RankTag(5) == lx.RankRangeTag(lo=5, hi=5)
    assert lx.RankRangeTag() == lx.RankRangeTag(lo=0, hi=None)
    assert isinstance(lx.MaxRankTag(5), lx.RankRangeTag)


def test_rank_tag_equality():
    assert lx.MaxRankTag(5) == lx.MaxRankTag(5)
    assert lx.MaxRankTag(5) != lx.MaxRankTag(3)
    assert lx.MaxRankTag(5) != lx.MinRankTag(5)
    assert lx.RankTag(5) != lx.MaxRankTag(5)


def test_rank_tag_hashable_frozenset_dedup():
    s = frozenset({lx.MaxRankTag(5), lx.MaxRankTag(5)})
    assert len(s) == 1
    # Dedup is by value, regardless of which constructor built the tag.
    s = frozenset({lx.RankTag(3), lx.RankRangeTag(3, 3)})
    assert len(s) == 1
    s = frozenset({lx.MinRankTag(2), lx.RankRangeTag(lo=2)})
    assert len(s) == 1


def test_rank_tag_zero_valid():
    tag = lx.MaxRankTag(0)
    assert (tag.lo, tag.hi) == (0, 0)
    tag = lx.MinRankTag(0)
    assert (tag.lo, tag.hi) == (0, None)


@pytest.mark.parametrize(
    "make",
    (
        lambda: lx.MaxRankTag(-1),
        lambda: lx.MinRankTag(-1),
        lambda: lx.RankTag(-1),
        lambda: lx.RankRangeTag(lo=-1),
        lambda: lx.RankRangeTag(lo=3, hi=2),  # contradictory
    ),
)
def test_rank_tag_invalid_raises(make):
    with pytest.raises(ValueError):
        make()


def test_rank_tag_repr():
    assert repr(lx.MaxRankTag(7)) == "max_rank_tag(7)"
    assert repr(lx.MinRankTag(7)) == "min_rank_tag(7)"
    assert repr(lx.RankTag(7)) == "rank_tag(7)"
    assert repr(lx.RankRangeTag(3, 7)) == "rank_range_tag(3, 7)"
    assert repr(lx.RankRangeTag()) == "rank_range_tag()"


# ---------------------------------------------------------------------------
# Basic operator dispatch
# ---------------------------------------------------------------------------


def test_max_rank_matrix_no_tag():
    op = lx.MatrixLinearOperator(jnp.eye(4))
    assert lx.max_rank(op) == 4  # min(4, 4)


def test_max_rank_matrix_rectangular_no_tag():
    op = lx.MatrixLinearOperator(jnp.zeros((6, 3)))
    assert lx.max_rank(op) == 3  # min(6, 3)


def test_max_rank_matrix_with_tag():
    op = lx.MatrixLinearOperator(jnp.zeros((10, 10)), lx.MaxRankTag(3))
    assert lx.max_rank(op) == 3


def test_max_rank_matrix_tag_capped_by_dimension():
    # Tag claims rank 20 but matrix is only 4×4 — dimension wins.
    op = lx.MatrixLinearOperator(jnp.zeros((4, 4)), lx.MaxRankTag(20))
    assert lx.max_rank(op) == 4


def test_max_rank_tagged_operator():
    inner = lx.MatrixLinearOperator(jnp.zeros((8, 8)))
    op = lx.TaggedLinearOperator(inner, lx.MaxRankTag(2))
    assert lx.max_rank(op) == 2


def test_max_rank_tagged_operator_narrows_inner():
    inner = lx.MatrixLinearOperator(jnp.zeros((8, 8)), lx.MaxRankTag(5))
    op = lx.TaggedLinearOperator(inner, lx.MaxRankTag(3))
    assert lx.max_rank(op) == 3


def test_max_rank_identity():
    import jax

    struct = jax.ShapeDtypeStruct((5,), jnp.float32)
    op = lx.IdentityLinearOperator(input_structure=struct)
    assert lx.max_rank(op) == 5


def test_max_rank_diagonal():
    op = lx.DiagonalLinearOperator(jnp.ones(6))
    assert lx.max_rank(op) == 6


# ---------------------------------------------------------------------------
# Composition rules
# ---------------------------------------------------------------------------


def test_max_rank_composed_both_tagged():
    # n×k @ k×n: min(k, k) = k
    k, n = 3, 10
    U = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
    Vt = lx.MatrixLinearOperator(jnp.zeros((k, n)), lx.MaxRankTag(k))
    assert lx.max_rank(U @ Vt) == k


def test_max_rank_composed_one_tagged():
    k, n = 3, 10
    U = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
    A = lx.MatrixLinearOperator(jnp.zeros((k, n)))  # no tag → min(k,n)=k
    assert lx.max_rank(U @ A) == k


def test_max_rank_composed_chain():
    k, n = 5, 100
    U = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
    C = lx.MatrixLinearOperator(jnp.zeros((k, k)), lx.MaxRankTag(k))
    Vt = lx.MatrixLinearOperator(jnp.zeros((k, n)), lx.MaxRankTag(k))
    assert lx.max_rank(U @ C @ Vt) == k


def test_max_rank_add_both_tagged():
    op1 = lx.MatrixLinearOperator(jnp.zeros((6, 6)), lx.MaxRankTag(2))
    op2 = lx.MatrixLinearOperator(jnp.zeros((6, 6)), lx.MaxRankTag(1))
    assert lx.max_rank(op1 + op2) == 3  # min(2+1, 6)


def test_max_rank_add_capped_by_dimension():
    op1 = lx.MatrixLinearOperator(jnp.zeros((4, 4)), lx.MaxRankTag(3))
    op2 = lx.MatrixLinearOperator(jnp.zeros((4, 4)), lx.MaxRankTag(3))
    assert lx.max_rank(op1 + op2) == 4  # min(3+3=6, 4) = 4


def test_max_rank_add_one_untagged():
    op1 = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    op2 = lx.MatrixLinearOperator(jnp.zeros((5, 5)))  # no tag → 5
    assert lx.max_rank(op1 + op2) == 5  # min(2+5, 5) = 5


# ---------------------------------------------------------------------------
# Scalar multiplication
# ---------------------------------------------------------------------------


def test_max_rank_scalar_mul_propagates():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(3.0 * op) == 2


def test_max_rank_scalar_zero_gives_rank_zero():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(0 * op) == 0


def test_max_rank_neg_propagates():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(-op) == 2


def test_max_rank_div_propagates():
    op = lx.MatrixLinearOperator(jnp.zeros((5, 5)), lx.MaxRankTag(2))
    assert lx.max_rank(op / 2.0) == 2


# ---------------------------------------------------------------------------
# rank_range / min_rank / is_full_rank
# ---------------------------------------------------------------------------


def _mat(shape, *tags):
    return lx.MatrixLinearOperator(jnp.zeros(shape), tags)


@pytest.mark.parametrize(
    "tags, expected",
    (
        ((), (0, 6)),
        ((lx.MaxRankTag(3),), (0, 3)),
        ((lx.MinRankTag(2),), (2, 6)),
        ((lx.RankTag(4),), (4, 4)),
        ((lx.RankRangeTag(2, 5),), (2, 5)),
        # Stacked tags combine into the tightest range.
        ((lx.MaxRankTag(5), lx.MinRankTag(2), lx.RankRangeTag(1, 4)), (2, 4)),
        # Tags beyond the shape are capped by it.
        ((lx.MinRankTag(20),), (6, 6)),
        ((lx.RankTag(20),), (6, 6)),
        # A contradictory combination is clamped rather than raising here; the
        # solvers' rank checks surface it at solve time instead.
        ((lx.MaxRankTag(2), lx.MinRankTag(4)), (2, 2)),
    ),
)
@pytest.mark.parametrize("wrap", (False, True))
def test_rank_range_tags(tags, expected, wrap):
    if wrap:
        op = lx.TaggedLinearOperator(_mat((6, 6)), tags)
    else:
        op = _mat((6, 6), *tags)
    lo, hi = expected
    assert lx.rank_range(op) == expected
    assert lx.min_rank(op) == lo
    assert lx.max_rank(op) == hi
    assert lx.is_full_rank(op) == (lo == 6)
    assert all(type(x) is int for x in lx.rank_range(op))


def test_rank_range_tagged_operator_combines_with_inner():
    inner = _mat((8, 8), lx.RankRangeTag(2, 6))
    op = lx.TaggedLinearOperator(inner, lx.RankRangeTag(3, 7))
    assert lx.rank_range(op) == (3, 6)


def test_rank_range_rectangular():
    op = _mat((6, 3), lx.MinRankTag(3))
    assert lx.rank_range(op) == (3, 3)
    assert lx.is_full_rank(op)
    assert not lx.is_full_rank(_mat((6, 3)))


def test_rank_range_identity_full_rank():
    op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((5,), jnp.float32))
    assert lx.rank_range(op) == (5, 5)
    assert lx.is_full_rank(op)


def test_docstring_example():
    k, n = 5, 100
    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)
    U = lx.MatrixLinearOperator(jax.random.normal(key1, (n, k)), lx.RankTag(k))
    Vt = lx.MatrixLinearOperator(jax.random.normal(key2, (k, n)), lx.MinRankTag(k))
    assert lx.rank_range(U @ Vt) == (k, k)
    assert lx.rank_range(U.T) == (k, k)
    partial = lx.MatrixLinearOperator(
        jax.random.normal(key3, (n, n)), lx.RankRangeTag(3, 7)
    )
    assert lx.rank_range(partial) == (3, 7)


def test_tags_from_checks_rank():
    assert lx.RankRangeTag(2, 4) in lx.tags_from_checks(
        _mat((6, 6), lx.RankRangeTag(2, 4))
    )
    assert lx.MinRankTag(2) not in lx.tags_from_checks(_mat((6, 6)))
    # A lower bound alone is reported, resolved against the shape.
    assert lx.RankRangeTag(2, 6) in lx.tags_from_checks(_mat((6, 6), lx.MinRankTag(2)))
    # The shape bound alone is redundant, so no rank tag is added.
    tags = lx.tags_from_checks(_mat((6, 6)))
    assert not any(isinstance(t, lx.RankRangeTag) for t in tags)


# ---------------------------------------------------------------------------
# Lower-bound composition rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tags1, tags2, expected",
    (
        # Sylvester: lo = max(0, lo1 + lo2 - inner); hi = min(hi1, hi2)
        ((lx.RankTag(4),), (lx.RankTag(4),), (4, 4)),  # 4 + 4 - 4
        ((lx.RankTag(3),), (lx.RankTag(3),), (2, 3)),  # 3 + 3 - 4
        ((lx.MinRankTag(1),), (lx.MinRankTag(2),), (0, 4)),  # 1 + 2 - 4 < 0 clamps
        ((lx.MinRankTag(4),), (lx.MaxRankTag(2),), (0, 2)),  # 4 + 0 - 4
        ((lx.MinRankTag(4),), (lx.RankTag(2),), (2, 2)),  # 4 + 2 - 4
    ),
)
def test_rank_range_composed(tags1, tags2, expected):
    op1 = _mat((6, 4), *tags1)
    op2 = _mat((4, 5), *tags2)
    assert lx.rank_range(op1 @ op2) == expected


def test_rank_range_composed_chain_full_rank():
    # Tall full-column-rank @ square full-rank is full rank...
    op = _mat((6, 3), lx.MinRankTag(3)) @ _mat((3, 3), lx.RankTag(3))
    assert lx.rank_range(op) == (3, 3)
    assert lx.is_full_rank(op)
    # ...and then @ wide full-row-rank has rank exactly 3, but is not full rank (6x7).
    op = op @ _mat((3, 7), lx.MinRankTag(3))
    assert lx.rank_range(op) == (3, 3)
    assert not lx.is_full_rank(op)


@pytest.mark.parametrize(
    "tags1, tags2, expected",
    (
        # lo = min(lo1, lo2, max(0, lo1 - hi2, lo2 - hi1)); hi = min(hi1 + hi2, dim)
        ((lx.RankTag(4),), (lx.RankTag(1),), (1, 5)),
        ((lx.RankTag(1),), (lx.RankTag(4),), (1, 5)),
        ((lx.MinRankTag(6),), (lx.RankTag(2),), (2, 6)),
        ((lx.RankTag(2),), (lx.RankTag(1),), (1, 3)),
        # The exact bound would be 4 here, but the rank-<=1 operand might dominate.
        ((lx.RankTag(5),), (lx.MaxRankTag(1),), (0, 6)),
        ((lx.MaxRankTag(1),), (lx.RankTag(5),), (0, 6)),
        ((lx.MinRankTag(6),), (lx.MaxRankTag(2),), (0, 6)),
        ((lx.RankTag(5),), (lx.RankTag(5),), (0, 6)),  # may cancel
        ((lx.MinRankTag(3),), (), (0, 6)),  # untagged operand may cancel it
    ),
)
def test_rank_range_add(tags1, tags2, expected):
    assert lx.rank_range(_mat((6, 6), *tags1) + _mat((6, 6), *tags2)) == expected


def test_add_swamped_full_rank_does_not_raise():
    # `I + U @ V^T` with a huge rank-1 `U @ V^T`: exactly full rank, but numerically
    # rank 1 at float32 precision. Every tag is true, so no inferred lower bound may
    # trip the solver's rank check; this must match the untagged masked solve.
    n = 6
    identity = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((n,), jnp.float32))
    u = lx.MatrixLinearOperator(1e6 * jnp.ones((n, 1), jnp.float32), lx.MaxRankTag(1))
    vt = lx.MatrixLinearOperator(jnp.ones((1, n), jnp.float32), lx.MaxRankTag(1))
    operator = identity + u @ vt
    assert lx.rank_range(operator) == (0, n)
    vector = jnp.arange(n, dtype=jnp.float32)
    plain = lx.MatrixLinearOperator(operator.as_matrix())
    sol = lx.linear_solve(operator, vector, lx.SVD())
    expected = lx.linear_solve(plain, vector, lx.SVD())
    assert sol.stats["rank"] == expected.stats["rank"] == 1
    assert jnp.allclose(sol.value, expected.value)


# ---------------------------------------------------------------------------
# Lower bounds through scalar operations and tangents
# ---------------------------------------------------------------------------


def test_rank_range_scalar_ops():
    op = _mat((5, 5), lx.RankRangeTag(2, 4))
    assert lx.rank_range(3.0 * op) == (2, 4)
    assert lx.rank_range(-2 * op) == (2, 4)
    assert lx.rank_range(0.0 * op) == (0, 0)
    assert lx.rank_range(-op) == (2, 4)
    assert lx.rank_range(op / 2.0) == (2, 4)


def test_rank_range_traced_scalar_drops_lower_bound():
    # A traced scalar could be zero at runtime, so only the upper bound survives.
    op = _mat((5, 5), lx.RankRangeTag(2, 4))

    @jax.jit
    def f(x):
        assert lx.rank_range(x * op) == (0, 4)
        assert lx.rank_range(op / x) == (2, 4)
        return x

    f(1.0)


def test_rank_range_tangent_has_no_lower_bound():
    op = _mat((6, 6), lx.RankRangeTag(2, 2))
    tangent = lx.TangentLinearOperator(op, op)
    assert lx.rank_range(tangent) == (0, 4)


# ---------------------------------------------------------------------------
# Transpose
# ---------------------------------------------------------------------------


def test_max_rank_transpose_preserves_tag():
    op = lx.MatrixLinearOperator(jnp.zeros((10, 3)), lx.MaxRankTag(3))
    assert lx.max_rank(op.T) == 3


def test_rank_range_transpose_preserves_tags():
    op = _mat((10, 3), lx.RankRangeTag(2, 3))
    assert lx.rank_range(op.T) == (2, 3)
    op = lx.TaggedLinearOperator(_mat((6, 4)), lx.MinRankTag(3))
    assert lx.rank_range(op.T) == (3, 4)


def test_max_rank_tagged_operator_transpose():
    inner = lx.MatrixLinearOperator(jnp.zeros((8, 8)))
    op = lx.TaggedLinearOperator(inner, lx.MaxRankTag(4))
    assert lx.max_rank(op.T) == 4


# ---------------------------------------------------------------------------
# Invert tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", (lx.invert_tags, lx.transpose_tags))
@pytest.mark.parametrize(
    "tag", (lx.MaxRankTag(3), lx.MinRankTag(3), lx.RankTag(3), lx.RankRangeTag(2, 4))
)
def test_rank_tags_preserved_by_invert_and_transpose(fn, tag):
    assert tag in fn(frozenset({tag}))


@pytest.mark.parametrize("fn", (lx.invert_tags, lx.transpose_tags))
def test_rank_tags_combined_by_invert_and_transpose(fn):
    tags = frozenset(
        {lx.MaxRankTag(5), lx.MaxRankTag(4), lx.MinRankTag(1), lx.MinRankTag(2)}
    )
    rank_tags = [t for t in fn(tags) if isinstance(t, lx.RankRangeTag)]
    assert rank_tags == [lx.RankRangeTag(2, 4)]
    # Only lower bounds: the upper bound stays unconstrained.
    tags = frozenset({lx.MinRankTag(1), lx.MinRankTag(2)})
    rank_tags = [t for t in fn(tags) if isinstance(t, lx.RankRangeTag)]
    assert rank_tags == [lx.MinRankTag(2)]


@pytest.mark.parametrize("fn", (lx.invert_tags, lx.transpose_tags))
def test_rank_tags_absent_when_no_rank_tag(fn):
    result = fn(frozenset({lx.lower_triangular_tag}))
    assert not any(isinstance(t, lx.RankRangeTag) for t in result)


# ---------------------------------------------------------------------------
# SVD truncation
# ---------------------------------------------------------------------------


def test_svd_truncates_to_max_rank():
    # A genuinely rank-2 matrix, declared rank 2: SVD truncates to 2 components, and
    # the solution matches the untruncated solve. The state keeps the full
    # decomposition, so the trailing components (the null space) remain recoverable.
    u = jax.random.normal(jax.random.PRNGKey(0), (10, 2))
    v = jax.random.normal(jax.random.PRNGKey(1), (10, 2))
    matrix = u @ v.T
    solver = lx.SVD()

    plain = lx.MatrixLinearOperator(matrix)
    (u_full, s_full, vt_full), ranks, _ = solver.init(plain, {})
    assert s_full.shape == (10,)
    assert ranks.value == (0, 10)

    tagged = lx.MatrixLinearOperator(matrix, lx.MaxRankTag(2))
    (u_t, s_t, vt_t), ranks, _ = solver.init(tagged, {})
    assert ranks.value == (0, 2)
    assert u_t.shape == (10, 10)
    assert s_t.shape == (10,)
    assert vt_t.shape == (10, 10)
    assert jnp.allclose(matrix @ vt_t[2:].T, 0, atol=1e-10)
    assert jnp.allclose(u_t[:, 2:].T @ matrix, 0, atol=1e-10)

    vector = jnp.arange(10.0) + 0.5
    x_plain = lx.linear_solve(plain, vector, solver).value
    x_tagged = lx.linear_solve(tagged, vector, solver).value
    assert jnp.allclose(x_plain, x_tagged)


def test_svd_raises_when_max_rank_too_small():
    # A genuinely rank-3 matrix declared rank 2: truncation would discard a
    # singular value above the rcond threshold, so the solve must raise.
    u = jax.random.normal(jax.random.PRNGKey(0), (10, 3))
    v = jax.random.normal(jax.random.PRNGKey(1), (10, 3))
    matrix = u @ v.T
    operator = lx.MatrixLinearOperator(matrix, lx.MaxRankTag(2))
    vector = jnp.arange(10.0) + 0.5
    with pytest.raises(eqx.EquinoxRuntimeError):
        lx.linear_solve(operator, vector, lx.SVD())


# ---------------------------------------------------------------------------
# HEVD truncation
# ---------------------------------------------------------------------------


def _hermitian_with_spectrum(key, eigvals):
    # `Q diag(eigvals) Q^T` with `Q` orthogonal: a (real-)Hermitian matrix with the
    # given eigenvalues. Zeros placed between nonzero eigenvalues of both signs land
    # in the interior of eigh's ascending order, exercising the fact that HEVD must
    # truncate by *magnitude* (unlike SVD, where the small values are a contiguous
    # tail).
    size = len(eigvals)
    q, _ = jnp.linalg.qr(jax.random.normal(key, (size, size)))
    d = jnp.asarray(eigvals, dtype=q.dtype)
    return (q * d[None, :]) @ q.T


def test_hevd_truncates_to_max_rank():
    # A genuinely rank-2 indefinite Hermitian matrix, declared rank 2: HEVD truncates
    # to 2 (eigenvalue, eigenvector) pairs and the solution is unchanged.
    matrix = _hermitian_with_spectrum(jax.random.PRNGKey(0), [3.0, -2.0, 0.0, 0.0, 0.0])
    solver = lx.HEVD()

    plain = lx.MatrixLinearOperator(matrix, lx.hermitian_tag)
    (w_full, v_full), ranks, _ = solver.init(plain, {})
    assert w_full.shape == (5,)
    assert v_full.shape == (5, 5)
    assert ranks.value == (0, 5)

    tagged = lx.MatrixLinearOperator(matrix, (lx.hermitian_tag, lx.MaxRankTag(2)))
    (w_t, v_t), ranks, _ = solver.init(tagged, {})
    assert ranks.value == (0, 2)
    assert w_t.shape == (5,)
    assert v_t.shape == (5, 5)

    vector = jnp.arange(5.0) + 0.5
    x_plain = lx.linear_solve(plain, vector, solver).value
    x_tagged = lx.linear_solve(tagged, vector, solver).value
    assert jnp.allclose(x_plain, x_tagged)


# PSD (eigenvalues >= 0) and NSD (<= 0) operators have their near-zero eigenvalues at
# a contiguous *end* of eigh's ascending order, so truncation is a slice rather than a
# reorder. `semidefinite_tag` is also a slice, but at an offset chosen at runtime from
# the sign of the spectrum. Indefinite operators need the reordering gather. Cover all
# four branches, and both signs of the runtime-resolved one.
@pytest.mark.parametrize(
    "tag, eigvals",
    (
        (lx.hermitian_tag, [3.0, -2.0, 0.0, 0.0, 0.0]),  # indefinite -> reorder
        (lx.positive_semidefinite_tag, [3.0, 2.0, 0.0, 0.0, 0.0]),  # PSD -> slice tail
        (lx.negative_semidefinite_tag, [-3.0, -2.0, 0.0, 0.0, 0.0]),  # NSD -> head
        (lx.semidefinite_tag, [3.0, 2.0, 0.0, 0.0, 0.0]),  # +ve -> dynamic slice tail
        (lx.semidefinite_tag, [-3.0, -2.0, 0.0, 0.0, 0.0]),  # -ve -> dynamic slice head
    ),
)
def test_hevd_truncation_branches(tag, eigvals):
    matrix = _hermitian_with_spectrum(jax.random.PRNGKey(0), eigvals)
    solver = lx.HEVD()
    plain = lx.MatrixLinearOperator(matrix, tag)
    tagged = lx.MatrixLinearOperator(matrix, (tag, lx.MaxRankTag(2)))

    (w_t, v_t), ranks, _ = solver.init(tagged, {})
    assert ranks.value == (0, 2)
    # The state keeps every eigenpair, ordered by descending magnitude: the retained
    # pairs lead, and the trailing ones span the null space.
    assert w_t.shape == (5,)
    assert v_t.shape == (5, 5)
    expected = sorted(eigvals, key=abs, reverse=True)
    assert jnp.allclose(w_t, jnp.array(expected), atol=1e-10)
    assert jnp.allclose(matrix @ v_t[:, 2:], 0, atol=1e-10)
    assert jnp.allclose(matrix @ v_t[:, :2], v_t[:, :2] * w_t[None, :2], atol=1e-10)

    vector = jnp.arange(5.0) + 0.5
    x_plain = lx.linear_solve(plain, vector, solver).value
    x_tagged = lx.linear_solve(tagged, vector, solver).value
    assert jnp.allclose(x_plain, x_tagged)


@pytest.mark.parametrize(
    "tag, eigvals",
    (
        (lx.hermitian_tag, [3.0, -2.0, 1.5, 0.0, 0.0]),
        (lx.positive_semidefinite_tag, [3.0, 2.0, 1.5, 0.0, 0.0]),
        (lx.negative_semidefinite_tag, [-3.0, -2.0, -1.5, 0.0, 0.0]),
        (lx.semidefinite_tag, [3.0, 2.0, 1.5, 0.0, 0.0]),
        (lx.semidefinite_tag, [-3.0, -2.0, -1.5, 0.0, 0.0]),
    ),
)
def test_hevd_raises_when_max_rank_too_small(tag, eigvals):
    # A genuinely rank-3 Hermitian matrix declared rank 2: truncation would discard an
    # eigenvalue above the rcond threshold, so the solve must raise.
    matrix = _hermitian_with_spectrum(jax.random.PRNGKey(0), eigvals)
    operator = lx.MatrixLinearOperator(matrix, (tag, lx.MaxRankTag(2)))
    vector = jnp.arange(5.0) + 0.5
    with pytest.raises(eqx.EquinoxRuntimeError):
        lx.linear_solve(operator, vector, lx.HEVD())


# ---------------------------------------------------------------------------
# Lower-bound verification and the exact-rank fast path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", (lx.SVD(), lx.HEVD()))
@pytest.mark.parametrize("tag", (lx.MinRankTag(6), lx.RankTag(6)))
def test_full_rank_tag_solves(solver, tag, getkey):
    matrix = jax.random.normal(getkey(), (6, 6))
    if isinstance(solver, lx.HEVD):
        matrix = matrix @ matrix.T + jnp.eye(6)
        tags = (lx.hermitian_tag, tag)
    else:
        tags = (tag,)
    tagged = lx.MatrixLinearOperator(matrix, tags)
    assert lx.is_full_rank(tagged)
    _, ranks, _ = solver.init(tagged, {})
    assert ranks.value == (6, 6)
    vector = jax.random.normal(getkey(), (6,))
    sol = lx.linear_solve(tagged, vector, solver)
    assert jnp.allclose(sol.value, jnp.linalg.solve(matrix, vector))
    assert sol.stats["rank"] == 6


@pytest.mark.parametrize("solver", (lx.SVD(), lx.HEVD()))
def test_exact_rank_tag_truncates_and_skips_mask(solver, getkey):
    # A genuinely rank-2 matrix tagged `RankTag(2)`: truncated to 2 components, all
    # certified nonzero, and matching the untagged solve.
    u = jax.random.normal(getkey(), (6, 2))
    if isinstance(solver, lx.HEVD):
        matrix = u @ u.T
        base_tags = (lx.positive_semidefinite_tag,)
    else:
        matrix = u @ jax.random.normal(getkey(), (2, 6))
        base_tags = ()
    plain = lx.MatrixLinearOperator(matrix, base_tags)
    tagged = lx.MatrixLinearOperator(matrix, (*base_tags, lx.RankTag(2)))
    _, ranks, _ = solver.init(tagged, {})
    assert ranks.value == (2, 2)
    _, ranks, _ = solver.init(plain, {})
    assert ranks.value == (0, 6)
    vector = jax.random.normal(getkey(), (6,))
    sol = lx.linear_solve(tagged, vector, solver)
    assert jnp.allclose(sol.value, lx.linear_solve(plain, vector, solver).value)
    assert sol.stats["rank"] == 2


@pytest.mark.parametrize("solver", (lx.SVD(), lx.HEVD()))
@pytest.mark.parametrize(
    "tag", (lx.MinRankTag(3), lx.RankTag(3), lx.RankRangeTag(3, 5), lx.MinRankTag(6))
)
def test_raises_when_min_rank_too_large(solver, tag, getkey):
    # A genuinely rank-2 matrix claimed to have rank at least 3.
    u = jax.random.normal(getkey(), (6, 2))
    if isinstance(solver, lx.HEVD):
        matrix = u @ u.T
        tags = (lx.hermitian_tag, tag)
    else:
        matrix = u @ jax.random.normal(getkey(), (2, 6))
        tags = (tag,)
    operator = lx.MatrixLinearOperator(matrix, tags)
    vector = jax.random.normal(getkey(), (6,))
    with pytest.raises(eqx.EquinoxRuntimeError):
        lx.linear_solve(operator, vector, solver)


@pytest.mark.parametrize("solver", (lx.SVD(), lx.HEVD()))
def test_ill_conditioned_full_rank_tag_raises(solver, getkey):
    # Structurally full rank but numerically singular (a singular value below rcond).
    # Untagged, the solver silently truncates it, like `scipy.linalg.lstsq`. Tagged as
    # full rank, this is an intentional hard error instead.
    q, _ = jnp.linalg.qr(jax.random.normal(getkey(), (5, 5)))
    matrix = (q * jnp.array([3.0, 2.0, 1.0, 0.5, 1e-20])[None, :]) @ q.T
    base_tags = (lx.hermitian_tag,) if isinstance(solver, lx.HEVD) else ()
    vector = jax.random.normal(getkey(), (5,))

    plain = lx.MatrixLinearOperator(matrix, base_tags)
    sol = lx.linear_solve(plain, vector, solver)
    assert sol.stats["rank"] == 4

    tagged = lx.MatrixLinearOperator(matrix, (*base_tags, lx.MinRankTag(5)))
    with pytest.raises(eqx.EquinoxRuntimeError):
        lx.linear_solve(tagged, vector, solver)


def test_full_rank_svd_jvp_and_gram(getkey):
    # The exact-rank flag survives the transpose/gram shortcuts used by the JVP.
    matrix = jax.random.normal(getkey(), (7, 4))
    vector = jax.random.normal(getkey(), (7,))
    t_matrix = jax.random.normal(getkey(), (7, 4))

    def solve(m, tags):
        op = lx.MatrixLinearOperator(m, tags)
        return lx.linear_solve(op, vector, lx.SVD()).value

    expected = jax.jvp(lambda m: solve(m, ()), (matrix,), (t_matrix,))
    got = jax.jvp(lambda m: solve(m, lx.MinRankTag(4)), (matrix,), (t_matrix,))
    assert jnp.allclose(expected[0], got[0])
    assert jnp.allclose(expected[1], got[1])


# ---------------------------------------------------------------------------
# Diagonal and Circulant: lower-bound verification and skipping the mask
# ---------------------------------------------------------------------------


def _circulant_with_spectrum(eigvals):
    # A real circulant operator needs a conjugate-symmetric spectrum; `eigvals` is the
    # `rfft` half of it.
    n = 2 * (len(eigvals) - 1)
    column = jnp.fft.irfft(jnp.asarray(eigvals, dtype=jnp.complex128), n=n)
    return lx.CirculantLinearOperator(column)


def _rank_deficient(solver_type, tags=()):
    # Rank 3 out of 6, for each of Diagonal and Circulant.
    if solver_type is lx.Diagonal:
        diag = jnp.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        return lx.DiagonalLinearOperator(diag), lx.MatrixLinearOperator(
            jnp.diag(diag), (lx.diagonal_tag, *tags)
        )
    # `rfft` half of a length-6 spectrum: DC, two conjugate pairs, Nyquist. Zeroing one
    # pair and the Nyquist term leaves rank 1 + 2 = 3.
    op = _circulant_with_spectrum([3.0, 2.0, 0.0, 0.0])
    return op, lx.TaggedLinearOperator(op, tags)


@pytest.mark.parametrize("solver_type", (lx.Diagonal, lx.Circulant))
def test_diagonal_circulant_full_rank_skips_mask(solver_type, getkey):
    if solver_type is lx.Diagonal:
        matrix = jnp.diag(jax.random.uniform(getkey(), (6,)) + 1)
        tagged = lx.MatrixLinearOperator(matrix, (lx.diagonal_tag, lx.RankTag(6)))
    else:
        op = _circulant_with_spectrum([4.0, 3.0, 2.0, 1.0])
        matrix = op.as_matrix()
        tagged = lx.TaggedLinearOperator(op, lx.MinRankTag(6))
    solver = solver_type()
    if solver_type is lx.Diagonal:
        _, full_rank, _ = lx.Diagonal().init(tagged, {})
    else:
        (_, _, full_rank), _ = lx.Circulant().init(tagged, {})
    assert full_rank.value
    vector = jax.random.normal(getkey(), (6,))
    sol = lx.linear_solve(tagged, vector, solver)
    assert jnp.allclose(sol.value, jnp.linalg.solve(matrix, vector))
    sign, logabsdet = lx.slogdet(tagged, solver)
    expected_sign, expected_logabsdet = jnp.linalg.slogdet(matrix)
    assert jnp.allclose(sign, expected_sign)
    assert jnp.allclose(logabsdet, expected_logabsdet)


@pytest.mark.parametrize("solver_type", (lx.Diagonal, lx.Circulant))
def test_diagonal_circulant_correct_min_rank_solves(solver_type, getkey):
    # A genuinely rank-3 operator claimed to have rank at least 3: no error, and the
    # usual masked pseudoinverse solve.
    plain, tagged = _rank_deficient(solver_type, (lx.MinRankTag(3),))
    vector = jax.random.normal(getkey(), (6,))
    x_plain = lx.linear_solve(plain, vector, solver_type()).value
    x_tagged = lx.linear_solve(tagged, vector, solver_type()).value
    assert jnp.allclose(x_plain, x_tagged)


@pytest.mark.parametrize("solver_type", (lx.Diagonal, lx.Circulant))
@pytest.mark.parametrize("tag", (lx.MinRankTag(4), lx.RankTag(6)))
def test_diagonal_circulant_raises_when_min_rank_too_large(solver_type, tag, getkey):
    _, tagged = _rank_deficient(solver_type, (tag,))
    vector = jax.random.normal(getkey(), (6,))
    with pytest.raises(eqx.EquinoxRuntimeError):
        lx.linear_solve(tagged, vector, solver_type())


@pytest.mark.parametrize("solver_type", (lx.Diagonal, lx.Circulant))
def test_diagonal_circulant_well_posed_ignores_min_rank(solver_type, getkey):
    # `well_posed=True` never masks, so there is no rcond cutoff to check against: an
    # entry below the default cutoff (~5e-15 here) does not trip the rank check.
    if solver_type is lx.Diagonal:
        op = lx.MatrixLinearOperator(
            jnp.diag(jnp.array([3.0, 2.0, 1e-15])), (lx.diagonal_tag, lx.RankTag(3))
        )
    else:
        op = lx.TaggedLinearOperator(
            _circulant_with_spectrum([3.0, 2.0, 1e-15]), lx.RankTag(4)
        )
    vector = jax.random.normal(getkey(), (op.in_size(),))
    sol = lx.linear_solve(op, vector, solver_type(well_posed=True))
    assert jnp.all(jnp.isfinite(sol.value))
