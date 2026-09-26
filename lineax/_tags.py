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

import dataclasses
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ._operator import AbstractLinearOperator


class _HasRepr:
    def __init__(self, string: str):
        self.string = string

    def __repr__(self):
        return self.string


symmetric_tag = _HasRepr("symmetric_tag")
hermitian_tag = _HasRepr("hermitian_tag")
diagonal_tag = _HasRepr("diagonal_tag")
tridiagonal_tag = _HasRepr("tridiagonal_tag")
unit_diagonal_tag = _HasRepr("unit_diagonal_tag")
lower_triangular_tag = _HasRepr("lower_triangular_tag")
upper_triangular_tag = _HasRepr("upper_triangular_tag")
positive_semidefinite_tag = _HasRepr("positive_semidefinite_tag")
negative_semidefinite_tag = _HasRepr("negative_semidefinite_tag")
semidefinite_tag = _HasRepr("semidefinite_tag")
circulant_tag = _HasRepr("circulant_tag")


@dataclasses.dataclass(frozen=True)
class RankRangeTag:
    """Marks that an operator's rank lies in the closed interval `[lo, hi]`. Use
    [`lineax.rank_range`][] to query the bounds.

    `hi=None` means that this tag does not itself constrain the upper bound (beyond the
    operator's shape); it does *not* mean infinite rank. Prefer the
    [`lineax.MaxRankTag`][], [`lineax.MinRankTag`][] and [`lineax.RankTag`][]
    constructors unless you genuinely know both a nontrivial lower *and* upper bound,
    e.g. `RankRangeTag(lo=3, hi=7)`. All four produce a `RankRangeTag`, so two tags with
    the same bounds compare equal regardless of which constructor built them.

    Rank bounds are preserved through transposition and inversion (rank is invariant
    under both). Multiple rank tags on one operator are combined into the tightest
    range they imply. They compose through `@` and `+` as:

    - `A @ B`: `hi = min(hi_A, hi_B)` and, by Sylvester's rank inequality,
        `lo = max(0, lo_A + lo_B - k)`, where `k` is the inner dimension.
    - `A + B`: `hi = min(hi_A + hi_B, in_size, out_size)` and
        `lo = min(lo_A, lo_B, max(0, lo_A - hi_B, lo_B - hi_A))`. The last term is the
        reverse triangle inequality for rank; the `min` with `lo_A` and `lo_B` keeps
        the bound valid at finite `rcond` whichever operand dominates in magnitude.

    An operator is considered rank-deficient if
    `lx.max_rank(operator) < min(operator.in_size(), operator.out_size())`. Full-rank
    solvers (e.g. `lx.AutoLinearSolver(well_posed=None/True)`) will raise a `ValueError`
    if asked to solve a rank-deficient system. Rank-deficient solvers MAY make internal
    optimisations based on [`lineax.rank_range`][]:

    - Upper bound: tagging `MaxRankTag(r)` and solving with [`lineax.SVD`][] will
        truncate to the `r` largest singular values after decomposition. As such,
        correctness may be impacted to the extent that an operator's actual rank exceeds
        `max_rank` (in exactly the same way that specifying an overly high `rcond` in
        the solver might). If a discarded singular value is above the `rcond` threshold,
        an error is raised instead.
    - Lower bound: [`lineax.SVD`][], [`lineax.HEVD`][], and (with `well_posed=False`)
        [`lineax.Diagonal`][] and [`lineax.Circulant`][] raise an error if fewer than
        `min_rank` singular values/eigenvalues lie above the `rcond` threshold. If the
        operator is known to be full rank ([`lineax.is_full_rank`][]) then they also
        skip masking the spectrum at solve time, as do [`lineax.SVD`][] and
        [`lineax.HEVD`][] whenever the rank is known exactly. Note that this means a
        full-rank-tagged but numerically ill-conditioned operator raises an error,
        rather than being silently truncated as an untagged operator would be.

    !!! info

        Any internal optimisations made by rank-deficient direct solvers are essentially
        [continuous retraction mappings](https://en.wikipedia.org/wiki/Retraction_(topology))
        of the operator A to another operator Â with rank `<=r`. Therefore, if an
        operator A exceeds `max_rank` (e.g. due to floating point roundoff error, or
        an attempt to obtain a low rank approximation of a full rank matrix), the
        solver solves against Â instead of A. In the case of SVD's singular value
        truncation the retraction mapping is the Frobenius norm projection.

        When differentiating a rank-deficient solve, the tangent dA is orthogonally
        projected onto the tangent space of the rank-r locus at Â (this is what the
        [Moore-Penrose pseudoinverse derivative](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Derivative)
        computes). Consequently derivatives are also accurate to the same degree
        that A is approximately rank r.

    **Arguments:**

    - `lo`: non-negative integer lower bound on the rank. Defaults to `0`.
    - `hi`: integer upper bound on the rank, at least `lo`, or `None` to leave the upper
        bound unconstrained. Defaults to `None`.
    """

    lo: int = 0
    hi: int | None = None

    def __post_init__(self):
        if not isinstance(self.lo, int) or self.lo < 0:
            raise ValueError(
                f"RankRangeTag.lo must be a non-negative integer, got {self.lo!r}"
            )
        if self.hi is not None and (not isinstance(self.hi, int) or self.hi < self.lo):
            raise ValueError(
                "RankRangeTag.hi must be an integer >= lo, got "
                f"hi={self.hi!r}, lo={self.lo!r}"
            )

    def __repr__(self):
        if self.lo == self.hi:
            return f"rank_tag({self.lo})"
        if self.hi is None:
            if self.lo == 0:
                return "rank_range_tag()"
            return f"min_rank_tag({self.lo})"
        if self.lo == 0:
            return f"max_rank_tag({self.hi})"
        return f"rank_range_tag({self.lo}, {self.hi})"


def MaxRankTag(r: int) -> RankRangeTag:
    """Marks that an operator's rank is no more than `r`. Shorthand for
    `RankRangeTag(hi=r)`; see [`lineax.RankRangeTag`][] for how rank bounds are
    propagated and used. Use [`lineax.max_rank`][] to query the bound.

    `MaxRankTag(0)` is valid, and represents the zero operator.

    !!! Example

        ```python
        k, n = 5, 100
        U  = lx.MatrixLinearOperator(jnp.zeros((n, k)), lx.MaxRankTag(k))
        C  = lx.MatrixLinearOperator(jnp.zeros((k, k)), lx.MaxRankTag(k))
        Vt = lx.MatrixLinearOperator(jnp.zeros((k, n)), lx.MaxRankTag(k))

        update = U @ C @ Vt
        assert lx.max_rank(update) == k   # propagated automatically through composition
        ```

    **Arguments:**

    - `r`: non-negative integer upper bound on the rank.
    """
    return RankRangeTag(hi=r)


def MinRankTag(r: int) -> RankRangeTag:
    """Marks that an operator's rank is at least `r`. Shorthand for
    `RankRangeTag(lo=r)`; see [`lineax.RankRangeTag`][] for how rank bounds are
    propagated and used. Use [`lineax.min_rank`][] to query the bound.

    !!! Example

        ```python
        k, n = 5, 100
        U  = lx.MatrixLinearOperator(jax.random.normal(key1, (n, k)), lx.RankTag(k))
        Vt = lx.MatrixLinearOperator(jax.random.normal(key2, (k, n)), lx.MinRankTag(k))

        assert lx.rank_range(U @ Vt) == (k, k)   # Sylvester: k + k - k <= rank <= k
        assert lx.rank_range(U.T) == (k, k)      # transposition preserves rank
        partial = lx.MatrixLinearOperator(
            jax.random.normal(key3, (n, n)), lx.RankRangeTag(3, 7)
        )
        assert lx.rank_range(partial) == (3, 7)
        ```

    **Arguments:**

    - `r`: non-negative integer lower bound on the rank.
    """
    return RankRangeTag(lo=r)


def RankTag(r: int) -> RankRangeTag:
    """Marks that an operator's rank is exactly `r`. Shorthand for
    `RankRangeTag(lo=r, hi=r)`; see [`lineax.RankRangeTag`][] for how rank bounds are
    propagated and used.

    **Arguments:**

    - `r`: the non-negative integer rank.
    """
    return RankRangeTag(lo=r, hi=r)


def _combine_rank_tags(tags: frozenset[object]) -> RankRangeTag | None:
    rank_tags = [t for t in tags if isinstance(t, RankRangeTag)]
    if not rank_tags:
        return None
    lo = max(t.lo for t in rank_tags)
    his = [t.hi for t in rank_tags if t.hi is not None]
    hi = min(his) if his else None
    # Contradictory tags (`lo > hi`) are a user error that the solvers' rank checks
    # surface at solve time; don't fail tag propagation over them.
    if hi is not None:
        lo = min(lo, hi)
    return RankRangeTag(lo, hi)


def tags_from_checks(operator: "AbstractLinearOperator") -> frozenset[object]:
    """Inspects an operator using all standard property checks and
    returns a frozenset of the tags that apply to it.

    This is the canonical way to collect the full set of tags for any operator,
    regardless of whether that operator stores its properties as an explicit `.tags`
    frozenset or encodes them structurally (e.g. a `DiagonalLinearOperator`
    is always diagonal).

    **Arguments:**

    - `operator`: the linear operator to inspect.

    **Returns:**

    A `frozenset` of tags.
    """
    # Lazy import to avoid a circular dependency: _operator.py imports _tags.py
    # at module level, so we defer the reverse import to call time when both
    # modules are fully initialised.
    from ._operator import (
        has_unit_diagonal,
        is_circulant,
        is_diagonal,
        is_hermitian,
        is_lower_triangular,
        is_negative_semidefinite,
        is_positive_semidefinite,
        is_semidefinite,
        is_symmetric,
        is_tridiagonal,
        is_upper_triangular,
        rank_range,
    )

    tags: set[object] = {
        tag
        for check, tag in [
            (is_symmetric, symmetric_tag),
            (is_hermitian, hermitian_tag),
            (is_diagonal, diagonal_tag),
            (is_lower_triangular, lower_triangular_tag),
            (is_upper_triangular, upper_triangular_tag),
            (is_positive_semidefinite, positive_semidefinite_tag),
            (is_negative_semidefinite, negative_semidefinite_tag),
            (is_semidefinite, semidefinite_tag),
            (has_unit_diagonal, unit_diagonal_tag),
            (is_tridiagonal, tridiagonal_tag),
            (is_circulant, circulant_tag),
        ]
        if check(operator)
    }
    dim_bound = min(operator.in_size(), operator.out_size())
    lo, hi = rank_range(operator)
    # verify that adding a rank tag wouldn't be redundant
    if lo > 0 or hi < dim_bound:
        tags.add(RankRangeTag(lo, hi))
    return frozenset(tags)


transpose_tags_rules = []


for tag in (
    symmetric_tag,
    hermitian_tag,
    unit_diagonal_tag,
    diagonal_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
    semidefinite_tag,
    tridiagonal_tag,
    circulant_tag,
):

    @transpose_tags_rules.append
    def _(tags: frozenset[object], tag=tag):
        if tag in tags:
            return tag


@transpose_tags_rules.append
def _(tags: frozenset[object]):
    if lower_triangular_tag in tags:
        return upper_triangular_tag


@transpose_tags_rules.append
def _(tags: frozenset[object]):
    if upper_triangular_tag in tags:
        return lower_triangular_tag


# Rank bounds are invariant under transposition. Multiple rank tags are combined into
# the single tightest one.
transpose_tags_rules.append(_combine_rank_tags)


def transpose_tags(tags: frozenset[object]):
    """Lineax uses "tags" to declare that a particular linear operator exhibits some
    property, e.g. symmetry.

    This function takes in a collection of tags representing a linear operator, and
    returns a collection of tags that should be associated with the transpose of that
    linear operator.

    **Arguments:**

    - `tags`: a `frozenset` of tags.

    **Returns:**

    A `frozenset` of tags.
    """
    if symmetric_tag in tags:
        return tags
    new_tags = []
    for rule in transpose_tags_rules:
        out = rule(tags)
        if out is not None:
            new_tags.append(out)
    return frozenset(new_tags)


invert_tags_rules = []


for tag in (
    symmetric_tag,
    hermitian_tag,
    diagonal_tag,
    lower_triangular_tag,
    upper_triangular_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
    semidefinite_tag,
    circulant_tag,
):

    @invert_tags_rules.append
    def _(tags: frozenset[object], tag=tag):
        if tag in tags:
            return tag


@invert_tags_rules.append
def _(tags: frozenset[object]):
    if unit_diagonal_tag in tags and (
        diagonal_tag in tags
        or lower_triangular_tag in tags
        or upper_triangular_tag in tags
    ):
        return unit_diagonal_tag


# Rank bounds are invariant under (pseudo)inversion.
invert_tags_rules.append(_combine_rank_tags)


# tridiagonal_tag intentionally absent: inverse of tridiagonal matrix generally dense.


def invert_tags(tags: frozenset[object]) -> frozenset[object]:
    """Lineax uses "tags" to declare that a particular linear operator exhibits some
    property, e.g. symmetry.

    This function takes in a collection of tags representing a linear operator, and
    returns a collection of tags that should be associated with the (pseudo)inverse
    of that linear operator.

    Most structural properties are preserved by inversion (symmetric, diagonal,
    triangular, positive/negative semidefinite).  Notable exceptions:

    - `tridiagonal_tag` is **not** preserved — the inverse of a tridiagonal matrix
      is generally dense.
    - `unit_diagonal_tag` is only preserved when the operator is also diagonal or
      triangular.

    **Arguments:**

    - `tags`: a `frozenset` of tags.

    **Returns:**

    A `frozenset` of tags.
    """
    new_tags = []
    for rule in invert_tags_rules:
        out = rule(tags)
        if out is not None:
            new_tags.append(out)
    return frozenset(new_tags)
