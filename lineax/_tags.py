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

from collections.abc import Callable, Iterable
from typing import Any


class _HasRepr:
    def __init__(self, string: str):
        self.string = string

    def __repr__(self):
        return self.string


symmetric_tag = _HasRepr("symmetric_tag")
diagonal_tag = _HasRepr("diagonal_tag")
tridiagonal_tag = _HasRepr("tridiagonal_tag")
unit_diagonal_tag = _HasRepr("unit_diagonal_tag")
lower_triangular_tag = _HasRepr("lower_triangular_tag")
upper_triangular_tag = _HasRepr("upper_triangular_tag")
positive_semidefinite_tag = _HasRepr("positive_semidefinite_tag")
negative_semidefinite_tag = _HasRepr("negative_semidefinite_tag")


def tags_from_checks(
    operator: Any,
    check_tag_pairs: Iterable[tuple[Callable[..., bool], object]],
) -> frozenset[object]:
    """Lineax uses "tags" to declare that a particular linear operator exhibits some
    property, e.g. symmetry.

    This function collects tags for an operator by evaluating a sequence of
    `(check, tag)` pairs and returning a frozenset of the tags whose check returns
    `True`.  The checks are typically the singledispatch predicate functions
    (`is_symmetric`, `is_diagonal`, etc.) defined in `lineax._operator`.

    **Arguments:**

    - `operator`: the linear operator to inspect.
    - `check_tag_pairs`: an iterable of `(check, tag)` tuples.  For each pair,
      `check(operator)` is called and the corresponding `tag` is included in the
      result if it returns `True`.

    **Returns:**

    A `frozenset` of tags.
    """
    return frozenset(tag for check, tag in check_tag_pairs if check(operator))


transpose_tags_rules = []


for tag in (
    symmetric_tag,
    unit_diagonal_tag,
    diagonal_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
    tridiagonal_tag,
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
    diagonal_tag,
    lower_triangular_tag,
    upper_triangular_tag,
    positive_semidefinite_tag,
    negative_semidefinite_tag,
):

    @invert_tags_rules.append
    def _(tags: frozenset[object], tag=tag):
        if tag in tags:
            return tag


@invert_tags_rules.append
def _(tags: frozenset[object]):
    # The inverse of a unit-diagonal triangular (or diagonal) matrix is also
    # unit-diagonal.  The unit-diagonal property alone is not sufficient — a
    # general matrix with 1s on the diagonal does not have this property for
    # its inverse.
    if unit_diagonal_tag in tags and (
        diagonal_tag in tags
        or lower_triangular_tag in tags
        or upper_triangular_tag in tags
    ):
        return unit_diagonal_tag


# tridiagonal_tag is intentionally absent: the inverse of a tridiagonal matrix
# is generally dense.


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
      triangular, because only unit-diagonal triangular (and diagonal) matrices
      have unit-diagonal inverses.

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
