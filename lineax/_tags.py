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


def tags_from_checks(operator: Any) -> frozenset[object]:
    """Lineax uses "tags" to declare that a particular linear operator exhibits some
    property, e.g. symmetry.

    This function inspects an operator using all standard property checks and
    returns a frozenset of the tags that apply to it.  It is the canonical way
    to collect the full set of tags for any operator, regardless of whether that
    operator stores its properties as an explicit `.tags` frozenset or encodes
    them structurally (e.g. `DiagonalLinearOperator`).

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
        is_diagonal,
        is_lower_triangular,
        is_negative_semidefinite,
        is_positive_semidefinite,
        is_symmetric,
        is_tridiagonal,
        is_upper_triangular,
    )

    return frozenset(
        tag
        for check, tag in [
            (is_symmetric, symmetric_tag),
            (is_diagonal, diagonal_tag),
            (is_lower_triangular, lower_triangular_tag),
            (is_upper_triangular, upper_triangular_tag),
            (is_positive_semidefinite, positive_semidefinite_tag),
            (is_negative_semidefinite, negative_semidefinite_tag),
            (has_unit_diagonal, unit_diagonal_tag),
            (is_tridiagonal, tridiagonal_tag),
        ]
        if check(operator)
    )


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
