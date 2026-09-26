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

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond, unit_phase
from .._operator import (
    AbstractLinearOperator,
    diagonal,
    has_unit_diagonal,
    is_diagonal,
    rank_range,
)
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


# The static flag records that the operator is certified full rank, so that `compute`
# need not mask out (near-)zero diagonal entries.
_DiagonalState: TypeAlias = tuple[Array | None, eqxi.Static[bool], PackedStructures]


class Diagonal(AbstractDirectLinearSolver[_DiagonalState]):
    """Diagonal solver for linear systems.

    Requires that the operator be diagonal. Then $Ax = b$, with $A = diag[a]$, is
    solved simply by doing an elementwise division $x = b / a$.

    This solver can handle singular operators (i.e. diagonal entries with value 0).
    """

    well_posed: bool = False
    rcond: float | None = None

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _DiagonalState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Diagonal` may only be used for linear solves with square matrices"
            )
        if not is_diagonal(operator):
            raise ValueError(
                "`Diagonal` may only be used for linear solves with diagonal matrices"
            )
        packed_structures = pack_structures(operator)
        if has_unit_diagonal(operator):
            return None, eqxi.Static(True), packed_structures
        diag = diagonal(operator)
        lo, _ = rank_range(operator)
        (size,) = diag.shape
        if not self.well_posed and lo > 0:
            # `compute` masks out entries `|d_i| <= rcond * max|d|`, so an operator
            # declared to have rank at least `lo` must have at least `lo` entries
            # surviving that mask.
            rcond = resolve_rcond(self.rcond, size, size, diag.dtype)
            abs_diag = jnp.abs(diag)
            diag = eqx.error_if(
                diag,
                jnp.sum(abs_diag > rcond * jnp.max(abs_diag)) < lo,
                "lineax.Diagonal: the operator was declared (via a "
                "`MinRankTag`/`RankTag`, or by composition rules) to have rank at "
                f"least {lo}, but fewer than {lo} of its diagonal entries lie above "
                "the rcond threshold in magnitude. Either the rank claim is "
                "incorrect, or the operator is too ill-conditioned for its rank to be "
                "resolved at this `rcond` (e.g. a composition of individually "
                "well-conditioned factors). Remove/loosen the rank tag, decrease "
                "`rcond`, or set `EQX_ON_ERROR=off` to skip this check.",
            )
        # If the operator is known to be full rank, every entry is certified above the
        # cutoff by the check above. (Note that this means an ill-conditioned operator
        # tagged as full rank raises an error, rather than being silently masked as an
        # untagged one would be.)
        return diag, eqxi.Static(lo == size), packed_structures

    def compute(
        self, state: _DiagonalState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        diag, full_rank, packed_structures = state
        del state, options
        unit_diagonal = diag is None
        vector = ravel_vector(vector, packed_structures)
        if unit_diagonal:
            solution = vector
        else:
            if not (self.well_posed or full_rank.value):
                (size,) = diag.shape
                rcond = resolve_rcond(self.rcond, size, size, diag.dtype)
                abs_diag = jnp.abs(diag)
                diag = jnp.where(abs_diag > rcond * jnp.max(abs_diag), diag, jnp.inf)  # pyright: ignore
            solution = vector / diag
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _DiagonalState, options: dict[str, Any]):
        del options
        diag, full_rank, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = diag, full_rank, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _DiagonalState, options: dict[str, Any]):
        del options
        diag, full_rank, packed_structures = state
        if diag is None:
            conj_diag = None
        else:
            conj_diag = diag.conj()
        conj_options = {}
        conj_state = conj_diag, full_rank, packed_structures
        return conj_state, conj_options

    def slogdet(
        self, state: _DiagonalState, options: dict[str, Any]
    ) -> tuple[Array, Array]:
        del options
        diag, full_rank, packed_structures = state
        if diag is None:
            # A unit diagonal has determinant one. `sign` takes the operator's dtype,
            # so a complex operator yields a complex `sign`, matching the other paths.
            leaves, treedef = packed_structures.value
            out_structure, _ = jtu.tree_unflatten(treedef, leaves)
            with jax.numpy_dtype_promotion("standard"):
                dtype = jnp.result_type(*jtu.tree_leaves(out_structure))
            return jnp.ones((), dtype), jnp.zeros((), jnp.finfo(dtype).dtype)
        if not (self.well_posed or full_rank.value):
            (size,) = diag.shape
            rcond = resolve_rcond(self.rcond, size, size, diag.dtype)
            abs_diag = jnp.abs(diag)
            mask = abs_diag > rcond * jnp.max(abs_diag)
            safe_diag = jnp.where(mask, diag, 1.0)
            sign = jnp.prod(unit_phase(safe_diag))
            lad = jnp.sum(jnp.where(mask, jnp.log(abs_diag), 0.0))
        else:
            sign = jnp.prod(unit_phase(diag))
            lad = jnp.sum(jnp.log(jnp.abs(diag)))
        return sign, lad

    def assume_full_rank(self):
        return self.well_posed


Diagonal.__init__.__doc__ = """**Arguments**:

- `well_posed`: if `False`, then singular operators are accepted, and the pseudoinverse
    solution is returned. If `True` then passing a singular operator will cause an error
    to be raised instead.
- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `N`, where `N` is the input (or output) size of the operator.
    Only used if `well_posed=False`
"""
