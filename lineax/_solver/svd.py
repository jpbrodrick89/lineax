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
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator, rank_range
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


# The static pair is the operator's rank range `(lo, hi)`, as certified in `init`.
# `compute` uses only the leading `hi` singular components, and when `lo == hi` every
# one of those is known to lie above the rcond cutoff, so it skips masking. The full
# decomposition is kept, so that the trailing components (e.g. a null space) remain
# recoverable.
_SVDState: TypeAlias = tuple[
    tuple[Array, Array, Array], eqxi.Static[tuple[int, int]], PackedStructures
]


def _retained(state: _SVDState) -> tuple[Array, Array, Array]:
    (u, s, vt), ranks, _ = state
    _, hi = ranks.value
    return u[:, :hi], s[:hi], vt[:hi, :]


class SVD(AbstractDirectLinearSolver[_SVDState]):
    """SVD solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Equivalent to `scipy.linalg.lstsq`.

    If the operator is Hermitian (e.g. real-symmetric, or positive/negative
    semidefinite) then [`lineax.HEVD`][] is usually faster, as a Hermitian
    eigendecomposition is cheaper than a general SVD.
    """

    rcond: float | None = None

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        u, s, vt = jsp.linalg.svd(operator.as_matrix(), full_matrices=False)
        lo, hi = rank_range(operator)
        if lo > 0 or hi < s.shape[0]:
            # `compute` masks out `s_i <= rcond * s[0]`, so that is the cutoff that any
            # claim about the rank is checked against.
            # (s.size > 0 since 0 < lo <= size or hi < size.)
            m, n = u.shape[0], vt.shape[1]
            rcond = resolve_rcond(self.rcond, n, m, s.dtype) * s[0]
            if lo > 0:
                # If the operator is known to have rank at least `lo`, its leading `lo`
                # singular values must survive the mask. `s` is descending, so testing
                # `s[lo - 1]` certifies them all.
                s = eqx.error_if(
                    s,
                    s[lo - 1] <= rcond,
                    "lineax.SVD: the operator was declared (via a "
                    "`MinRankTag`/`RankTag`, or by composition rules) to have rank at "
                    f"least {lo}, but its {lo}-th singular value falls at or below "
                    "the rcond threshold. Either the rank claim is incorrect, or the "
                    "operator is too ill-conditioned for its rank to be resolved at "
                    "this `rcond` (e.g. a composition of individually "
                    "well-conditioned factors). Remove/loosen the rank tag, decrease "
                    "`rcond`, or set `EQX_ON_ERROR=off` to skip this check.",
                )
            if hi < s.shape[0]:
                # If the operator is known to have rank at most `hi`, the trailing
                # singular values are mathematically zero, so `compute` statically
                # truncates to the leading `hi` components. The state itself is not
                # truncated, as that would discard the trailing components
                # irrecoverably; if they are never used then JAX eliminates them as dead
                # code. Truncating is lossless iff the tail all sits below the cutoff.
                # Otherwise the `max_rank` claim is false and truncating would change
                # the solution. `s` is descending, so testing the largest discarded
                # value `s[hi]` certifies the tail.
                s = eqx.error_if(
                    s,
                    s[hi] > rcond,
                    "lineax.SVD: the operator was declared (via a `MaxRankTag`/"
                    f"`RankTag`, or by composition rules) to have rank at most {hi}, "
                    "but it has a singular value above the rcond threshold beyond "
                    "that rank. Truncating to the declared rank would change the "
                    "solution, so the rank claim appears to be incorrect. "
                    "Remove/loosen the rank tag, increase `rcond` if you intend a "
                    "low-rank approximation, or set `EQX_ON_ERROR=off` to skip this "
                    "check.",
                )
        # If the rank is known exactly (`lo == hi`) then the `hi` singular values that
        # `compute` retains are all certified above the cutoff by the checks above, so
        # it need not mask them. (Note that this means an ill-conditioned operator
        # tagged as full rank raises an error, rather than being silently truncated as
        # an untagged one would be.)
        packed_structures = pack_structures(operator)
        return (u, s, vt), eqxi.Static((lo, hi)), packed_structures

    def compute(
        self,
        state: _SVDState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        _, ranks, packed_structures = state
        u, s, vt = _retained(state)
        vector = ravel_vector(vector, packed_structures)
        lo, hi = ranks.value
        if lo == hi:
            # Every retained singular value was certified above the cutoff in `init`.
            rank = jnp.array(hi, dtype=int)
            s_inv = (jnp.array(1.0) / s).astype(u.dtype)
        else:
            m, _ = u.shape
            _, n = vt.shape
            rcond = resolve_rcond(self.rcond, n, m, s.dtype)
            rcond = jnp.array(rcond, dtype=s.dtype)
            if s.size > 0:
                rcond = rcond * s[0]
            # Not >=, or this fails with a matrix of all-zeros.
            mask = s > rcond
            rank = mask.sum()
            safe_s = jnp.where(mask, s, 1)
            s_inv = jnp.where(mask, jnp.array(1.0) / safe_s, 0).astype(u.dtype)
        uTb = jnp.matmul(u.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(vt.conj().T, s_inv * uTb, precision=lax.Precision.HIGHEST)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {"rank": rank}

    def transpose(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), ranks, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (vt.T, s, u.T), ranks, transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), ranks, packed_structures = state
        conj_state = (u.conj(), s, vt.conj()), ranks, packed_structures
        conj_options = {}
        return conj_state, conj_options

    def slogdet(self, state: _SVDState, options: dict[str, Any]) -> tuple[Array, Array]:
        del options
        u, s, vt = _retained(state)
        m, _ = u.shape
        _, n = vt.shape
        rcond = resolve_rcond(self.rcond, n, m, s.dtype)
        rcond_arr = jnp.array(rcond, dtype=s.dtype)
        if s.size > 0:
            threshold = rcond_arr * s[0]
        else:
            threshold = rcond_arr
        mask = s > threshold
        # Log-pseudodeterminant: sum of logs of non-zero singular values only.
        # Zero singular values (below threshold) contribute 0 via log(1) = 0.
        # For full-rank operators this equals the true logabsdet.
        safe_s = jnp.where(mask, s, 1.0)
        lad = jnp.sum(jnp.log(safe_s))
        # Sign is not recoverable from SVD alone:
        #   full-rank square: needs sign(det(U)) * sign(det(V^T)), O(n^3) extra work
        #   full-rank rectangular: future work via QR Householder vectors
        #   rank-deficient: needs an eigensolver for pseudodeterminant sign
        sign = jnp.full((), jnp.nan, dtype=s.dtype)
        return sign, lad

    def assume_full_rank(self):
        return False


SVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""
