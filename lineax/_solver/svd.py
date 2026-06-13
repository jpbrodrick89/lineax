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

import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator
from .._solution import RESULTS
from .._solve import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_SVDState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class SVD(AbstractDirectLinearSolver[_SVDState]):
    """SVD solver for linear systems.

    This solver can handle any operator, even nonsquare or singular ones. In these
    cases it will return the pseudoinverse solution to the linear system.

    Equivalent to `scipy.linalg.lstsq`.
    """

    rcond: float | None = None

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        svd = jsp.linalg.svd(operator.as_matrix(), full_matrices=False)
        packed_structures = pack_structures(operator)
        return svd, packed_structures

    def compute(
        self,
        state: _SVDState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        (u, s, vt), packed_structures = state
        vector = ravel_vector(vector, packed_structures)
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
        (u, s, vt), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_state = (vt.T, s, u.T), transposed_packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _SVDState, options: dict[str, Any]):
        del options
        (u, s, vt), packed_structures = state
        conj_state = (u.conj(), s, vt.conj()), packed_structures
        conj_options = {}
        return conj_state, conj_options

    def slogdet(self, state: _SVDState, options: dict[str, Any]) -> tuple[Array, Array]:
        del options
        (u, s, vt), _ = state
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
        float_dtype = np.result_type(s.dtype, np.float32)
        sign = jnp.full((), jnp.nan, dtype=float_dtype)
        return sign, lad

    def assume_full_rank(self):
        return False


SVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `max(N, M)`, where `(N, M)` is the shape of the operator. (I.e.
    `N` is the output size and `M` is the input size.)
"""
