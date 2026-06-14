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
import jax.lax as lax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._misc import resolve_rcond
from .._operator import AbstractLinearOperator, is_hermitian, max_rank
from .._solution import RESULTS
from .._solve import AbstractLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    unravel_solution,
)


_HEVDState: TypeAlias = tuple[tuple[Array, Array], PackedStructures]


class HEVD(AbstractLinearSolver[_HEVDState]):
    """Hermitian eigenvalue decomposition solver for linear systems.

    The operator must be Hermitian (self-adjoint), i.e. tagged with
    [`lineax.hermitian_tag`][] (or be real-symmetric, positive/negative semidefinite,
    etc.). Unlike [`lineax.Cholesky`][] the operator need not be definite or even
    nonsingular: in the singular case this solver returns the pseudoinverse solution,
    just like [`lineax.SVD`][].

    This is the Hermitian analogue of [`lineax.SVD`][]: for a Hermitian `A` the
    eigendecomposition `A = V diag(w) V^H` (with `V` unitary and `w` real) is both an
    eigendecomposition and -- up to signs -- a singular value decomposition, so the
    same pseudoinverse machinery applies but using a single (cheaper) Hermitian
    eigensolve.
    """

    rcond: float | None = None

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if not is_hermitian(operator):
            raise ValueError(
                "`HEVD()` may only be used for Hermitian (self-adjoint) linear "
                "operators. (Real-symmetric operators are Hermitian.)"
            )
        w, v = jnp.linalg.eigh(operator.as_matrix())
        # `jnp.linalg.eigh` returns eigenvalues in ascending (signed) order. In the
        # common case we leave them in that order: `compute` masks the small
        # eigenvalues by magnitude and so does not care about the ordering.
        r = max_rank(operator)
        if r < w.shape[0]:
            # The operator is declared to have rank at most `r`, so all but the `r`
            # largest-magnitude eigenvalues are mathematically zero. Statically drop
            # them to shrink the matmuls (and storage) in `compute`.
            #
            # Unlike `SVD`'s singular values -- already sorted descending, so
            # truncation is a free slice -- eigenvalues are signed and ascending, so
            # the small-magnitude ones sit in the *interior* of the spectrum.
            # Selecting the `r` largest-magnitude therefore needs a reordering
            # gather. We only pay for it when a rank tag is actually present, and it
            # is O(n^2): dominated by the O(n^3) eigendecomposition above.
            order = jnp.argsort(jnp.abs(w))[::-1]
            w = w[order]
            v = v[:, order]
            # `compute` masks out `|w_i| <= rcond * max|w|`, so dropping the tail is
            # lossless iff it all sits below that floor (using the same rcond).
            # Otherwise the `max_rank` claim is false and truncating would change
            # the solution. `w` is now sorted by descending magnitude, so testing
            # the largest discarded value `|w[r]|` certifies the tail.
            m = v.shape[0]
            # w.size > 0 since r < size
            rcond = resolve_rcond(self.rcond, m, m, w.dtype) * jnp.abs(w[0])
            w = eqx.error_if(
                w,
                jnp.abs(w[r]) > rcond,
                "lineax.HEVD: the operator was declared (via a `MaxRankTag`, or by "
                f"composition rules) to have rank at most {r}, but it has an "
                "eigenvalue above the rcond threshold beyond that rank. Truncating to "
                "the declared rank would change the solution, so the rank claim "
                "appears to be incorrect. Remove/loosen the rank tag, increase "
                "`rcond` if you intend a low-rank approximation, or set "
                "`EQX_ON_ERROR=off` to skip this check.",
            )
            w = w[:r]
            v = v[:, :r]
        packed_structures = pack_structures(operator)
        return (w, v), packed_structures

    def compute(
        self,
        state: _HEVDState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        (w, v), packed_structures = state
        vector = ravel_vector(vector, packed_structures)
        m = v.shape[0]
        rcond = resolve_rcond(self.rcond, m, m, w.dtype)
        rcond = jnp.array(rcond, dtype=w.dtype)
        abs_w = jnp.abs(w)
        if w.size > 0:
            # `w` is not assumed sorted, so take the largest magnitude directly.
            rcond = rcond * jnp.max(abs_w)
        # Not >=, or this fails with a matrix of all-zeros.
        mask = abs_w > rcond
        rank = mask.sum()
        safe_w = jnp.where(mask, w, 1)
        w_inv = jnp.where(mask, jnp.array(1.0) / safe_w, 0).astype(v.dtype)
        # A = V diag(w) V^H  =>  A^+ b = V diag(w_inv) V^H b
        vHb = jnp.matmul(v.conj().T, vector, precision=lax.Precision.HIGHEST)
        solution = jnp.matmul(v, w_inv * vHb, precision=lax.Precision.HIGHEST)
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {"rank": rank}

    def transpose(self, state: _HEVDState, options: dict[str, Any]):
        del options
        (w, v), packed_structures = state
        # `A` is Hermitian, so `A^T = conj(A) = conj(V) diag(w) conj(V)^H` (with `w`
        # real). The structure is square and symmetric, so packed structures are
        # unchanged.
        transpose_state = (w, v.conj()), packed_structures
        transpose_options = {}
        return transpose_state, transpose_options

    def conj(self, state: _HEVDState, options: dict[str, Any]):
        del options
        (w, v), packed_structures = state
        # `A` is Hermitian, so `conj(A) = conj(V) diag(w) conj(V)^H` (with `w` real).
        conj_state = (w, v.conj()), packed_structures
        conj_options = {}
        return conj_state, conj_options

    def assume_full_rank(self):
        return False


HEVD.__init__.__doc__ = """**Arguments**:

- `rcond`: the cutoff for handling zero entries on the diagonal. Defaults to machine
    precision times `N`, where `(N, N)` is the shape of the operator.
"""
