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
from jaxtyping import Array, PyTree

from .._misc import unit_phase
from .._operator import AbstractLinearOperator, is_tridiagonal, tridiagonal
from .._solution import RESULTS
from .base import AbstractDirectLinearSolver
from .misc import (
    pack_structures,
    PackedStructures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)


_TridiagonalState: TypeAlias = tuple[tuple[Array, Array, Array], PackedStructures]


class Tridiagonal(AbstractDirectLinearSolver[_TridiagonalState]):
    """Tridiagonal solver for linear systems, uses the LAPACK/cusparse implementation
    of Gaussian elimination with partial pivotting (which increases stability).
    ."""

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]):
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Tridiagonal` may only be used for linear solves with square matrices"
            )
        if not is_tridiagonal(operator):
            raise ValueError(
                "`Tridiagonal` may only be used for linear solves with tridiagonal "
                "matrices"
            )
        return tridiagonal(operator), pack_structures(operator)

    def compute(
        self,
        state: _TridiagonalState,
        vector,
        options,
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        del state, options
        vector = ravel_vector(vector, packed_structures)

        solution = lax.linalg.tridiagonal_solve(
            jnp.append(0.0, lower_diagonal),
            diagonal,
            jnp.append(upper_diagonal, 0.0),
            vector[:, None],
        ).flatten()

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _TridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        transpose_diagonals = (diagonal, upper_diagonal, lower_diagonal)
        transpose_state = (transpose_diagonals, transposed_packed_structures)
        return transpose_state, options

    def conj(self, state: _TridiagonalState, options: dict[str, Any]):
        (diagonal, lower_diagonal, upper_diagonal), packed_structures = state
        conj_diagonals = (diagonal.conj(), lower_diagonal.conj(), upper_diagonal.conj())
        conj_state = (conj_diagonals, packed_structures)
        return conj_state, options

    def slogdet(
        self, state: _TridiagonalState, options: dict[str, Any]
    ) -> tuple[Array, Array]:
        del options
        (diagonal, lower_diagonal, upper_diagonal), _ = state
        if diagonal.shape[0] == 1:
            return unit_phase(diagonal[0]), jnp.log(jnp.abs(diagonal[0]))
        # Two evaluations of the same three-term minor recurrence, differing only in how
        # they associate it: sequentially, which is optimal on CPU, or as a tree, which
        # is what GPUs need. See the block comment below the class.
        return lax.platform_dependent(
            diagonal,
            lower_diagonal,
            upper_diagonal,
            cpu=_slogdet_sequential,
            default=_slogdet_parallel,
        )

    def assume_full_rank(self):
        return True


Tridiagonal.__init__.__doc__ = """**Arguments:**

Nothing.
"""


# ----------------------------------------------------------------------------------
# Determinants.
#
# Both implementations evaluate the same three-term recurrence on the leading
# principal minors D_i = det(A[:i+1, :i+1]):
#
#     D_i = d_i D_{i-1} - l_{i-1} u_{i-1} D_{i-2},      D_{-1} = 1, D_0 = d_0
#
# with det(A) = D_{n-1}. This is division-free, unlike the LU-pivot recurrence
# p_i = d_i - l_{i-1} u_{i-1} / p_{i-1}, which divides by zero as soon as any *leading*
# minor is singular -- even when A itself is invertible, e.g. [[0, 1], [1, 0]].
#
# They differ only in how the recurrence is associated:
#
#   * `_slogdet_sequential` walks it in order, which is optimal on CPU.
#   * `_slogdet_parallel` rewrites it as a product of 2x2 transfer matrices and
#     evaluates that product with a tree, which is what GPUs need: there, each
#     `lax.scan` iteration costs a kernel launch (~8us) whatever the work inside it,
#     so the sequential form costs ~8us * n / block: on an A100 in float64 that is 15x
#     slower at n = 512, 199x at n = 8192, and 3377x at n = 131072.
#
# `Tridiagonal.slogdet` picks between them with `lax.platform_dependent`. `lx.slogdet`'s
# JVP rule differentiates that, so each platform also reverses the implementation it
# uses for the primal, which is what you want on both. On GPU the tree's primal
# advantage decides it. On CPU the scan wins on both counts: in float64 at n = 8192 its
# primal is 85us against 171us, and its backward pass costs a further 1.4-1.8x against
# 3.9-12.3x for the tree (reversing a scan is a second serial pass over stored
# residuals; reversing a tree is another tree, but over much more data).
# ----------------------------------------------------------------------------------

# Number of raw recurrence steps between renormalisations. Bigger blocks amortise the
# renormalisation over more steps; smaller blocks let the minors decay further before a
# block underflows. A block underflows once the minors decay past float range within it,
# i.e. roughly when R * K > 308 for an operator whose entries span 10**-R, so this caps
# the tolerated grading. Largest R that both implementations get right, in float64,
# over six draws of `test_tridiagonal_slogdet_graded`'s construction at n = 128:
#
#     K:                 2     4     8    16    32
#     sequential:      108    66    38    20    10
#     parallel:         68    66    38    20    10
#     heuristic 308/K: 154    77    38    19    10
#
# The heuristic is accurate for K >= 8 and optimistic below it, where something else
# binds first: the tree in `_slogdet_parallel` has its own ceiling near 68 decades,
# because a tree group spans exponentially many steps between renormalisations. At
# K = 4 both paths break at the same R, so the GPU path costs no grading tolerance.
#
# 4 is also the smallest value that still beats LAPACK `gttrf` on CPU at every size
# (1.1-1.5x unbatched, 2.2-3.0x batched). Going to 2 buys tolerance only on CPU, only
# reaches parity with `gttrf` unbatched, and is *less* safe rather than more: its
# per-block decay lands inside the denormal band, so instead of underflowing cleanly
# to -inf it returns a silently wrong answer (~5e-4 relative) for R in 120..150.
#
# 8 would be 1.3x faster again on CPU (float64, every size measured), but its limit of
# 38 decades sits just below the
# 40 that `test_tridiagonal_slogdet_graded` pins, so it fails outright -- and on
# GPU it fails silently rather than underflowing to -inf. It buys nothing there anyway:
# `_slogdet_parallel` is within noise between 4 and 8 at every size measured, because
# its launch count is `block + log_radix(n / block)` rather than `n / block`.
_SLOGDET_BLOCK = 4

# Fan-in of the tree in `_slogdet_parallel`: how many chunk transfer matrices are
# multiplied together between renormalisations, so it only affects `_slogdet_parallel`
# and hence only GPU. 2 is reliably the slowest (fewer matrices per level means more
# levels, hence more kernel launches); 4, 8 and 16 land within run-to-run scatter of
# each other, individual runs disagreeing by ~15% on which is quickest (A100, float64,
# n = 131072: 64, 49, 42, 48us of device time). 4 is the smallest of those, which is the
# safe end: fewer products between renormalisations, and radix 2 is ~3x more accurate
# than 4 on near-defective operators (1.6e-9 against 5.2e-9 relative in float64 at
# n = 262144) if that ever matters more than speed.
_SLOGDET_RADIX = 4


def _prescale(
    diagonal: Array, lower_diagonal: Array, upper_diagonal: Array
) -> tuple[Array, Array, Array]:
    """Rescale the operator by a power of two so that every entry is bounded by 1.

    The minors grow or decay by roughly one entry-magnitude per step, so for an operator
    whose entries are far from unit scale they leave float range fast. Normalising the
    whole operator first uses det(A) = sigma**n det(A / sigma), i.e.

        logabsdet(A) = n log(sigma) + logabsdet(A / sigma)

    Choosing sigma as a power of two makes the rescale exact, and choosing it to bound
    both scaled arrays by 1 bounds the growth to |D_i| <= |D_{i-1}| + |D_{i-2}|, i.e. at
    most 2**block per block -- so the recurrence can no longer overflow at all. This is
    a single reduction, outside the serial loop.

    Returns the scaled diagonal, the scaled `coupling` l_i u_i, and log2(sigma).
    """
    dtype = diagonal.dtype
    real_dtype = jnp.finfo(dtype).dtype
    zero = jnp.zeros((), real_dtype)
    # Take the magnitude from the raw entries rather than from `lower * upper`:
    # forming that product first overflows to `inf` (and thence `nan`) whenever
    # |entries| > sqrt(max_float), which is only ~1.8e19 in float32 -- precisely the
    # badly-scaled operators this rescale exists to handle.
    magnitude = jnp.maximum(
        jnp.max(jnp.abs(diagonal), initial=zero),
        jnp.maximum(
            jnp.max(jnp.abs(lower_diagonal), initial=zero),
            jnp.max(jnp.abs(upper_diagonal), initial=zero),
        ),
    )
    _, exponent = jnp.frexp(magnitude)
    sigma_inv = jnp.ldexp(jnp.ones((), real_dtype), -exponent).astype(dtype)
    coupling = (lower_diagonal * sigma_inv) * (upper_diagonal * sigma_inv)
    return diagonal * sigma_inv, coupling, exponent


def _pad_steps(
    diagonal: Array, coupling: Array, block: int
) -> tuple[Array, Array, int]:
    """Pad the `n - 1` recurrence steps up to a whole number of blocks.

    The padding steps are `(d, coupling) = (1, 0)`, which map D_i -> D_{i-1} and so
    leave the value alone. They go at the very end.
    """
    dtype = diagonal.dtype
    steps = diagonal.shape[0] - 1
    pad = (-steps) % block
    if pad:
        diagonal = jnp.concatenate([diagonal[1:], jnp.ones((pad,), dtype)])
        coupling = jnp.concatenate([coupling, jnp.zeros((pad,), dtype)])
    else:
        diagonal = diagonal[1:]
    return diagonal, coupling, (steps + pad) // block


def _slogdet_sequential(
    diagonal: Array, lower_diagonal: Array, upper_diagonal: Array
) -> tuple[Array, Array]:
    """Walk the minor recurrence in order. Optimal on CPU."""
    n = diagonal.shape[0]
    dtype = diagonal.dtype
    real_dtype = jnp.finfo(dtype).dtype
    block = _SLOGDET_BLOCK
    diagonal, coupling, exponent = _prescale(diagonal, lower_diagonal, upper_diagonal)

    def unit_step(carry, args):
        prev, curr = carry
        d_i, coupling_i = args
        return (curr, d_i * curr - coupling_i * prev), None

    def block_step(carry, args):
        (prev, curr), _ = lax.scan(unit_step, carry, args, unroll=block)
        scale = jnp.maximum(jnp.abs(prev), jnp.abs(curr))
        # A singular A drives both minors to exactly zero. Dividing by `scale` would
        # turn that into `nan`; holding the pair at zero instead lets the `log(scale)`
        # sum absorb the `-inf` and yields `(sign, lad) = (0, -inf)`, matching
        # `jnp.linalg.slogdet` on a singular input.
        nonzero = jnp.where(scale == 0, 1.0, scale)
        # Reciprocal-and-multiply rather than two divides: measurably faster, and
        # `nonzero >=` the larger minor keeps the reciprocal in range.
        inv_scale = (1.0 / nonzero).astype(dtype)
        # The scales are emitted rather than accumulated so that their logs happen in
        # one vectorised pass instead of once per step of the serial loop.
        return (prev * inv_scale, curr * inv_scale), scale

    diagonal_rest, coupling_rest, num_blocks = _pad_steps(diagonal, coupling, block)

    # Normalise the initial pair (D_{-1}, D_0) = (1, d_0) the same way.
    scale0 = jnp.maximum(jnp.abs(diagonal[0]), 1.0)
    init = (
        jnp.ones((), dtype) / scale0.astype(dtype),
        diagonal[0] / scale0.astype(dtype),
    )
    (_, det), scales = lax.scan(
        block_step,
        init,
        (
            diagonal_rest.reshape(num_blocks, block),
            coupling_rest.reshape(num_blocks, block),
        ),
    )
    lad = (
        n * exponent.astype(real_dtype) * jnp.log(jnp.array(2.0, real_dtype))
        + jnp.log(scale0)
        + jnp.sum(jnp.log(scales))
        + jnp.log(jnp.abs(det))
    )
    return unit_phase(det), lad


# The same recurrence, written as a product of 2x2 transfer matrices:
#
#     (D_{i-1}, D_i) = M_i (D_{i-2}, D_{i-1}),    M_i = [[0, 1], [-l u, d_i]]
#
# so det(A) is the second component of (M_{n-1} ... M_1) (1, d_0). Matrix products
# associate, so the product can be evaluated as a tree of depth log(n) rather than a
# chain of length n. The price is 4 multiplies per step instead of 2 (a chunk must
# propagate both basis vectors, not knowing its incoming pair), which is irrelevant on
# a GPU that was launch-bound anyway.
#
# The matrices are carried as four separate arrays rather than one `(m, 2, 2)` array:
# `a @ b` on stacked 2x2s lowers to a batched `dot_general`, i.e. an unfusable cuBLAS
# call. At n = 2048 x batch 4096 in float64 that costs 25.7ms against 2.9-3.5ms for the
# component-wise form, a 7-9x difference.


def _mul(hi: tuple, lo: tuple) -> tuple:
    """Product of two 2x2 matrices held as component tuples: `hi @ lo`."""
    a1, b1, c1, d1 = hi
    a2, b2, c2, d2 = lo
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
    )


def _normalise(m: tuple, log_scale: Array) -> tuple[tuple, Array]:
    """Divide out the largest entry, accumulating its log."""
    a, b, c, d = m
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(a), jnp.abs(b)), jnp.maximum(jnp.abs(c), jnp.abs(d))
    )
    # As in `_slogdet_sequential`: hold an exactly-singular block at zero rather than
    # producing `nan`, and let the `-inf` appear in the final log.
    inv = (1.0 / jnp.where(scale == 0, 1.0, scale)).astype(a.dtype)
    return (a * inv, b * inv, c * inv, d * inv), log_scale + jnp.log(scale)


def _reduce_level(m: tuple, log_scale: Array, size: int, radix: int):
    """Multiply groups of `radix` neighbouring matrices, renormalising once per group.

    Renormalising once per *group* rather than once per product matters: dividing by
    the largest entry rounds all four entries, and for a near-defective operator (the
    discrete Laplacian being the standard example) the transfer matrices are close to
    rank 1, so that rounding lands in the small entries which carry the answer and is
    then amplified by every later product. Grouping keeps the count of roundings down.
    """
    rem = -size % radix
    if rem:
        a, b, c, d = m
        one = jnp.ones((rem,), a.dtype)
        zero = jnp.zeros((rem,), a.dtype)
        # Identity matrices at the high end leave the product unchanged.
        m = (
            jnp.concatenate([a, one]),
            jnp.concatenate([b, zero]),
            jnp.concatenate([c, zero]),
            jnp.concatenate([d, one]),
        )
        log_scale = jnp.concatenate([log_scale, jnp.zeros((rem,), log_scale.dtype)])
        size += rem
    groups = size // radix
    m = tuple(x.reshape(groups, radix) for x in m)
    log_scale = log_scale.reshape(groups, radix).sum(axis=1)
    acc = tuple(x[:, 0] for x in m)
    for j in range(1, radix):
        # Matrix j applies *after* matrix j - 1, so it multiplies on the left.
        acc = _mul(tuple(x[:, j] for x in m), acc)
    acc, log_scale = _normalise(acc, log_scale)
    return acc, log_scale, groups


def _slogdet_parallel(
    diagonal: Array, lower_diagonal: Array, upper_diagonal: Array
) -> tuple[Array, Array]:
    """Evaluate the transfer-matrix product with a tree. What GPUs want.

    The `n - 1` steps are cut into chunks of `_SLOGDET_BLOCK`; each chunk accumulates
    its own transfer matrix sequentially (one unrolled scan, all chunks in parallel),
    renormalising once at the end, and the chunk matrices are then combined by a
    radix-`_SLOGDET_RADIX` tree. Total kernel launches are `block + log_radix(n/block)`
    rather than `n / block`.
    """
    n = diagonal.shape[0]
    dtype = diagonal.dtype
    real_dtype = jnp.finfo(dtype).dtype
    block = _SLOGDET_BLOCK
    diagonal, coupling, exponent = _prescale(diagonal, lower_diagonal, upper_diagonal)
    diagonal_rest, coupling_rest, n_chunks = _pad_steps(diagonal, coupling, block)
    # (n_chunks, block) -> (block, n_chunks): scan the sequential axis, with the chunks
    # vectorised across it.
    d_steps = diagonal_rest.reshape(n_chunks, block).T
    c_steps = coupling_rest.reshape(n_chunks, block).T

    def unit_step(carry, args):
        # One step maps the chunk's matrix [[a, b], [c, d]] to
        # [[c, d], [d_i c - k_i a, d_i d - k_i b]].
        a, b, c, d = carry
        d_i, k_i = args
        return (c, d, d_i * c - k_i * a, d_i * d - k_i * b), None

    ones = jnp.ones((n_chunks,), dtype)
    zeros = jnp.zeros((n_chunks,), dtype)
    m, _ = lax.scan(
        unit_step, (ones, zeros, zeros, ones), (d_steps, c_steps), unroll=block
    )
    m, log_scale = _normalise(m, jnp.zeros((n_chunks,), real_dtype))

    size = n_chunks
    while size > 1:
        m, log_scale, size = _reduce_level(
            m, log_scale, size, min(_SLOGDET_RADIX, size)
        )

    # Apply the total product to (D_{-1}, D_0) = (1, d_0); the second component is
    # det(A / sigma). `|d_0| <= 1` after the prescale, so this needs no renormalising.
    _, _, c, d = (x[0] for x in m)
    det = c + d * diagonal[0]
    lad = (
        n * exponent.astype(real_dtype) * jnp.log(jnp.array(2.0, real_dtype))
        + log_scale[0]
        + jnp.log(jnp.abs(det))
    )
    return unit_phase(det), lad
