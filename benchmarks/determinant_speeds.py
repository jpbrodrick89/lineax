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

"""Re-derives the measurements quoted in the determinant code's comments.

`lineax._solver.tridiagonal` chooses between two implementations with
`lax.platform_dependent`, and pins two constants (`_SLOGDET_BLOCK`, `_SLOGDET_RADIX`)
on measured grounds. `lineax._determinant` routes some operators around the generic
JVP on measured grounds too. Those numbers will drift as JAX and XLA move, so this
script recomputes them rather than leaving the comments to rot.

Run it once per platform; sections that only make sense on one skip themselves:

    python benchmarks/determinant_speeds.py
    python benchmarks/determinant_speeds.py --platform cpu

`--quick` shrinks every sweep, for a smoke test rather than a measurement.
"""

import argparse
import os
import time


os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lineax as lx  # noqa: E402
import numpy as np  # noqa: E402
from lineax._solver import tridiagonal as _tri  # noqa: E402


def time_us(fn, args, reps, trials=3):
    """Best-of-`trials` mean over `reps` calls, dispatched asynchronously.

    On GPU the floor is Python dispatch (~50us here), so treat small absolute numbers
    as noise and read the ratios instead.
    """
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    best = float("inf")
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(reps):
            out = jitted(*args)
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - start) / reps)
    return best * 1e6


def time_us_device(fn, args, reps=(2, 6, 14), trials=3):
    """Marginal device time per call, from the slope of an in-jit repeat loop.

    Wall-clock bottoms out at the Python dispatch floor (~75us here), which is larger
    than the differences that separate `_SLOGDET_BLOCK` and `_SLOGDET_RADIX` settings.
    Timing `r` dependent repetitions inside one jit and taking d(time)/d(r) cancels
    both the dispatch cost and the loop's own overhead.
    """

    def repeat(r):
        def wrapped(*operands):
            def body(_, carry):
                arrays, acc = carry
                _, lad = fn(*arrays)
                # A real but negligible dependency, so the repetitions cannot be
                # collapsed into one.
                nudged = (arrays[0] + lad * jnp.asarray(1e-30, arrays[0].dtype),)
                return (nudged + arrays[1:], acc + lad)

            _, acc = jax.lax.fori_loop(
                0, r, body, (operands, jnp.zeros((), operands[0].dtype))
            )
            return acc

        return wrapped

    times = [time_us(repeat(r), args, 5, trials) for r in reps]
    x = np.asarray(reps, dtype=float)
    y = np.asarray(times)
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def tridiagonal_arrays(n, seed=0, dtype=np.float64):
    """A diagonally dominant tridiagonal operator, as three diagonals."""
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray((3.0 + rng.uniform(size=n)).astype(dtype)),
        jnp.asarray((rng.uniform(-1.0, 1.0, n - 1) * 0.3).astype(dtype)),
        jnp.asarray((rng.uniform(-1.0, 1.0, n - 1) * 0.3).astype(dtype)),
    )


def structured_operator(kind, n, seed=0):
    """(operator, tangent operator, solver) for each structure with a fast JVP."""
    rng = np.random.default_rng(seed)
    if kind == "diagonal":
        make = lx.DiagonalLinearOperator
        args = lambda r: (jnp.asarray(3.0 + r.uniform(size=n)),)  # noqa: E731
        solver = lx.Diagonal(well_posed=True)
    elif kind == "tridiagonal":
        make = lx.TridiagonalLinearOperator
        args = lambda r: (  # noqa: E731
            jnp.asarray(3.0 + r.uniform(size=n)),
            jnp.asarray(r.uniform(-1, 1, n - 1) * 0.3),
            jnp.asarray(r.uniform(-1, 1, n - 1) * 0.3),
        )
        solver = lx.Tridiagonal()
    elif kind == "circulant":

        def args(r):
            column = r.uniform(size=n)
            column[0] += n
            return (jnp.asarray(column),)

        make = lx.CirculantLinearOperator
        solver = lx.Circulant(well_posed=True)
    elif kind == "triangular":

        def make(matrix):
            return lx.MatrixLinearOperator(matrix, lx.lower_triangular_tag)

        def args(r):
            matrix = np.tril(r.uniform(-1, 1, (n, n)))
            np.fill_diagonal(matrix, 3.0 + r.uniform(size=n))
            return (jnp.asarray(matrix),)

        solver = lx.Triangular()
    else:
        raise ValueError(kind)
    return make(*args(rng)), make(*args(rng)), solver


def section_platform_dispatch(sizes):
    """Why `Tridiagonal.slogdet` dispatches on platform at all."""
    print("\n=== primal: sequential scan vs 2x2-transfer-matrix tree (float64) ===")
    print("The scan costs a kernel launch per block, so it is hopeless on GPU; the")
    print("tree does more arithmetic, so it loses on CPU. Hence the dispatch.")
    print(f"{'n':>9} {'sequential':>12} {'tree':>10} {'seq / tree':>12}")
    for n in sizes:
        arrays = tridiagonal_arrays(n)
        reps = 20 if n <= 8192 else 5
        seq = time_us(_tri._slogdet_sequential, arrays, reps)
        par = time_us(_tri._slogdet_parallel, arrays, reps)
        print(f"{n:>9} {seq:>12.1f} {par:>10.1f} {seq / par:>12.1f}")


def section_block(sizes, on_gpu):
    """`_SLOGDET_BLOCK`: renormalisation interval, and chunk length on GPU."""
    which = "tree" if on_gpu else "scan"
    impl = _tri._slogdet_parallel if on_gpu else _tri._slogdet_sequential
    print(f"\n=== `_SLOGDET_BLOCK` sweep, timing the {which} (float64, us) ===")
    print("Governs both implementations, so it is swept on whichever runs here.")
    print("Device time, not wall clock: the settings differ by less than dispatch.")
    header = "".join(f"{b:>9}" for b in (2, 4, 8, 16, 32))
    print(f"{'n':>9}{header}")
    original = _tri._SLOGDET_BLOCK
    try:
        for n in sizes:
            arrays = tridiagonal_arrays(n)
            row = []
            for block in (2, 4, 8, 16, 32):
                _tri._SLOGDET_BLOCK = block
                # A fresh lambda per setting: `jax.jit` caches on function identity, so
                # reusing the module function would silently reuse the first trace.
                fn = lambda *a, impl=impl: impl(*a)  # noqa: E731
                row.append(time_us_device(fn, arrays))
            print(f"{n:>9}" + "".join(f"{v:>9.1f}" for v in row))
    finally:
        _tri._SLOGDET_BLOCK = original


def section_radix(sizes):
    """`_SLOGDET_RADIX`: tree fan-in. Only `_slogdet_parallel` has a tree."""
    print("\n=== `_SLOGDET_RADIX` sweep, timing the tree (float64, us) ===")
    print("Device time, not wall clock, as above.")
    header = "".join(f"{r:>9}" for r in (2, 4, 8, 16))
    print(f"{'n':>9}{header}")
    original = _tri._SLOGDET_RADIX
    try:
        for n in sizes:
            arrays = tridiagonal_arrays(n)
            row = []
            for radix in (2, 4, 8, 16):
                _tri._SLOGDET_RADIX = radix
                fn = lambda *a: _tri._slogdet_parallel(*a)  # noqa: E731
                row.append(time_us_device(fn, arrays))
            print(f"{n:>9}" + "".join(f"{v:>9.1f}" for v in row))
    finally:
        _tri._SLOGDET_RADIX = original


def section_grading(spans, n=128, seeds=4):
    """The tolerated entry grading, which is what really pins `_SLOGDET_BLOCK`.

    Pure numerics, so the answer barely depends on the platform. An operator whose
    entries span `10**-span` decays past float range within a block once
    `span * block > ~308`, and the determinant underflows (or, worse, comes back
    silently wrong).
    """
    print("\n=== largest tolerated grading, by `_SLOGDET_BLOCK` (float64) ===")
    print("Entries spanning 10**-span; 'ok' means both implementations agree with a")
    print("dense reference. The arithmetic is the same on either platform, though")
    print("reduction order can move the boundary by a step of the sweep.")
    print(f"{'block':>7} {'heuristic':>10} {'largest ok span':>17}")
    original = _tri._SLOGDET_BLOCK
    try:
        for block in (2, 4, 8, 16, 32):
            _tri._SLOGDET_BLOCK = block
            last_ok = 0
            for span in spans:
                ok = True
                for seed in range(seeds):
                    rng = np.random.default_rng(seed)
                    grade = 10.0 ** (-span * np.arange(n) / n)
                    diag = (rng.normal(size=n) + 4.0) * grade
                    off = rng.normal(size=(2, n - 1)) * 0.5 * grade[None, :-1]
                    matrix = np.diag(diag) + np.diag(off[0], -1) + np.diag(off[1], 1)
                    _, ref = np.linalg.slogdet(matrix)
                    args = (jnp.asarray(diag), jnp.asarray(off[0]), jnp.asarray(off[1]))
                    for impl in (_tri._slogdet_sequential, _tri._slogdet_parallel):
                        fn = lambda *a, impl=impl: impl(*a)  # noqa: E731
                        lad = float(jax.jit(fn)(*args)[1])
                        if not np.isfinite(lad) or abs(lad - ref) > 1e-9 * abs(ref):
                            ok = False
                if not ok:
                    break
                last_ok = span
            print(f"{block:>7} {308 / block:>10.0f} {last_ok:>17}")
    finally:
        _tri._SLOGDET_BLOCK = original


def section_gradient(sizes):
    """Reverse mode through each implementation, on whichever platform this is.

    The JVP rule differentiates whatever `Tridiagonal.slogdet` dispatched to, so the
    relevant question is which implementation is cheaper to reverse *here*.
    """
    print(
        "\n=== gradient: primal, total, and the backward pass alone (float64, us) ==="
    )
    print(
        f"{'impl':>6} {'n':>9} {'primal':>10} {'grad':>10} {'grad - primal':>15}"
        f" {'marginal':>10}"
    )
    for name, impl in (
        ("scan", _tri._slogdet_sequential),
        ("tree", _tri._slogdet_parallel),
    ):
        for n in sizes:
            arrays = tridiagonal_arrays(n)
            reps = 20 if n <= 8192 else 5
            primal = time_us(impl, arrays, reps)
            grad_fn = jax.grad(lambda *a, impl=impl: impl(*a)[1], argnums=(0, 1, 2))
            grad = time_us(grad_fn, arrays, max(reps // 2, 3))
            print(
                f"{name:>6} {n:>9} {primal:>10.1f} {grad:>10.1f} "
                f"{grad - primal:>15.1f} {(grad - primal) / primal:>10.1f}"
            )


def section_jvp_dispatch(sizes):
    """Structured operators skip the generic one-solve-per-column JVP."""
    print("\n=== `lx.slogdet` gradient: structured fast path vs the generic rule ===")
    print(
        f"{'operator':>13} {'n':>7} {'fast path':>11} {'per column':>12} {'ratio':>7}"
    )
    for kind in ("diagonal", "triangular", "tridiagonal", "circulant"):
        for n in sizes:
            operator, tangent, solver = structured_operator(kind, n)
            state = solver.init(operator, {})

            def fast(o, solver=solver):
                return jax.grad(lambda x: lx.slogdet(x, solver)[1])(o)

            def generic(o, tangent=tangent, solver=solver, state=state):
                # What the rule does for an operator it has no fast path for.
                dense = lx.TangentLinearOperator(o, tangent).as_matrix()
                solve = lambda col: (
                    lx.linear_solve(  # noqa: E731
                        o, col, solver, state=state
                    ).value
                )
                return jnp.trace(jax.vmap(solve)(dense.T))

            a = time_us(fast, (operator,), 10)
            b = time_us(generic, (operator,), 3)
            print(f"{kind:>13} {n:>7} {a:>11.1f} {b:>12.1f} {b / a:>7.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default=None, help="e.g. cpu")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.platform is not None:
        jax.config.update("jax_platform_name", args.platform)
    on_gpu = jax.default_backend() == "gpu"
    print(
        f"jax {jax.__version__}, backend {jax.default_backend()}, "
        f"x64={jax.config.jax_enable_x64}"
    )

    if args.quick:
        sizes, jvp_sizes, tuning_sizes = (512, 8192), (1024,), (131072,)
        spans = range(4, 80, 8)
    else:
        sizes, jvp_sizes = (512, 8192, 131072), (1024, 8192)
        # The constants only become visible once the work clears the dispatch floor.
        tuning_sizes = (131072, 1048576, 4194304)
        spans = range(4, 120, 2)

    section_platform_dispatch(sizes)
    section_block(tuning_sizes, on_gpu)
    if on_gpu:
        section_radix(tuning_sizes)
    else:
        print("\n=== `_SLOGDET_RADIX` sweep: skipped ===")
        print("Only `_slogdet_parallel` has a tree, and CPU never runs it -- the")
        print("platform dispatch and the JVP both use the scan there.")
    section_grading(spans)
    section_gradient(sizes)
    section_jvp_dispatch(jvp_sizes)


if __name__ == "__main__":
    main()
