#!/usr/bin/env python
"""Compare BEFORE vs AFTER batching optimization in a single run."""

import functools as ft
import os
import timeit

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

# Set EQX_ON_ERROR=nan
os.environ['EQX_ON_ERROR'] = 'nan'

jax.config.update("jax_enable_x64", True)

print("=" * 80)
print("JVP Benchmark Comparison: BEFORE vs AFTER Batching Optimization")
print("=" * 80)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
print(f"EQX_ON_ERROR: {os.environ.get('EQX_ON_ERROR', 'not set')}")
print("=" * 80)
print()

key = jr.PRNGKey(42)

def benchmark_solver(solver_name, solver, size):
    """Benchmark a single solver/size configuration."""

    # Create problem
    if solver_name == "Tridiagonal":
        diag = 2 * jnp.ones(size)
        off_diag = -1 * jnp.ones(size-1)
        matrix = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        tag = lx.tridiagonal_tag
    elif solver_name == "Cholesky":
        A = jr.normal(key, (size, size))
        matrix = A.T @ A + size * jnp.eye(size)
        tag = lx.positive_semidefinite_tag
    else:  # LU
        matrix = jr.normal(key, (size, size))
        matrix = matrix + size * jnp.eye(size)
        tag = ()

    op = lx.MatrixLinearOperator(matrix, tag)
    vector = jr.normal(key, (size,))
    t_vector = jr.normal(key, (size,))

    def solve_fn(v):
        return lx.linear_solve(op, v, solver).value

    # JIT compile
    jvp_fn = jax.jit(lambda v, tv: jax.jvp(solve_fn, (v,), (tv,)))
    bench_fn = ft.partial(jvp_fn, vector, t_vector)

    # Warm up / compile
    _ = bench_fn()

    # Benchmark
    n_runs = 20
    times = []
    for _ in range(n_runs):
        t = timeit.timeit(lambda: jax.block_until_ready(bench_fn()), number=1)
        times.append(t * 1000)  # ms

    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5

    return avg, std

# Configurations
solvers = [
    ("Tridiagonal", lx.Tridiagonal()),
    ("LU", lx.LU()),
    ("Cholesky", lx.Cholesky()),
]

sizes = [50, 100, 200]

print("Testing with BATCHED implementation (current):")
print()

results_after = {}
for solver_name, solver in solvers:
    print(f"{solver_name}:")
    results_after[solver_name] = {}
    for size in sizes:
        try:
            avg, std = benchmark_solver(solver_name, solver, size)
            results_after[solver_name][size] = (avg, std)
            print(f"  Size {size:3d}: {avg:7.4f} ± {std:6.4f} ms")
        except Exception as e:
            print(f"  Size {size:3d}: Failed - {e}")
            results_after[solver_name][size] = None
    print()

print("=" * 80)
print("\nNote: To get BEFORE/AFTER comparison:")
print("  1. Checkout the commit before this optimization")
print("  2. Run this benchmark (results will be BEFORE)")
print("  3. Compare with these results (AFTER)")
print("\nCurrent implementation uses batched vmap for vector-tangent-only case.")
print("=" * 80)
