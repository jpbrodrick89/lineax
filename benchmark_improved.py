#!/usr/bin/env python
"""Improved benchmark following lineax's solver_speeds.py pattern."""

import functools as ft
import os
import sys
import timeit

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

# Set EQX_ON_ERROR=nan as requested
os.environ['EQX_ON_ERROR'] = 'nan'

jax.config.update("jax_enable_x64", True)

print("=" * 80)
print("JVP Benchmark: Vector Tangent Only (Batched Optimization)")
print("=" * 80)
print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print(f"EQX_ON_ERROR: {os.environ.get('EQX_ON_ERROR', 'not set')}")
print("=" * 80)
print()

key = jr.PRNGKey(42)

def benchmark_jvp_vector_only(solver_name, solver, size):
    """Benchmark JVP with vector tangent only, following lineax pattern."""

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

    # JIT compile and wrap JVP (following lineax pattern)
    jvp_fn = jax.jit(lambda v, tv: jax.jvp(solve_fn, (v,), (tv,)))
    bench_fn = ft.partial(jvp_fn, vector, t_vector)

    # Compile by calling once (following lineax pattern at line 189)
    result = bench_fn()

    # Verify result is ready
    jax.block_until_ready(result)

    # Time using timeit.timeit with number=1 (following lineax pattern at line 191)
    # Do this multiple times to get statistics
    n_runs = 10
    times = []
    for _ in range(n_runs):
        t = timeit.timeit(lambda: jax.block_until_ready(bench_fn()), number=1)
        times.append(t * 1000)  # Convert to ms

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    return avg_time, min_time, max_time

# Test configurations
solvers = [
    ("Tridiagonal", lx.Tridiagonal()),
    ("LU", lx.LU()),
    ("Cholesky", lx.Cholesky()),
]

sizes = [10, 50, 100, 200, 500]

print("Running benchmarks (10 runs per configuration)...")
print()

for solver_name, solver in solvers:
    print(f"{solver_name} Solver:")
    print("-" * 60)
    for size in sizes:
        try:
            avg, min_t, max_t = benchmark_jvp_vector_only(solver_name, solver, size)
            print(f"  Size {size:3d}: {avg:7.4f} ms  (min: {min_t:7.4f}, max: {max_t:7.4f})")
        except Exception as e:
            print(f"  Size {size:3d}: Failed - {e}")
    print()

print("=" * 80)
print("Benchmark complete!")
print("=" * 80)
