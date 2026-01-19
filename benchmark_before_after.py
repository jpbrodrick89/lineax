#!/usr/bin/env python
"""Benchmark to compare vector-only JVP: BEFORE vs AFTER batching optimization.

This measures the SAME operation (vector tangent only) with two implementations.
"""

import timeit
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

jax.config.update("jax_enable_x64", True)

key = jr.PRNGKey(42)

def single_jvp_benchmark(solver_name, solver, size):
    """Benchmark a single JVP operation with vector tangent only."""

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
    else:
        matrix = jr.normal(key, (size, size))
        matrix = matrix + size * jnp.eye(size)
        tag = ()

    op = lx.MatrixLinearOperator(matrix, tag)
    vector = jr.normal(key, (size,))
    t_vector = jr.normal(key, (size,))

    def solve_fn(v):
        return lx.linear_solve(op, v, solver).value

    # JIT compile the JVP
    jvp_fn = jax.jit(lambda v, tv: jax.jvp(solve_fn, (v,), (tv,)))

    # Warm up
    _ = jvp_fn(vector, t_vector)
    _ = jvp_fn(vector, t_vector)

    # Benchmark
    n_runs = 100
    times = []
    for _ in range(n_runs):
        start = timeit.default_timer()
        result = jvp_fn(vector, t_vector)
        jax.block_until_ready(result)
        end = timeit.default_timer()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    return avg_time * 1000, std_time * 1000  # Convert to ms

print("=" * 80)
print("Single JVP Benchmark: Vector Tangent Only (WITH batching optimization)")
print("=" * 80)
print()

solvers = [
    ("Tridiagonal", lx.Tridiagonal()),
    ("LU", lx.LU()),
    ("Cholesky", lx.Cholesky()),
]

sizes = [10, 50, 100, 200]

for solver_name, solver in solvers:
    print(f"\n{solver_name} Solver:")
    print("-" * 40)
    for size in sizes:
        try:
            avg, std = single_jvp_benchmark(solver_name, solver, size)
            print(f"  Size {size:3d}: {avg:.4f} ± {std:.4f} ms")
        except Exception as e:
            print(f"  Size {size:3d}: Failed - {e}")

print("\n" + "=" * 80)
print("\nNote: To compare against the old implementation, you would need to:")
print("  1. Revert the batching changes in _solve.py")
print("  2. Run this same benchmark")
print("  3. Compare the results")
print("\nThe batched version should be faster due to:")
print("  - Reduced overhead from fewer filter_primitive_bind calls")
print("  - Better utilization of batched BLAS operations (especially for direct solvers)")
print("=" * 80)
