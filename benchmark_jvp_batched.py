#!/usr/bin/env python
"""Benchmark JVP performance for b' only case (vector tangent only)."""

import timeit
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

jax.config.update("jax_enable_x64", True)

# Warm up JIT
key = jr.PRNGKey(0)
_ = jax.jit(lambda x: x + 1)(jnp.ones(10))

def benchmark_jvp_vector_only(solver_name, solver, size, n_steps=10):
    """Benchmark JVP with vector tangent only (b' only case)."""
    key = jr.PRNGKey(42)

    # Create problem based on solver
    if solver_name == "Tridiagonal":
        # Tridiagonal matrix (like linear diffusion)
        diag = 2 * jnp.ones(size)
        off_diag = -1 * jnp.ones(size-1)
        matrix = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        tag = lx.tridiagonal_tag
    elif solver_name == "Cholesky":
        # Positive definite matrix
        A = jr.normal(key, (size, size))
        matrix = A.T @ A + size * jnp.eye(size)
        tag = lx.positive_semidefinite_tag
    else:  # LU, QR, SVD
        matrix = jr.normal(key, (size, size))
        matrix = matrix + size * jnp.eye(size)  # Well-conditioned
        tag = ()

    op = lx.MatrixLinearOperator(matrix, tag)

    def solve_fn(v):
        return lx.linear_solve(op, v, solver).value

    # JIT compile
    jvp_fn = jax.jit(lambda v, tv: jax.jvp(solve_fn, (v,), (tv,)))

    # Simulate n_steps timesteps (like linear diffusion PDE)
    def timestep_sequence(v0, tv0):
        """Simulates n_steps of a linear PDE with JVP."""
        v, tv = v0, tv0
        for _ in range(n_steps):
            v, tv = jvp_fn(v, tv)
        return v, tv

    # Warm up
    vector = jr.normal(key, (size,))
    t_vector = jr.normal(key, (size,))
    _ = timestep_sequence(vector, t_vector)
    _ = timestep_sequence(vector, t_vector)  # Second call for good measure

    # Benchmark
    def run_benchmark():
        return timestep_sequence(vector, t_vector)

    # Time it
    n_runs = 10
    times = []
    for _ in range(n_runs):
        start = timeit.default_timer()
        _ = run_benchmark()
        jax.block_until_ready(_)
        end = timeit.default_timer()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time

def benchmark_jvp_operator_tangent(solver_name, solver, size):
    """Benchmark JVP with operator tangent (non-batched path)."""
    key = jr.PRNGKey(42)

    # Create base matrix
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

    t_matrix = jr.normal(key, (size, size)) * 0.01
    vector = jr.normal(key, (size,))
    t_vector = jr.normal(key, (size,))

    def solve_with_matrix(m, v):
        op = lx.MatrixLinearOperator(m, tag)
        return lx.linear_solve(op, v, solver).value

    # JIT compile
    jvp_fn = jax.jit(lambda m, v, tm, tv: jax.jvp(solve_with_matrix, (m, v), (tm, tv)))

    # Warm up
    _ = jvp_fn(matrix, vector, t_matrix, t_vector)
    _ = jvp_fn(matrix, vector, t_matrix, t_vector)

    # Benchmark
    def run_benchmark():
        return jvp_fn(matrix, vector, t_matrix, t_vector)

    n_runs = 10
    times = []
    for _ in range(n_runs):
        start = timeit.default_timer()
        _ = run_benchmark()
        jax.block_until_ready(_)
        end = timeit.default_timer()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time

# Run benchmarks
print("=" * 80)
print("JVP Performance Benchmark: Batched vs Non-Batched")
print("=" * 80)
print()

solvers = [
    ("Tridiagonal", lx.Tridiagonal()),
    ("LU", lx.LU()),
    ("Cholesky", lx.Cholesky()),
    ("QR", lx.QR()),
]

sizes = [50, 100]
n_steps = 10

for solver_name, solver in solvers:
    print(f"\n{solver_name} Solver")
    print("-" * 40)

    for size in sizes:
        print(f"\n  Problem size: {size}x{size}, {n_steps} timesteps")

        # Benchmark vector-only (batched path)
        try:
            avg_vec, std_vec = benchmark_jvp_vector_only(solver_name, solver, size, n_steps)
            print(f"    Vector tangent only (batched): {avg_vec*1000:.2f} ± {std_vec*1000:.2f} ms")
        except Exception as e:
            print(f"    Vector tangent only (batched): Failed - {e}")
            avg_vec = None

        # Benchmark with operator tangent (non-batched path)
        try:
            avg_op, std_op = benchmark_jvp_operator_tangent(solver_name, solver, size)
            print(f"    With operator tangent (non-batched): {avg_op*1000:.2f} ± {std_op*1000:.2f} ms")
        except Exception as e:
            print(f"    With operator tangent (non-batched): Failed - {e}")
            avg_op = None

        if avg_vec is not None and avg_op is not None:
            speedup = avg_op / avg_vec if avg_vec > 0 else float('inf')
            print(f"    Speedup: {speedup:.2f}x")

print("\n" + "=" * 80)
print("Benchmark complete!")
print("=" * 80)
