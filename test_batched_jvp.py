#!/usr/bin/env python
"""Quick test to verify the batched JVP optimization works correctly."""

import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

# Test case 1: Simple matrix solve with vector tangent only
print("Test 1: Vector tangent only (should use batched path)")
key = jr.PRNGKey(0)
matrix = jr.normal(key, (5, 5))
matrix = matrix + 5 * jnp.eye(5)  # Make well-conditioned
vector = jr.normal(key, (5,))
t_vector = jr.normal(key, (5,))

op = lx.MatrixLinearOperator(matrix)

def solve_fn(v):
    return lx.linear_solve(op, v, lx.LU()).value

# Compute JVP (vector tangent only)
sol, t_sol = jax.jvp(solve_fn, (vector,), (t_vector,))

# Check against expected: x = A^-1 b, x' = A^-1 b'
expected_sol = jnp.linalg.solve(matrix, vector)
expected_t_sol = jnp.linalg.solve(matrix, t_vector)

print(f"Primal solution matches: {jnp.allclose(sol, expected_sol)}")
print(f"Tangent solution matches: {jnp.allclose(t_sol, expected_t_sol)}")
assert jnp.allclose(sol, expected_sol, rtol=1e-5)
assert jnp.allclose(t_sol, expected_t_sol, rtol=1e-5)
print("✓ Test 1 passed\n")

# Test case 2: Tridiagonal solve (linear diffusion case)
print("Test 2: Tridiagonal solve (linear diffusion)")
n = 10
# Create tridiagonal matrix
diag = 2 * jnp.ones(n)
off_diag = -1 * jnp.ones(n-1)
tri_matrix = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

vector = jr.normal(key, (n,))
t_vector = jr.normal(key, (n,))

op = lx.MatrixLinearOperator(tri_matrix, lx.tridiagonal_tag)

def solve_tri(v):
    return lx.linear_solve(op, v, lx.Tridiagonal()).value

sol, t_sol = jax.jvp(solve_tri, (vector,), (t_vector,))

expected_sol = jnp.linalg.solve(tri_matrix, vector)
expected_t_sol = jnp.linalg.solve(tri_matrix, t_vector)

print(f"Primal solution matches: {jnp.allclose(sol, expected_sol)}")
print(f"Tangent solution matches: {jnp.allclose(t_sol, expected_t_sol)}")
assert jnp.allclose(sol, expected_sol, rtol=1e-5)
assert jnp.allclose(t_sol, expected_t_sol, rtol=1e-5)
print("✓ Test 2 passed\n")

# Test case 3: Operator tangent (should use old path)
print("Test 3: Operator tangent (should use non-batched path)")

def solve_with_matrix(m, v):
    op = lx.MatrixLinearOperator(m)
    return lx.linear_solve(op, v, lx.LU()).value

matrix = jr.normal(key, (5, 5))
matrix = matrix + 5 * jnp.eye(5)
t_matrix = jr.normal(key, (5, 5)) * 0.01
vector = jr.normal(key, (5,))
t_vector = jr.normal(key, (5,))

sol, t_sol = jax.jvp(solve_with_matrix, (matrix, vector), (t_matrix, t_vector))

# Check via finite differences
eps = 1e-5
sol_perturbed = jnp.linalg.solve(matrix + eps * t_matrix, vector + eps * t_vector)
expected_t_sol_approx = (sol_perturbed - sol) / eps

print(f"Primal solution correct: {jnp.allclose(sol, jnp.linalg.solve(matrix, vector))}")
print(f"Tangent approximately correct: {jnp.allclose(t_sol, expected_t_sol_approx, rtol=1e-3, atol=1e-3)}")
assert jnp.allclose(sol, jnp.linalg.solve(matrix, vector), rtol=1e-5)
# Looser tolerance for finite difference approximation
assert jnp.allclose(t_sol, expected_t_sol_approx, rtol=0.1, atol=0.1)
print("✓ Test 3 passed\n")

print("=" * 60)
print("All tests passed! Batched JVP optimization is working correctly.")
print("=" * 60)
