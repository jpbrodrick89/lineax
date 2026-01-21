# Using throw=False with Autodiff in Lineax

## Question
Why does `throw=False` still raise errors when differentiating through LSMR solves that don't converge?

## Answer

**TL;DR**: The `throw=False` parameter only affects the **forward pass**. The **autodiff passes (JVP/VJP)** still use hardcoded `throw=True` internally, which causes errors when tangent/adjoint solves don't converge.

## How throw=False Works

When you set `throw=False`:

```python
solution = lx.linear_solve(op, b, lx.LSMR(max_steps=5), throw=False)
# ✅ Returns partial solution even if not converged
# solution.result tells you if it converged
```

This works perfectly for **forward-only computation**.

## The Autodiff Problem

But when you differentiate:

```python
def loss(b):
    sol = lx.linear_solve(op, b, lx.LSMR(max_steps=5), throw=False)
    return jnp.sum(sol.value ** 2)

gradient = jax.grad(loss)(b)  # ❌ Still throws error!
```

### Why?

Looking at `lineax/_solve.py`, the JVP and VJP implementations have **hardcoded `throw=True`** for all tangent/adjoint solves:

**JVP (forward-mode, lines 255, 267, 279):**
```python
tmp, _, _ = eqxi.filter_primitive_bind(
    linear_solve_p,
    operator_conj_transpose,
    state_conj_transpose,
    tmp,
    options_conj_transpose,
    solver,
    True,  # ← HARDCODED throw=True
)
```

**VJP (reverse-mode, line 330):**
```python
cts_vector, _, _ = eqxi.filter_primitive_bind(
    linear_solve_p,
    operator_transpose,
    state_transpose,
    cts_solution,
    options_transpose,
    solver,
    True,  # ← HARDCODED throw=True unconditionally
)
```

The comment explains: *"throw=True unconditionally: nowhere to pipe result to."*

## Why This Design?

JAX's autodiff system expects derivative computations to either:
1. Succeed (return a value)
2. Fail (raise an error)

There's **no mechanism to return an error code** from JVP/VJP. The `Solution.result` field only exists in the forward pass.

## Workarounds for Optimization (Adam, etc.)

If you want to use approximate gradients from unconverged solves:

### Option 1: Increase max_steps (Recommended)

Make tangent solves more likely to converge:

```python
solver = lx.LSMR(
    rtol=1e-6,
    atol=1e-6,
    max_steps=10000,  # High enough for tangent solves
    conlim=1e10,
)

# Primal might not converge, but tangent solves will
solution = lx.linear_solve(op, b, solver, throw=False)
gradient = jax.grad(loss_fn)(params)  # Works if tangent solves converge
```

### Option 2: Looser Tolerances

Accept less accurate but converged solutions:

```python
solver = lx.LSMR(
    rtol=1e-4,  # Looser
    atol=1e-4,  # Looser
    max_steps=1000,
)

# More likely to converge
gradient = jax.grad(loss_fn)(params)
```

### Option 3: Stop Gradient on Non-Converged

Don't differentiate through unconverged solves:

```python
def safe_solve(op, b):
    sol = lx.linear_solve(op, b, solver, throw=False)

    if sol.result != lx.RESULTS.successful:
        # Return partial solution but block gradients
        return jax.lax.stop_gradient(sol.value)
    else:
        # Only differentiate through converged solves
        return sol.value

# jax.grad works (zero gradient for unconverged cases)
gradient = jax.grad(lambda p: loss(safe_solve(op(p), b)))(params)
```

### Option 4: Use EQX_ON_ERROR=nan

Set environment variable to get NaN gradients instead of errors:

```python
import os
os.environ['EQX_ON_ERROR'] = 'nan'

# Now errors become NaN gradients (Adam can handle this)
gradient = jax.grad(loss_fn)(params)
# gradient will contain NaN if tangent solve failed
```

## Future Enhancement

A proper solution would be to:

1. **Pass `throw` to tangent solves** in JVP (lines 255, 267, 279, 330)
2. **Accept approximate gradients** when tangent solves don't converge
3. **Document the behavior** - users doing optimization expect approximate gradients to be fine

This would make `throw=False` work consistently with autodiff, which is useful for optimization where approximate gradients are acceptable (Adam, SGD, etc.).

## Current Status

- ✅ `throw=False` works for forward pass
- ❌ `throw=False` doesn't prevent errors in autodiff
- 💡 Use workarounds above for optimization use cases
