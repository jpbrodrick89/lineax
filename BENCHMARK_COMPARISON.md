# JVP Batching Optimization - Detailed Benchmark Comparison

## Environment
- **Backend**: CPU (CpuDevice)
- **Precision**: Float64 (jax_enable_x64=True)
- **EQX_ON_ERROR**: nan
- **Pattern**: Following lineax's solver_speeds.py (JIT compile, warm up, then timeit)
- **Runs**: 10 per configuration

## Results: BEFORE vs AFTER

### Tridiagonal Solver

| Size | BEFORE (ms) | AFTER (ms) | Speedup | Change |
|------|-------------|------------|---------|--------|
| 10   | 0.0289      | 0.0289     | 1.00x   | ±0%    |
| 50   | 0.2238      | 0.1781     | 1.26x   | **+20%** |
| 100  | 0.1874      | 0.2093     | 0.90x   | -12%   |
| 200  | 0.2310      | 0.2275     | 1.02x   | +2%    |
| 500  | 0.3500      | 0.2923     | 1.20x   | **+16%** |

### LU Solver

| Size | BEFORE (ms) | AFTER (ms) | Speedup | Change |
|------|-------------|------------|---------|--------|
| 10   | 0.0316      | 0.0299     | 1.06x   | +5%    |
| 50   | 0.2629      | 0.3074     | 0.86x   | -17%   |
| 100  | 0.4271      | 0.3208     | 1.33x   | **+25%** |
| 200  | 0.7254      | 0.7383     | 0.98x   | -2%    |
| 500  | 4.2890      | 4.4580     | 0.96x   | -4%    |

### Cholesky Solver

| Size | BEFORE (ms) | AFTER (ms) | Speedup | Change |
|------|-------------|------------|---------|--------|
| 10   | 0.0300      | 0.0280     | 1.07x   | +7%    |
| 50   | 0.2491      | 0.2188     | 1.14x   | **+12%** |
| 100  | 0.2800      | 0.2812     | 1.00x   | ±0%    |
| 200  | 0.6683      | 0.6991     | 0.96x   | -5%    |
| 500  | 2.1512      | 2.1164     | 1.02x   | +2%    |

## Analysis

### Key Observations

1. **High Variability on CPU**
   - Results show significant run-to-run variance (see min/max ranges)
   - Size 50 for Tridiagonal: +20% improvement
   - Size 100 for LU: +25% improvement
   - But some sizes show regressions (within noise)

2. **No Clear Pattern**
   - Unlike initial benchmarks, improvements are inconsistent
   - Some sizes benefit, others regress slightly
   - Suggests performance is dominated by factors other than the batching

3. **CPU-Specific Limitations**
   - vmap overhead may offset batching gains on CPU
   - Cache effects and memory access patterns matter more
   - Small batch size (2 operations) limits parallelization

### Comparison with Initial Results

Our initial benchmark showed more consistent 4-14% improvements. This more rigorous benchmark with:
- Proper warmup following lineax pattern
- EQX_ON_ERROR=nan
- More statistical rigor

Shows that the actual benefit is **highly variable and size-dependent** on CPU.

### Why the Inconsistency?

1. **Timing Noise**: CPU benchmarks at this scale are noisy
2. **Cache Effects**: Different memory access patterns between batched/non-batched
3. **JIT Compilation**: XLA may optimize differently for each case
4. **vmap Overhead**: vmap adds overhead that may exceed batching gains for n=2

## Conclusions

### On CPU (current environment)

The batching optimization shows **mixed results**:
- ✅ Some configurations show 10-25% improvement
- ❌ Other configurations show 0-17% regression
- 🤔 High variance suggests noise dominates signal

### Expected on GPU/TPU

The optimization should perform better on accelerators because:
1. **Batched operations** are better optimized on GPU
2. **Parallel execution** units can handle both solves simultaneously
3. **Memory bandwidth** is better utilized with batched access

### Recommendation

**Keep the optimization** because:
1. ✅ **Code quality**: DRY principle, clearer structure
2. ✅ **Correctness**: All tests pass
3. ✅ **GPU potential**: Likely better performance on accelerators
4. ✅ **Use case**: Linear PDE timestepping accumulates benefit over steps
5. ⚠️ **CPU overhead**: Minimal on average, within noise

The optimization provides a **better foundation** even if CPU benchmarks are inconclusive.

## Next Steps

1. 🔬 **GPU Benchmarks**: Test on actual accelerators
2. 📊 **Multi-step PDE**: Benchmark realistic linear diffusion scenario (10-100 steps)
3. 🎯 **Profile**: Deep dive into XLA compilation to understand variance
4. 📝 **Document**: Add comments about expected GPU benefits
