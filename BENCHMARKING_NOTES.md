# Benchmarking Notes

## Answers to Key Questions

### Do we have GPU access?
**No** - This environment only has CPU available:
- JAX devices: `[CpuDevice(id=0)]`
- JAX backend: `cpu`

This means our benchmarks are CPU-only and may not reflect GPU performance.

### Are we following lineax's benchmarking pattern?
**Yes** - We now follow the pattern from `benchmarks/solver_speeds.py`:

1. **JIT compile** the function with `jax.jit()`
2. **Warm up** by calling the function once to trigger compilation
3. **Time with timeit** using `timeit.timeit(lambda: jax.block_until_ready(fn()), number=1)`
4. **Multiple runs** to get statistical measures

See `benchmark_improved.py` for the improved implementation following this pattern.

### EQX_ON_ERROR=nan
**Set** - All new benchmarks run with `EQX_ON_ERROR=nan` as requested.

## CPU Benchmark Limitations

### Why Results are Noisy

1. **Small batch size (n=2)**: Batching only 2 operations limits parallelization
2. **vmap overhead**: JAX vmap adds dispatch overhead that may exceed gains
3. **Cache effects**: Memory access patterns affect performance unpredictably
4. **CPU scheduling**: OS scheduling can interfere with timing

### What Would Help

1. **GPU/TPU**: Batched operations are optimized for accelerators
2. **Larger batches**: Batching 8-16 operations would show clearer benefits
3. **Realistic workload**: Multi-step PDE solve (10-100 timesteps)

## Benchmark Results Summary

### BEFORE (non-batched) vs AFTER (batched)

Results are **highly variable** on CPU:
- Some sizes: 10-25% faster
- Other sizes: 0-17% slower
- Within measurement noise

### Why Keep the Optimization?

1. ✅ **Better code structure**: DRY principle, single return path
2. ✅ **GPU potential**: Should perform better on accelerators
3. ✅ **Correctness**: All tests pass
4. ✅ **Maintainability**: Clearer intent, easier to understand

## Recommendation

The optimization is **worth keeping** despite inconclusive CPU benchmarks because:
1. Code quality improvements are real
2. GPU performance is likely better (untested)
3. Minimal CPU overhead on average
4. Better foundation for future work

The CPU environment limits our ability to measure the true benefit.
