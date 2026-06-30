# 3D Function Sampling Algorithm

This project implements a high-performance, parallel algorithm to sample signed random values from a 3D function $f(x, y, z)$ in a finite Cartesian volume.

## Architecture

- **Adaptive Octree**: Tabulates the absolute maximum $|f|_{max}$ and mean $|f|_{mean}$ in sub-volumes.
  - **Refinement Criteria**: Uses mass-based (Integrated $|f|_{max}$ with a default threshold of $0.01$) and efficiency-based ($\frac{|f|_{mean}}{|f|_{max}} > 0.5$) termination.
  - **Stencil**: Uses a 5x5x5 grid with precomputed coordinates for robust, high-performance local peak estimation.
  - **Deep Refinement**: Supports depths up to 30 to resolve highly localized features ($10^{-9}$ scale).
  - **Parallel Building**: Employs OpenMP-based recursive subdivision at the root level (`depth == 0`), distributing subtree builds across threads for up to a 5x construction speedup.
- **Walker's Alias Method**: Enables $O(1)$ sampling of the octree boxes based on their integrated $|f|_{max}$.
  - **Fast Range Reduction**: Employs unbiased 128-bit multiplication range reduction (`(uint64_t * range) >> 64`) to avoid expensive integer modulo divisions in the hot path.
  - **Allocation Safety**: Avoids dynamic container chunk allocations by using contiguous `std::vector` backings instead of standard `std::stack` structures.
  - **Zero-Sum Safety**: Gracefully handles zero-probability distributions by defaulting to a uniform sampling scheme.
- **Parallel Rejection Sampling**:
  - **OpenMP**: Utilizes multi-threading for the sampling loop.
  - **PRNG Isolation**: Uses `PCG64_OpenMP_Manager` to manage independent, thread-safe streams.
  - **Increment Protocol**: The manager initializes each thread stream with a fixed seed (1) and a unique odd increment `2*tid+1` to ensure independent, non-correlated random streams, supporting checkpointing and thread-safe dynamic resize.
  - **Warm-up**: Every generator instance is warmed up (`gen.warmup()`) inside the manager's initialization and resizing blocks.

## Project Status

- [x] Project structure initialization
- [x] Octree implementation (adaptive refinement with grid estimation)
- [x] Walker's Alias Method implementation (Unified Stream interface)
- [x] Parallel Sampling integration (OpenMP + Thread-local PRNG)
- [x] Validation (100M samples, Z-score analysis)
- [x] Performance Optimization (`-Ofast -march=native`)
- [x] Extended Unit Test suite (Alias, PCG64, Octree, Sampler3D)
- [x] Refinement Trigger Instrumentation (Mass vs. Flatness stats)
- [x] Default settings tuned for maximum compactness and efficiency (Cutoff=0.01, Flatness=0.5)

## Key Findings (Validation with 100M Samples)

- **Bias-free**: Z-scores remain low ($< 1.0$) across both sines and sharp Gaussian studies, validating statistical correctness.
- **Scaling**: Parallel sampling completes $10^8$ samples in $\sim 2.9$s on 8 threads.
- **Compactness**: Adaptive refinement using the default tuned parameters yields a 9x reduction in tree size for the Gaussian function (152k nodes vs 1.37M nodes).
- **Efficiency**: Global sampling efficiency is $\sim 37.7\%$ for smooth sine function and $\sim 60.8\%$ for sharp Gaussian.

## File Structure

- `include/`: Header files (`octree.h`, `alias_method.h`, `sampler.h`, `pcg64.h`).
- `src/`: Implementation and demo (`main.cpp`, `octree.cpp`, etc.).
- `tests/`: Correctness unit tests (`test_alias.cpp`, `test_pcg64.cpp`, `test_octree.cpp`, `test_sampler.cpp`).
- `CMakeLists.txt`: Build configuration with OpenMP support.
