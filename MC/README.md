# Sampler3D

A high-performance double-precision C++ library for parallel adaptive importance sampling of 3D functions.

## Prerequisites

- C++17 compatible compiler (GCC 8+, Clang 7+)
- CMake 3.10+
- OpenMP
- Make or Ninja build system

## Building

The project uses CMake and includes high-performance optimization flags (`-Ofast -march=native`) and OpenMP support by default.

```bash
mkdir -p build && cd build
cmake ../MC
make
```

## Running the Demo

The `sampler_demo` executable performs a parallel study using 100,000,000 (100M) samples across detected OpenMP threads, verifying accuracy against analytical moments.

```bash
cd build
./sampler_demo
```

## Running the Tests

A comprehensive unit test suite validates correctness across all core modules. To run the tests, compile the project and execute the test binaries:

```bash
cd build
./test_alias     # Validates the Walker's Alias table frequencies
./test_pcg64     # Validates the PRNG determinism, jumps, and constraints
./test_octree    # Validates Box properties, refinement, and volume conservation
./test_sampler   # Validates end-to-end integration and sign determinations
```

## Parallel Architecture

The sampler uses a `static thread_local` instance of the PCG64 random number generator. To ensure independent streams, the constructor takes an increment parameter:

```cpp
// Inside an OpenMP parallel region:
int tid = omp_get_thread_num();
Sampler3D sampler(octree, alias_method, my_func, 2 * tid + 1);
Point p = sampler.sample();
```

The increment `2*tid+1` ensures that each thread has a distinct, odd sequence increment, providing robust isolation between thread streams without requiring complex seeding strategies.

## Performance & Tuning Defaults

The default settings are tuned to the **Combined Aggressive Tuning** configuration (mass cutoff `threshold = 0.01`, flatness `efficiency_limit = 0.5`) to optimize tree compactness and memory footprint:

For a peaked Gaussian function ($t=100$) and 100 million samples:
- **Octree Construction Time**: ~0.04s (9x faster than baseline, parallelized over root-level children via OpenMP)
- **Active Leaf Boxes**: 133,400 (9x more compact than baseline)
- **Sampling Time**: ~2.9s (100M samples on 8 threads)
- **Efficiency**: ~60.8%
- **Z-Score**: ~0.77 (Statistically unbiased)
- **Relative Error**: ~0.008%
