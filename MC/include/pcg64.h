#pragma once

#include <cstdint>
#include <stdexcept>
#include <cstddef>

// Use 128-bit unsigned integer for internal state
using pcg_ulong_t = __uint128_t;

/**
 * @brief PCG64 Random Number Generator.
 * 
 * Implements the PCG-DXSM (Double Xor Shift Multiply) generator algorithm.
 * Suitable for high-performance parallel sampling due to its speed,
 * good statistical properties, and ability to support multiple independent streams.
 * 
 * See: https://www.pcg-random.org/using-pcg-cpp.html
 */
class PCG64 {
private:
    pcg_ulong_t state; ///< LCG internal state
    pcg_ulong_t inc;   ///< Stream sequence increment (must be odd)

    // Multiplier for the LCG step (constant for PCG64)
    static constexpr uint64_t MUL = 15750249268501108917ULL;

    // Default stream increment
    static constexpr pcg_ulong_t DEFAULT_INC = ((pcg_ulong_t)6364136223846793005ULL << 64) | (pcg_ulong_t)1442695040888963407ULL;

public:
    /**
     * @brief Construct a new PCG64 generator.
     * @param state Initial seed.
     * @param inc Unique stream increment. Must be odd for a full-period generator.
     * @throws std::invalid_argument if inc is even.
     */
    PCG64(pcg_ulong_t state = 1, pcg_ulong_t inc = DEFAULT_INC)
        : state(state), inc(inc)
    {
        if ((inc & 0x1) == 0) {
            throw std::invalid_argument("PCG64: Increment must be odd for full period.");
        }
    }

    /**
     * @brief Advance the LCG state and generate a 64-bit random unsigned integer.
     * @return uint64_t 64-bit random number.
     */
    inline uint64_t next() noexcept {
        /* Linear Congruential Generator step */
        pcg_ulong_t old_state = state;
        state = old_state * MUL + inc;

        /* DXSM (Double Xor Shift Multiply) permuted output */
        uint64_t hi = static_cast<uint64_t>(old_state >> 64);
        uint64_t lo = static_cast<uint64_t>(old_state | 1);
        
        hi ^= hi >> 32;
        hi *= MUL;
        hi ^= hi >> 48;
        hi *= lo;
        
        return hi;
    }

    /**
     * @brief Functor operator to generate a 64-bit random number.
     */
    inline uint64_t operator()() noexcept {
        return next();
    }

    /**
     * @brief Jump ahead by 2^n steps in O(log(2^n)) = O(n) time.
     * 
     * Efficiently advances the RNG state without generating intermediate numbers.
     * @param n The power of 2 steps to jump (i.e. advances state by 2^n steps).
     */
    void jump(size_t n) noexcept {
        pcg_ulong_t m = MUL;
        pcg_ulong_t c = inc;
        
        for (size_t i = 0; i < n; ++i) {
            c = c * (m + 1);
            m = m * m;
        }

        state = state * m + c;
    }

    /**
     * @brief Generate a double-precision floating-point number in [0, 1).
     * @return double random number in range [0, 1).
     */
    inline double nextDouble() noexcept {
        return (static_cast<double>(next() >> 11) * 0x1.0p-53);
    }

    /**
     * @brief Get the current internal state (seed) of the generator.
     */
    pcg_ulong_t getState() const noexcept { return state; }
    
    /**
     * @brief Reset the state (seed) to a specific value.
     */
    void setState(pcg_ulong_t s) noexcept { state = s; }

    /**
     * @brief Get the LCG multiplier value.
     */
    static constexpr uint64_t getMultiplier() noexcept { return MUL; }

    /**
     * @brief Get the stream sequence increment.
     */
    pcg_ulong_t getIncrement() const noexcept { return inc; }

    /**
     * @brief Sequentially warm up the generator to discard early state correlations.
     * @param nsteps Number of initial generator steps.
     */
    void warmup(int nsteps = 1000) noexcept {
        for (int i = 0; i < nsteps; ++i) {
            next();
        }
    }
};
