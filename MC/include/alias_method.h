#pragma once

#include <vector>
#include <cstddef>
#include "pcg64.h"

/**
 * @brief Walker's Alias Method for O(1) sampling from discrete probability distributions.
 * 
 * Precomputes a table that allows drawing a sample from an arbitrary discrete distribution
 * in constant time using two random numbers.
 */
class AliasMethod {
public:
    struct Entry {
        double prob;   ///< Accept probability for the original index
        size_t alias;  ///< Alias index if rejected
    };

    std::vector<Entry> table;

    /**
     * @brief Construct a new Alias Method object from a vector of weights.
     * @param weights The non-negative weights representing the probability distribution.
     */
    AliasMethod(const std::vector<double>& weights);

    /**
     * @brief Draw a sample index from the distribution.
     * 
     * Uses a fast 128-bit multiplication-based range reduction (unbiased) instead of
     * modulo arithmetic to avoid slow division instructions in the hot path.
     * 
     * @param gen The PCG64 random number generator.
     * @return size_t The sampled index.
     */
    inline size_t sample(PCG64& gen) {
        if (table.empty()) return 0;

        // Fast range mapping without modulo: (uint64_t * range) >> 64
        uint64_t r = gen.next();
        size_t i = static_cast<size_t>((static_cast<__uint128_t>(r) * table.size()) >> 64);
        
        if (gen.nextDouble() < table[i].prob) {
            return i;
        } else {
            return table[i].alias;
        }
    }
};
