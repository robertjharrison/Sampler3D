#include "alias_method.h"
#include <iostream>
#include <vector>
#include <map>
#include <cassert>

int main() {
    std::vector<double> weights = {1.0, 2.0, 7.0}; // Total 10.0
    AliasMethod alias(weights);
    PCG64 gen(12345);
    gen.warmup();

    std::map<size_t, int> counts;
    int num_samples = 100000;
    for (int i = 0; i < num_samples; ++i) {
        counts[alias.sample(gen)]++;
    }

    std::cout << "Testing Alias Method with weights: 1.0, 2.0, 7.0" << std::endl;
    for (auto const& [idx, count] : counts) {
        double freq = (double)count / num_samples;
        std::cout << "Index " << idx << ": frequency " << freq << " (expected ~" << weights[idx]/10.0 << ")" << std::endl;
        assert(std::abs(freq - weights[idx]/10.0) < 0.05);
    }

    std::cout << "Alias Method test passed!" << std::endl;
    return 0;
}
