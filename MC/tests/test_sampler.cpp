#include "sampler.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main() {
    std::cout << "Running Sampler3D tests..." << std::endl;

    // A simple product of sines function
    auto f = [](double x, double y, double z) {
        return std::sin(x) * std::sin(y) * std::sin(z);
    };

    Box volume(-M_PI, -M_PI, -M_PI, M_PI, M_PI, M_PI);
    Octree octree(volume, f, 4, 0.05);

    std::vector<double> weights;
    for (const auto* leaf : octree.leaves) {
        weights.push_back(leaf->max_abs_f * leaf->bounds.volume());
    }
    AliasMethod alias_method(weights);

    Sampler3D sampler(octree, alias_method, f, 1);

    // Generate samples
    const int num_samples = 1000;
    for (int i = 0; i < num_samples; ++i) {
        Point p = sampler.sample();
        
        // 1. Verify bounds
        assert(volume.contains(p.x, p.y, p.z));

        // 2. Verify sign
        double val = f(p.x, p.y, p.z);
        double expected_sign = val > 0 ? 1.0 : -1.0;
        assert(p.sign == expected_sign);
    }

    std::cout << "Sampler efficiency: " << sampler.efficiency() * 100.0 << "%" << std::endl;
    assert(sampler.efficiency() > 0.0 && sampler.efficiency() <= 1.0);

    std::cout << "Sampler3D tests passed!" << std::endl;
    return 0;
}
