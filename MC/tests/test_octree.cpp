#include "octree.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_box() {
    Box b(-1.0, -2.0, -3.0, 1.0, 2.0, 3.0);
    assert(std::abs(b.volume() - 48.0) < 1e-9);
    assert(b.contains(0.0, 0.0, 0.0));
    assert(!b.contains(2.0, 0.0, 0.0));
}

void test_octree_volume_conservation() {
    // Define a localized gaussian function
    auto f = [](double x, double y, double z) {
        return std::exp(-(x*x + y*y + z*z));
    };

    Box volume(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0);
    double expected_vol = volume.volume(); // 64.0

    // Construct octree with depth 4
    Octree octree(volume, f, 4, 0.01);

    double total_leaf_vol = 0.0;
    for (const auto* leaf : octree.leaves) {
        total_leaf_vol += leaf->bounds.volume();
    }

    std::cout << "Expected total volume: " << expected_vol << ", Leaves total volume: " << total_leaf_vol << std::endl;
    assert(std::abs(total_leaf_vol - expected_vol) < 1e-9);
}

void test_octree_refinement() {
    // A function that is 0 everywhere except in one corner
    auto f = [](double x, double y, double z) {
        if (x > 0.5 && y > 0.5 && z > 0.5) return 1.0;
        return 0.0;
    };

    Box volume(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);
    Octree octree(volume, f, 5, 0.01);

    // Ensure we have leaves collected
    assert(!octree.leaves.empty());
    
    // The leaf nodes with f != 0 should have smaller volume than the root
    for (const auto* leaf : octree.leaves) {
        assert(leaf->bounds.volume() < volume.volume());
    }
}

int main() {
    std::cout << "Running Octree tests..." << std::endl;
    test_box();
    test_octree_volume_conservation();
    test_octree_refinement();
    std::cout << "Octree tests passed!" << std::endl;
    return 0;
}
