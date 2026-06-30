#include "sampler.h"
#include <cmath>

Sampler3D::Sampler3D(const Octree& octree_ref, AliasMethod& alias_ref, std::function<double(double, double, double)> f, PCG64_OpenMP_Manager& mgr)
    : octree(octree_ref), 
      alias_method(alias_ref),
      func(f),
      rng_manager(mgr) {
}

Point Sampler3D::sample() {
    total_samples++;

    // Obtain the thread-local stream proxy
    auto gen = rng_manager.stream();

    while (true) {
        total_trials++;
        
        // Step 1: Pick a box using Alias Method (O(1) complexity)
        size_t idx = alias_method.sample(gen);
        const OctreeNode* leaf = octree.leaves[idx];

        // Step 2: Sample coordinates uniformly within the selected box
        double x = leaf->bounds.x0 + gen.nextDouble() * leaf->bounds.dx;
        double y = leaf->bounds.y0 + gen.nextDouble() * leaf->bounds.dy;
        double z = leaf->bounds.z0 + gen.nextDouble() * leaf->bounds.dz;

        // Step 3: Rejection Sampling check
        double val = func(x, y, z);
        double accept_prob = std::abs(val) / leaf->max_abs_f;

        if (gen.nextDouble() < accept_prob) {
            return {x, y, z, val > 0.0 ? 1.0 : -1.0};
        }
    }
}
