#pragma once

#include "octree.h"
#include "alias_method.h"
#include "pcg64_openmp.h"
#include <functional>

/**
 * @brief Represents a sampled 3D point with its coordinates and function sign.
 */
struct Point {
    double x;    ///< X-coordinate
    double y;    ///< Y-coordinate
    double z;    ///< Z-coordinate
    double sign; ///< Sign of the function value at (x,y,z): +1.0 or -1.0
};

/**
 * @brief Thread-safe 3D function sampler using parallel rejection sampling and Walker's Alias Method.
 */
class Sampler3D {
public:
    const Octree& octree;                               ///< Reference to the adaptive octree
    AliasMethod& alias_method;                          ///< Reference to the alias method distribution
    std::function<double(double, double, double)> func; ///< Target function to sample
    PCG64_OpenMP_Manager& rng_manager;                  ///< Reference to the OpenMP RNG manager
    
    long long total_trials = 0;                         ///< Number of trials (for efficiency measurement)
    long long total_samples = 0;                        ///< Number of accepted samples

    /**
     * @brief Construct a new Sampler3D object.
     * @param octree_ref The constructed Octree.
     * @param alias_ref The constructed AliasMethod table.
     * @param f The 3D function to sample.
     * @param mgr The PCG64 OpenMP manager.
     */
    Sampler3D(const Octree& octree_ref, AliasMethod& alias_ref, std::function<double(double, double, double)> f, PCG64_OpenMP_Manager& mgr);

    /**
     * @brief Draw a single accepted point sample using rejection sampling.
     * @return Point The sampled point.
     */
    Point sample();

    /**
     * @brief Get the sampling efficiency (accepted samples / total trials).
     */
    double efficiency() const { 
        return total_trials > 0 ? (double)total_samples / total_trials : 0.0; 
    }
};
