#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <atomic>

/**
 * @brief Bounding Box structure in 3D Cartesian coordinates.
 */
struct Box {
    double x0, y0, z0; ///< Minimum coordinates
    double x1, y1, z1; ///< Maximum coordinates
    double vol;        ///< Volume of the box
    double dx, dy, dz; ///< Precomputed dimensions for uniform sampling

    Box(double x0_, double y0_, double z0_, double x1_, double y1_, double z1_)
        : x0(x0_), y0(y0_), z0(z0_), x1(x1_), y1(y1_), z1(z1_),
          vol((x1_ - x0_) * (y1_ - y0_) * (z1_ - z0_)),
          dx(x1_ - x0_), dy(y1_ - y0_), dz(z1_ - z0_) {}

    struct Dyadic {};
    Box(double x0_, double y0_, double z0_, double dx_, double dy_, double dz_, Dyadic)
        : x0(x0_), y0(y0_), z0(z0_),
          x1(x0_ + dx_), y1(y0_ + dy_), z1(z0_ + dz_),
          vol(dx_ * dy_ * dz_),
          dx(dx_), dy(dy_), dz(dz_) {}

    double volume() const {
        return vol;
    }

    bool contains(double x, double y, double z) const {
        return x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1;
    }
};

/**
 * @brief Statistics tracked during octree generation refinement.
 */
struct RefineStats {
    std::atomic<size_t> mass_terminations{0};       ///< Count of terminations due to mass/threshold
    std::atomic<size_t> depth_terminations{0};      ///< Count of terminations due to reaching max depth
    std::atomic<size_t> efficiency_terminations{0}; ///< Count of terminations due to flatness (mean/max > efficiency_limit)
    std::atomic<size_t> total_nodes{0};             ///< Total nodes constructed (internal + leaf)
    std::atomic<size_t> total_leaves{0};            ///< Total leaf nodes constructed

    RefineStats() = default;
    
    // Custom copy constructor because std::atomic is not copyable
    RefineStats(const RefineStats& other) {
        mass_terminations.store(other.mass_terminations.load());
        depth_terminations.store(other.depth_terminations.load());
        efficiency_terminations.store(other.efficiency_terminations.load());
        total_nodes.store(other.total_nodes.load());
        total_leaves.store(other.total_leaves.load());
    }
};

/**
 * @brief Result of estimating statistics within a box.
 */
struct StencilResult {
    double max_val;
    double mean_val;
};

/**
 * @brief Represents a node in the 3D adaptive octree.
 */
class OctreeNode {
public:
    Box bounds;
    double max_abs_f = 0.0;
    double mean_abs_f = 0.0;
    std::unique_ptr<OctreeNode> children[8];
    bool is_leaf = true;

    OctreeNode(Box b) : bounds(b) {}

    void refine(const std::function<double(double, double, double)>& f, int depth, int max_depth, double threshold, double root_mass, double safety, double efficiency_limit, RefineStats& stats);
    void collect_leaves(std::vector<const OctreeNode*>& leaf_nodes) const;
};

/**
 * @brief Represents the adaptive spatial partitioning octree.
 */
class Octree {
public:
    std::unique_ptr<OctreeNode> root;
    std::vector<const OctreeNode*> leaves;
    double safety;
    double efficiency_limit;
    RefineStats stats;

    Octree(Box bounds, const std::function<double(double, double, double)>& f, int max_depth, double threshold = 0.01, double safety = 0.1, double efficiency_limit = 0.5);

    static StencilResult estimate_stats(const Box& b, const std::function<double(double, double, double)>& f);
};
