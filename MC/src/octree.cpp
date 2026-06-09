#include "octree.h"
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

StencilResult Octree::estimate_stats(const Box& b, const std::function<double(double, double, double)>& f) {
    double max_val = 0.0;
    double sum_val = 0.0;
    const int N = 5;
    
    // Precompute grid coordinates to eliminate arithmetic operations in the hot nested loops
    double xs[N], ys[N], zs[N];
    double dx = (b.x1 - b.x0) / (N - 1.0);
    double dy = (b.y1 - b.y0) / (N - 1.0);
    double dz = (b.z1 - b.z0) / (N - 1.0);
    
    for (int i = 0; i < N; ++i) {
        xs[i] = b.x0 + i * dx;
        ys[i] = b.y0 + i * dy;
        zs[i] = b.z0 + i * dz;
    }

    for (int i = 0; i < N; ++i) {
        double x = xs[i];
        for (int j = 0; j < N; ++j) {
            double y = ys[j];
            for (int k = 0; k < N; ++k) {
                double val = std::abs(f(x, y, zs[k]));
                max_val = std::max(max_val, val);
                sum_val += val;
            }
        }
    }
    return {max_val, sum_val / (N * N * N)};
}

void OctreeNode::refine(const std::function<double(double, double, double)>& f, int depth, int max_depth, double threshold, double root_mass, double safety, double efficiency_limit, RefineStats& stats) {
    double mass = max_abs_f * bounds.volume();

    if (mass < threshold * root_mass && depth > 0) {
        is_leaf = true;
        stats.mass_terminations++;
        stats.total_leaves++;
        return;
    }

    if (depth >= max_depth) {
        is_leaf = true;
        stats.depth_terminations++;
        stats.total_leaves++;
        return;
    }

    if (max_abs_f > 1e-15 && (mean_abs_f / max_abs_f) > efficiency_limit) {
        is_leaf = true;
        stats.efficiency_terminations++;
        stats.total_leaves++;
        return;
    }

    is_leaf = false;
    double c_dx = bounds.dx * 0.5;
    double c_dy = bounds.dy * 0.5;
    double c_dz = bounds.dz * 0.5;

    double mx = bounds.x0 + c_dx;
    double my = bounds.y0 + c_dy;
    double mz = bounds.z0 + c_dz;

    // Parallelize the top-level recursion (depth == 0) using OpenMP.
    // Each thread will build and refine one of the 8 disjoint subtrees in parallel.
    #pragma omp parallel for if(depth == 0) schedule(dynamic)
    for (int i = 0; i < 8; ++i) {
        double nx0 = (i & 1) ? mx : bounds.x0;
        double ny0 = (i & 2) ? my : bounds.y0;
        double nz0 = (i & 4) ? mz : bounds.z0;
        Box b(nx0, ny0, nz0, c_dx, c_dy, c_dz, typename Box::Dyadic{});

        children[i] = std::make_unique<OctreeNode>(b);
        stats.total_nodes++;
        
        auto child_stats = Octree::estimate_stats(b, f);
        children[i]->max_abs_f = child_stats.max_val * (1.0 + safety) + 1e-12;
        children[i]->mean_abs_f = child_stats.mean_val;
        children[i]->refine(f, depth + 1, max_depth, threshold, root_mass, safety, efficiency_limit, stats);
    }
}

void OctreeNode::collect_leaves(std::vector<const OctreeNode*>& leaf_nodes) const {
    if (is_leaf) {
        if (max_abs_f > 1e-15) {
            leaf_nodes.push_back(this);
        }
        return;
    }
    for (int i = 0; i < 8; ++i) {
        if (children[i]) {
            children[i]->collect_leaves(leaf_nodes);
        }
    }
}

Octree::Octree(Box bounds, const std::function<double(double, double, double)>& f, int max_depth, double threshold, double safety_val, double efficiency_limit_val) 
    : safety(safety_val), efficiency_limit(efficiency_limit_val) {
    root = std::make_unique<OctreeNode>(bounds);
    stats.total_nodes.store(1);
    
    auto root_stats = estimate_stats(bounds, f);
    root->max_abs_f = root_stats.max_val * (1.0 + safety) + 1e-12;
    root->mean_abs_f = root_stats.mean_val;
    double root_mass = root->max_abs_f * bounds.volume();
    root->refine(f, 0, max_depth, threshold, root_mass, safety, efficiency_limit, stats);
    root->collect_leaves(leaves);
}
