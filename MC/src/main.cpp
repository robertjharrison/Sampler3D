#include "sampler.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <map>
#include <chrono>
#include <omp.h>
#include <string>

// --- TEST FUNCTION 1 (SINES) ---
double f1(double x, double y, double z) {
    return std::sin(x) * std::cos(y) * std::sin(z);
}

double i_sin(int n) {
    if (n % 2 == 0) return 0.0;
    if (n == 1) return 2.0 * M_PI;
    if (n == 3) return 2.0 * std::pow(M_PI, 3) - 12.0 * M_PI;
    if (n == 5) return 2.0 * std::pow(M_PI, 5) - 40.0 * std::pow(M_PI, 3) + 240.0 * M_PI;
    return 0.0;
}

double i_cos(int n) {
    if (n % 2 != 0) return 0.0;
    if (n == 0) return 0.0;
    if (n == 2) return -4.0 * M_PI;
    if (n == 4) return -8.0 * std::pow(M_PI, 3) + 48.0 * M_PI;
    if (n == 6) return -12.0 * std::pow(M_PI, 5) + 240.0 * std::pow(M_PI, 3) - 1440.0 * M_PI;
    return 0.0;
}

double analytic_f1(int i, int j, int k) {
    return i_sin(i) * i_cos(j) * i_sin(k);
}

// --- TEST FUNCTION 2 (GAUSSIAN) ---
double f2(double x, double y, double z) {
    const double t = 100.0;
    return x * y * z * std::exp(-t * (x*x + y*y + z*z));
}

double H_integral(int n, double alpha) {
    if (n % 2 != 0) return 0.0;
    double H0 = std::sqrt(M_PI / alpha) * std::erf(M_PI * std::sqrt(alpha));
    if (n == 0) return H0;
    double H2 = (1.0 / (2.0 * alpha)) * (H0 - 2.0 * M_PI * std::exp(-alpha * M_PI * M_PI));
    if (n == 2) return H2;
    double H4 = (3.0 / (2.0 * alpha)) * H2 - (std::pow(M_PI, 3) / alpha) * std::exp(-alpha * M_PI * M_PI);
    if (n == 4) return H4;
    double H6 = (5.0 / (2.0 * alpha)) * H4 - (std::pow(M_PI, 5) / alpha) * std::exp(-alpha * M_PI * M_PI);
    if (n == 6) return H6;
    return 0.0;
}

double analytic_f2(int i, int j, int k) {
    const double t = 100.0;
    return H_integral(i + 1, t) * H_integral(j + 1, t) * H_integral(k + 1, t);
}

double norm_f2() {
    const double t = 100.0;
    double L = (1.0 / t) * (1.0 - std::exp(-t * M_PI * M_PI));
    return L * L * L;
}

void run_parallel_study(const std::string& name, std::function<double(double, double, double)> f, 
                          std::function<double(int, int, int)> analytic_fn, double approx_norm,
                          int target_i, int target_j, int target_k,
                          double threshold = 0.01, double efficiency_limit = 0.5) {
    std::cout << "\n=== Parallel Study: " << name << " ===" << std::endl;
    std::cout << "Parameters: Threshold=" << threshold << ", Efficiency Limit=" << efficiency_limit << std::endl;
    std::cout << "Target Moment: (" << target_i << "," << target_j << "," << target_k << ")" << std::endl;
    
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Octree Construction
    Box volume = {-M_PI, -M_PI, -M_PI, M_PI, M_PI, M_PI};
    Octree octree(volume, f, 30, threshold, 0.1, efficiency_limit);
    
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "\nOctree Structure Statistics:" << std::endl;
    std::cout << "  Total Nodes in Tree:       " << octree.stats.total_nodes.load() << std::endl;
    std::cout << "  Active Leaf Boxes (Leaves): " << octree.leaves.size() << std::endl;
    std::cout << "  Total Leaf Nodes generated: " << octree.stats.total_leaves.load() << std::endl;
    std::cout << "  Termination Triggers:" << std::endl;
    std::cout << "    - Mass Threshold reached: " << octree.stats.mass_terminations.load() << std::endl;
    std::cout << "    - Max Depth reached:      " << octree.stats.depth_terminations.load() << std::endl;
    std::cout << "    - Efficiency reached:     " << octree.stats.efficiency_terminations.load() << std::endl;

    // 2. Alias Table Generation
    std::vector<double> weights;
    for (const auto* leaf : octree.leaves) {
        weights.push_back(leaf->max_abs_f * leaf->bounds.volume());
    }
    AliasMethod alias_method(weights);
    
    auto t2 = std::chrono::high_resolution_clock::now();

    // 3. Parallel Sampling
    long long num_samples = 100000000; // 100M samples
    double total_sum_w = 0;
    double total_sum_w2 = 0;
    long long total_trials = 0;

    #pragma omp parallel reduction(+:total_sum_w, total_sum_w2, total_trials)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        long long nsamples_thread = num_samples / nthreads;
        if (tid == nthreads - 1) nsamples_thread += num_samples % nthreads;

        // Correct PRNG initialization: fixed seed (1), unique odd increment (2*tid+1)
        Sampler3D sampler(octree, alias_method, f, (pcg_ulong_t)(2 * tid + 1));

        double local_sum_w = 0;
        double local_sum_w2 = 0;

        for (long long s = 0; s < nsamples_thread; ++s) {
            Point p = sampler.sample();
            double w = std::pow(p.x, target_i) * std::pow(p.y, target_j) * std::pow(p.z, target_k) * p.sign;
            local_sum_w += w;
            local_sum_w2 += w * w;
        }
        total_sum_w = local_sum_w;
        total_sum_w2 = local_sum_w2;
        total_trials = sampler.total_trials;
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    // Results Calculation
    double analytic = analytic_fn(target_i, target_j, target_k);
    double numeric = (approx_norm / num_samples) * total_sum_w;
    double rel_err = std::abs(analytic - numeric) / (std::abs(analytic) + 1e-15);
    double mean_w = total_sum_w / num_samples;
    double var_w = (total_sum_w2 / num_samples) - (mean_w * mean_w);
    double std_err_integral = approx_norm * std::sqrt(var_w / num_samples);
    double z_score = std::abs(numeric - analytic) / (std_err_integral + 1e-15);
    double efficiency = (double)num_samples / total_trials;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Analytic:    " << std::scientific << analytic << std::endl;
    std::cout << "Numeric:     " << std::scientific << numeric << std::endl;
    std::cout << "Rel. Error:  " << std::fixed << rel_err * 100.0 << "%" << std::endl;
    std::cout << "Std Error:   " << std::scientific << std_err_integral << std::endl;
    std::cout << "Z-Score:     " << std::fixed << z_score << std::endl;
    std::cout << "Efficiency:  " << std::fixed << efficiency * 100.0 << "%" << std::endl;

    std::cout << "\nTiming Report:" << std::endl;
    std::cout << "  Octree Const: " << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
    std::cout << "  Alias Table:  " << std::chrono::duration<double>(t2 - t1).count() << " s" << std::endl;
    std::cout << "  Sampling:     " << std::chrono::duration<double>(t3 - t2).count() << " s" << std::endl;
    std::cout << "  Total:        " << std::chrono::duration<double>(t3 - t0).count() << " s" << std::endl;
}

int main() {
    int nthreads = omp_get_max_threads();
    std::cout << "Running with " << nthreads << " OpenMP threads." << std::endl;

    // Run the studies using the default parameters (threshold = 0.01, efficiency_limit = 0.5)
    run_parallel_study("Sine Product Function", f1, analytic_f1, 64.0, 1, 2, 1);
    run_parallel_study("Gaussian Function", f2, analytic_f2, norm_f2(), 1, 1, 1);

    return 0;
}
