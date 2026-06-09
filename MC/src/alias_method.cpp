#include "alias_method.h"
#include <numeric>
#include <vector>
#include <stdexcept>
#include <algorithm>

AliasMethod::AliasMethod(const std::vector<double>& weights) {
    size_t n = weights.size();
    if (n == 0) return;

    // Validate weights and compute sum
    double sum = 0.0;
    for (double w : weights) {
        if (w < 0.0) {
            throw std::invalid_argument("AliasMethod: Weights must be non-negative.");
        }
        sum += w;
    }

    table.resize(n);
    std::vector<double> probs(n);

    // If total sum is zero or extremely small, treat as uniform distribution
    if (sum <= 1e-15) {
        for (size_t i = 0; i < n; ++i) {
            probs[i] = 1.0;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            probs[i] = (weights[i] * n) / sum;
        }
    }

    // Use vectors instead of std::stack to avoid default std::deque dynamic chunk allocation
    std::vector<size_t> small;
    std::vector<size_t> large;
    small.reserve(n);
    large.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        if (probs[i] < 1.0) {
            small.push_back(i);
        } else {
            large.push_back(i);
        }
    }

    while (!small.empty() && !large.empty()) {
        size_t s = small.back(); small.pop_back();
        size_t l = large.back(); large.pop_back();

        table[s].prob = probs[s];
        table[s].alias = l;

        probs[l] = (probs[l] + probs[s]) - 1.0;
        if (probs[l] < 1.0) {
            small.push_back(l);
        } else {
            large.push_back(l);
        }
    }

    // Handle remaining elements due to floating point inaccuracies
    while (!large.empty()) {
        size_t l = large.back(); large.pop_back();
        table[l].prob = 1.0;
        table[l].alias = l;
    }

    while (!small.empty()) {
        size_t s = small.back(); small.pop_back();
        table[s].prob = 1.0;
        table[s].alias = s;
    }
}
