#include "pcg64.h"
#include "pcg64_openmp.h"
#include <iostream>
#include <cassert>
#include <stdexcept>
#include <random>
#include <sstream>
#include <vector>
#include <type_traits>
#include <limits>
#include <omp.h>

void test_increment_validation() {
    // Odd increment should succeed
    try {
        PCG64 gen(1, 3);
    } catch (...) {
        assert(false && "PCG64 constructor threw on odd increment");
    }

    // Even increment should throw std::invalid_argument
    bool threw = false;
    try {
        PCG64 gen(1, 4);
    } catch (const std::invalid_argument& e) {
        threw = true;
    }
    assert(threw && "PCG64 constructor did not throw on even increment");
}

void test_determinism() {
    PCG64 gen1(42, 1234567);
    PCG64 gen2(42, 1234567);

    for (int i = 0; i < 100; ++i) {
        assert(gen1.next() == gen2.next());
    }
}

void test_jump() {
    PCG64 gen1(42, 1234567);
    PCG64 gen2(42, 1234567);

    // Generate 8 numbers (2^3) from gen1
    for (int i = 0; i < 8; ++i) {
        gen1.next();
    }

    // Jump gen2 by 2^3 steps
    gen2.jump(3);

    assert(gen1.next() == gen2.next());
}

// 1. Test PCG64 C++ STL Random Number Engine compatibility
void test_stl_compliance_pcg64() {
    std::cout << "  Testing STL compliance for PCG64..." << std::endl;

    // Check basic types and min/max
    static_assert(std::is_same_v<PCG64::result_type, uint64_t>, "result_type must be uint64_t");
    assert(PCG64::min() == 0);
    assert(PCG64::max() == std::numeric_limits<uint64_t>::max());
    assert(PCG64::default_seed == 1ULL);

    // Test default constructor
    PCG64 gen_def;
    PCG64 gen_seed1(PCG64::default_seed);
    assert(gen_def == gen_seed1);

    // Test seeding
    PCG64 gen3(100, 12345);
    gen3.seed(PCG64::default_seed);
    assert(gen3 == gen_def);

    // Test seed sequence constructor & seed function
    std::seed_seq sseq{1, 2, 3, 4};
    PCG64 gen_sseq(sseq);
    PCG64 gen_sseq_seeded;
    gen_sseq_seeded.seed(sseq);
    assert(gen_sseq == gen_sseq_seeded);

    // Test usage with STL distributions
    std::uniform_int_distribution<int> int_dist(1, 10);
    std::uniform_real_distribution<double> real_dist(0.0, 1.0);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    PCG64 gen_stl(123456);
    // Just verify they run without compile/runtime errors
    for (int i = 0; i < 100; ++i) {
        int r_int = int_dist(gen_stl);
        double r_real = real_dist(gen_stl);
        double r_norm = norm_dist(gen_stl);
        assert(r_int >= 1 && r_int <= 10);
        assert(r_real >= 0.0 && r_real < 1.0);
        (void)r_norm; // keep compiler happy
    }

    // Test discard (equivalent to skipping step-by-step)
    PCG64 gen_disc1(42, 1234567);
    PCG64 gen_disc2(42, 1234567);
    gen_disc1.discard(100);
    for (int i = 0; i < 100; ++i) {
        gen_disc2.next();
    }
    assert(gen_disc1 == gen_disc2);

    // Test operators == and !=
    PCG64 gen_eq1(42, 1234567);
    PCG64 gen_eq2(42, 1234567);
    assert(gen_eq1 == gen_eq2);
    gen_eq1.next();
    assert(gen_eq1 != gen_eq2);

    // Test stream operators << and >> (serialization)
    PCG64 gen_ser1(98765, 432109);
    gen_ser1.next();
    gen_ser1.next();

    std::stringstream ss;
    ss << gen_ser1;

    PCG64 gen_ser2;
    ss >> gen_ser2;

    assert(gen_ser1 == gen_ser2);
    assert(gen_ser1.next() == gen_ser2.next());
}

// 2. Test OpenMP variant compliance & functionality
void test_openmp_variant() {
    std::cout << "  Testing OpenMP variant and stream isolation..." << std::endl;

    // Check basic types and min/max for ThreadStream
    static_assert(std::is_same_v<PCG64_OpenMP_Manager::ThreadStream::result_type, uint64_t>, "result_type must be uint64_t");
    assert(PCG64_OpenMP_Manager::ThreadStream::min() == 0);
    assert(PCG64_OpenMP_Manager::ThreadStream::max() == std::numeric_limits<uint64_t>::max());

    PCG64_OpenMP_Manager manager(42ULL);

    // Verify ThreadStream compiles/works with STL distributions
    {
        auto stream = manager.stream();
        std::uniform_int_distribution<int> dist(1, 100);
        int val = dist(stream);
        assert(val >= 1 && val <= 100);
    }

    // Verify reentrancy guard behavior
    {
        auto s1 = manager.stream();
        uint64_t val1 = s1();
        {
            auto s2 = manager.stream();
            uint64_t val2 = s2();
            // Since s2 is nested in the same thread, it should continue the stream sequence
            assert(val1 != val2);
        }
    }

    // Verify stream isolation and that each thread generates a different stream of numbers
    int num_threads = omp_get_max_threads();
    if (num_threads < 2) {
        // Force at least 2 threads for test if supported
        omp_set_num_threads(4);
        num_threads = omp_get_max_threads();
    }
    std::cout << "    Running parallel region with " << num_threads << " threads..." << std::endl;

    std::vector<std::vector<uint64_t>> generated_numbers(num_threads, std::vector<uint64_t>(10));

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto stream = manager.stream();
        for (int i = 0; i < 10; ++i) {
            generated_numbers[tid][i] = stream();
        }
    }

    // Check that each thread generated a different sequence of numbers (stream isolation)
    for (int i = 0; i < num_threads; ++i) {
        for (int j = i + 1; j < num_threads; ++j) {
            // They should not match
            bool match = true;
            for (int k = 0; k < 10; ++k) {
                if (generated_numbers[i][k] != generated_numbers[j][k]) {
                    match = false;
                    break;
                }
            }
            assert(!match && "Streams of different threads must be independent/non-correlated");
        }
    }

    // Verify checkpoint and restore functionality
    std::cout << "    Testing Checkpoint and Restore..." << std::endl;
    PCG64_OpenMP_Manager mgr_ckpt(999ULL);

    // Save initial state (checkpoint)
    auto saved_states = mgr_ckpt.get_states();

    // Generate some numbers across threads
    std::vector<uint64_t> first_run(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto stream = mgr_ckpt.stream();
        first_run[tid] = stream();
    }

    // Restore state from checkpoint
    mgr_ckpt.set_states(saved_states);

    // Generate again
    std::vector<uint64_t> second_run(num_threads);
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto stream = mgr_ckpt.stream();
        second_run[tid] = stream();
    }

    // Verify determinism across checkpoint restore
    for (int i = 0; i < num_threads; ++i) {
        assert(first_run[i] == second_run[i] && "Checkpoint restore did not reproduce the same stream");
    }
}

int main() {
    std::cout << "Running PCG64 tests..." << std::endl;
    test_increment_validation();
    test_determinism();
    test_jump();
    test_stl_compliance_pcg64();
    test_openmp_variant();
    std::cout << "PCG64 tests passed!" << std::endl;
    return 0;
}
