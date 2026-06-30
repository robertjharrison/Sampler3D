#pragma once

#include <omp.h>
#include <vector>
#include <limits>

#include "pcg64.h"

struct PCG64_State {
    pcg_ulong_t state;
    pcg_ulong_t inc;
};

class PCG64_OpenMP_Manager {
private:
    std::vector<PCG64_State> persistent_states;
    
    // --- The Active TLS State ---
    static inline thread_local PCG64 local_rng;
    
    // --- Reentrancy Trackers ---
    // Tracks how many ThreadStream proxies exist in the current thread's call stack
    static inline thread_local int guard_count = 0; 
    
    // Caches the thread ID to ensure the destructor writes back to the correct slot
    static inline thread_local int cached_tid = -1;

public:
    PCG64_OpenMP_Manager() {
        int num_threads = omp_get_max_threads();
        persistent_states.resize(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            persistent_states[i].state = 1;
            persistent_states[i].inc = 2 * i + 1;
            PCG64 tmp(persistent_states[i].state, persistent_states[i].inc);
            tmp.warmup();
            persistent_states[i].state = tmp.getState();
        }
    }

    explicit PCG64_OpenMP_Manager(pcg_ulong_t base_seed, int num_threads = -1) {
        if (num_threads <= 0) {
            num_threads = omp_get_max_threads();
        }
        persistent_states.resize(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            persistent_states[i].state = base_seed;
            persistent_states[i].inc = 2 * i + 1;
            PCG64 tmp(persistent_states[i].state, persistent_states[i].inc);
            tmp.warmup();
            persistent_states[i].state = tmp.getState();
        }
    }

    // Checkpoint methods to save and restore persistent generator states
    const std::vector<PCG64_State>& get_states() const noexcept {
        return persistent_states;
    }

    void set_states(const std::vector<PCG64_State>& states) {
        persistent_states = states;
    }

    PCG64_State get_state(int tid) const {
        if (tid < 0 || tid >= static_cast<int>(persistent_states.size())) {
            throw std::out_of_range("PCG64_OpenMP_Manager: thread ID out of range");
        }
        return persistent_states[tid];
    }

    void set_state(int tid, const PCG64_State& s) {
        if (tid < 0 || tid >= static_cast<int>(persistent_states.size())) {
            throw std::out_of_range("PCG64_OpenMP_Manager: thread ID out of range");
        }
        persistent_states[tid] = s;
    }

    // =========================================================================
    // REENTRANT THREAD STREAM PROXY
    // =========================================================================
    class ThreadStream {
    private:
        PCG64_OpenMP_Manager& manager;

    public:
        using result_type = uint64_t;
        static constexpr result_type min() { return 0; }
        static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

        explicit ThreadStream(PCG64_OpenMP_Manager& mgr) noexcept : manager(mgr) {
            // ONLY load the state if this is the outermost proxy in the call stack
            if (guard_count == 0) {
                cached_tid = omp_get_thread_num();
                if (cached_tid >= static_cast<int>(manager.persistent_states.size())) {
                    #pragma omp critical(pcg64_openmp_resize)
                    {
                        if (cached_tid >= static_cast<int>(manager.persistent_states.size())) {
                            int old_size = manager.persistent_states.size();
                            manager.persistent_states.resize(cached_tid + 1);
                            for (int i = old_size; i <= cached_tid; ++i) {
                                manager.persistent_states[i].state = 1;
                                manager.persistent_states[i].inc = 2 * i + 1;
                                PCG64 tmp(manager.persistent_states[i].state, manager.persistent_states[i].inc);
                                tmp.warmup();
                                manager.persistent_states[i].state = tmp.getState();
                            }
                        }
                    }
                }
                const auto& state_data = manager.persistent_states[cached_tid];
                local_rng = PCG64(state_data.state, state_data.inc);
            }
            guard_count++;
        }

        ~ThreadStream() noexcept {
            guard_count--;
            // ONLY save the state if we are tearing down the outermost proxy
            if (guard_count == 0) {
                manager.persistent_states[cached_tid].state = local_rng.getState();
                manager.persistent_states[cached_tid].inc = local_rng.getIncrement();
            }
        }

        ThreadStream(const ThreadStream&) = delete;
        ThreadStream& operator=(const ThreadStream&) = delete;

        // --- Generator Interface (ZERO additional overhead) ---
        inline uint64_t next() noexcept { return local_rng.next(); }
        inline double nextDouble() noexcept { return local_rng.nextDouble(); }
        inline uint64_t operator()() noexcept { return local_rng.next(); }
    };

    ThreadStream stream() noexcept {
        return ThreadStream(*this);
    }
};

