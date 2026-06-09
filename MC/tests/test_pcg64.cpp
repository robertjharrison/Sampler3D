#include "pcg64.h"
#include <iostream>
#include <cassert>
#include <stdexcept>

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

int main() {
    std::cout << "Running PCG64 tests..." << std::endl;
    test_increment_validation();
    test_determinism();
    test_jump();
    std::cout << "PCG64 tests passed!" << std::endl;
    return 0;
}
