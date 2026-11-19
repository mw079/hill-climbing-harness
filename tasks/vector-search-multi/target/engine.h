#pragma once
#include <cstdint>
#include <stdexcept>

// Implement the index and query algorithm here. The base pointer remains valid
// for the lifetime of Engine. No starting search implementation is supplied.
class Engine {
public:
    Engine(const float* base, uint32_t n, uint32_t dimension) {
        throw std::runtime_error("not implemented");
    }
    void search(const float* query, uint32_t k, uint32_t* ids) {
        throw std::runtime_error("not implemented");
    }
};
