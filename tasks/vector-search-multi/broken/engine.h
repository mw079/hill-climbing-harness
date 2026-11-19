#pragma once
#include <cstdint>
class Engine {
public:
    Engine(const float*, uint32_t, uint32_t) {}
    void search(const float*, uint32_t k, uint32_t* ids) {
        for (uint32_t i = 0; i < k; ++i) ids[i] = 0;
    }
};
