#pragma once
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_factory.h>
#include <omp.h>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include "frozen.h"

// Hidden reference only. The candidate environment never receives this header.
class Engine {
    std::unique_ptr<faiss::Index> index;
public:
    Engine(const float* base, uint32_t n, uint32_t d) {
        omp_set_num_threads(1);
        std::string factory = HC_FACTORY;
        int parameter = HC_SEARCH_PARAMETER;
#ifdef HC_TUNING
        if (const char* p = std::getenv("HC_TUNE_FACTORY")) factory = p;
        if (const char* p = std::getenv("HC_TUNE_PARAMETER")) parameter = std::atoi(p);
#endif
        index.reset(faiss::index_factory(d, factory.c_str(), faiss::METRIC_L2));
        if (auto* h = dynamic_cast<faiss::IndexHNSW*>(index.get())) {
            h->hnsw.efConstruction = 100;
            h->hnsw.efSearch = parameter;
        }
        if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get())) ivf->nprobe = parameter;
        index->train(n, base);
        index->add(n, base);
    }
    void search(const float* query, uint32_t k, uint32_t* ids) {
        std::array<faiss::idx_t, 10> labels;
        std::array<float, 10> distances;
        index->search(1, query, k, distances.data(), labels.data());
        for (uint32_t i = 0; i < k; ++i) ids[i] = static_cast<uint32_t>(labels[i]);
    }
};
