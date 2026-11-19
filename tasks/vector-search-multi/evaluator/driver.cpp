// Evaluator-owned I/O, warmup, timing, repeatability, and resource accounting.
// No search algorithm is supplied. A copy is included in the agent workspace.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}
struct Matrix {
    uint32_t n = 0, d = 0;
    std::vector<float> values;
    explicit Matrix(const char* file) {
        std::ifstream in(file, std::ios::binary);
        in.read(reinterpret_cast<char*>(&n), 4);
        in.read(reinterpret_cast<char*>(&d), 4);
        if (!in || n < 10 || n > 1000000 || !d || d > 4096)
            throw std::runtime_error("invalid input shape");
        values.resize(size_t(n) * d);
        in.read(reinterpret_cast<char*>(values.data()), values.size() * 4);
        if (!in || in.peek() != EOF) throw std::runtime_error("invalid input size");
        for (float x : values) if (!std::isfinite(x)) throw std::runtime_error("non-finite input");
    }
};
static double peak_mb() {
    rusage r{};
    getrusage(RUSAGE_SELF, &r);
#ifdef __APPLE__
    return double(r.ru_maxrss) / (1024 * 1024);
#else
    return double(r.ru_maxrss) / 1024;
#endif
}
static void single_thread() {
#ifdef __APPLE__
    thread_act_array_t threads;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS)
        throw std::runtime_error("thread check failed");
    for (mach_msg_type_number_t i = 0; i < count; ++i) mach_port_deallocate(mach_task_self(), threads[i]);
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), count * sizeof(thread_t));
    if (count != 1) throw std::runtime_error("multiple threads are forbidden");
#endif
}

#include "engine.h"

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: engine base.bin queries.bin ids.bin metrics.json min_seconds build_limit rss_limit");
        rlimit cpu{40, 40};
        setrlimit(RLIMIT_CPU, &cpu);
        Matrix base(argv[1]);
        const double min_seconds = std::stod(argv[5]);
        const double build_limit = std::stod(argv[6]);
        const double rss_limit = std::stod(argv[7]);
        auto t = Clock::now();
        Engine index(base.values.data(), base.n, base.d);
        double build_s = seconds(t);
        if (build_s > build_limit) throw std::runtime_error("build time limit exceeded");
        single_thread();
        if (peak_mb() > rss_limit) throw std::runtime_error("memory limit exceeded");
        // Queries are loaded only after the index has been built.
        Matrix queries(argv[2]);
        if (queries.d != base.d) throw std::runtime_error("dimension mismatch");
        constexpr uint32_t k = 10;
        std::vector<uint32_t> warm(size_t(queries.n) * k);
        std::vector<uint32_t> result(k);
        for (uint32_t i = 0; i < queries.n; ++i) {
            index.search(queries.values.data() + size_t(i) * base.d, k, warm.data() + size_t(i) * k);
        }
        // Every timed answer must match warmup; returning garbage only during timing
        // does not pass. Independent recall is computed by the Python evaluator.
        uint64_t passes = 0;
        t = Clock::now();
        do {
            for (uint32_t i = 0; i < queries.n; ++i) {
                std::fill(result.begin(), result.end(), UINT32_MAX);
                index.search(queries.values.data() + size_t(i) * base.d, k, result.data());
                if (std::memcmp(result.data(), warm.data() + size_t(i) * k, k * 4))
                    throw std::runtime_error("non-deterministic search result");
            }
            ++passes;
        } while (seconds(t) < min_seconds);
        const double search_s = seconds(t);
        single_thread();
        double rss = peak_mb();
        if (rss > rss_limit) throw std::runtime_error("memory limit exceeded");
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char*>(warm.data()), warm.size() * 4);
        std::ofstream metrics(argv[4]);
        metrics << std::setprecision(17) << "{\"build_seconds\":" << build_s
                << ",\"search_seconds\":" << search_s << ",\"passes\":" << passes
                << ",\"queries\":" << queries.n << ",\"peak_rss_mb\":" << rss << "}\n";
        if (!output || !metrics) throw std::runtime_error("output write failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
