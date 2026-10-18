// bw.cpp: usage: bw <read|copy> <cpu,cpu,...> [MB per thread=64] [iters=20]
#include <arm_neon.h>
#include <sched.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static float read_kernel(const float* p, size_t n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < n; i += 16) {
        a0 = vaddq_f32(a0, vld1q_f32(p + i));
        a1 = vaddq_f32(a1, vld1q_f32(p + i + 4));
        a2 = vaddq_f32(a2, vld1q_f32(p + i + 8));
        a3 = vaddq_f32(a3, vld1q_f32(p + i + 12));
    }
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

int main(int argc, char** argv) {
    if (argc < 3) { puts("usage: bw <read|copy> <cpus,comma-sep> [MB=256] [iters=20]"); return 1; }
    bool is_read = !strcmp(argv[1], "read");
    std::vector<int> cpus;
    { std::stringstream ss(argv[2]); std::string t; while (std::getline(ss, t, ',')) cpus.push_back(atoi(t.c_str())); }
    size_t mb = argc > 3 ? atoi(argv[3]) : 64;
    int iters = argc > 4 ? atoi(argv[4]) : 20;
    size_t bytes = mb << 20, n = bytes / sizeof(float);

    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> th;
    std::vector<float> sink(cpus.size());

    for (size_t t = 0; t < cpus.size(); t++) {
        th.emplace_back([&, t] {
            pin(cpus[t]);
            float* a = (float*)aligned_alloc(64, bytes);
            float* b = is_read ? nullptr : (float*)aligned_alloc(64, bytes);
            memset(a, 1, bytes);
            if (b) memset(b, 2, bytes);
            ready++;
            while (!go) {}
            float acc = 0;
            for (int i = 0; i < iters; i++) {
                if (is_read) acc += read_kernel(a, n);
                else memcpy(b, a, bytes);
            }
            sink[t] = acc;
        });
    }
    while (ready < (int)cpus.size()) {}
    auto t0 = std::chrono::steady_clock::now();
    go = true;
    for (auto& x : th) x.join();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double moved = (double)bytes * iters * cpus.size() * (is_read ? 1 : 2);  // copy = read + write
    printf("%s cpus=%s  %.2f GB/s  (%.2fs)\n", argv[1], argv[2], moved / sec / 1e9, sec);
    return 0;
}