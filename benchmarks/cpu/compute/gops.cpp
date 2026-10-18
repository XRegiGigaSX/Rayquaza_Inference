// gops.cpp: usage: gops <sdot|smmla|fp32|fp16> <cpus,comma-sep> [iters=200000000]
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

#define NACC 8   // independent accumulators to hide instruction latency

static void pin(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    sched_setaffinity(0, sizeof(s), &s);
}

static int32_t k_sdot(long iters) {
    int8x16_t a = vdupq_n_s8(1), b = vdupq_n_s8(2);
    int32x4_t c[NACC]; for (auto& x : c) x = vdupq_n_s32(0);
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < NACC; i++) c[i] = vdotq_s32(c[i], a, b);
    int32x4_t s = c[0]; for (int i = 1; i < NACC; i++) s = vaddq_s32(s, c[i]);
    return vaddvq_s32(s);
}
static int32_t k_smmla(long iters) {
    int8x16_t a = vdupq_n_s8(1), b = vdupq_n_s8(2);
    int32x4_t c[NACC]; for (auto& x : c) x = vdupq_n_s32(0);
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < NACC; i++) c[i] = vmmlaq_s32(c[i], a, b);
    int32x4_t s = c[0]; for (int i = 1; i < NACC; i++) s = vaddq_s32(s, c[i]);
    return vaddvq_s32(s);
}
static float k_fp32(long iters) {
    float32x4_t a = vdupq_n_f32(1.0001f), b = vdupq_n_f32(0.9999f);
    float32x4_t c[NACC]; for (auto& x : c) x = vdupq_n_f32(0);
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < NACC; i++) c[i] = vfmaq_f32(c[i], a, b);
    float32x4_t s = c[0]; for (int i = 1; i < NACC; i++) s = vaddq_f32(s, c[i]);
    return vaddvq_f32(s);
}
static float k_fp16(long iters) {
    float16x8_t a = vdupq_n_f16(1.0f), b = vdupq_n_f16(1.0f);
    float16x8_t c[NACC]; for (auto& x : c) x = vdupq_n_f16(0);
    for (long it = 0; it < iters; it++)
        for (int i = 0; i < NACC; i++) c[i] = vfmaq_f16(c[i], a, b);
    float16x8_t s = c[0]; for (int i = 1; i < NACC; i++) s = vaddq_f16(s, c[i]);
    return (float)vgetq_lane_f16(s, 0);
}

int main(int argc, char** argv) {
    if (argc < 3) { puts("usage: gops <sdot|smmla|fp32|fp16> <cpus> [iters]"); return 1; }
    std::string kind = argv[1];
    std::vector<int> cpus;
    { std::stringstream ss(argv[2]); std::string t; while (std::getline(ss, t, ',')) cpus.push_back(atoi(t.c_str())); }
    long iters = argc > 3 ? atol(argv[3]) : 200000000L;

    // ops per instruction (1 MAC = 2 ops)
    double ops_per_instr = kind == "sdot" ? 32 : kind == "smmla" ? 64 : kind == "fp32" ? 8 : 16;

    std::atomic<int> ready{0}; std::atomic<bool> go{false};
    std::vector<std::thread> th; std::vector<double> sink(cpus.size());
    for (size_t t = 0; t < cpus.size(); t++) th.emplace_back([&, t] {
            pin(cpus[t]); ready++; while (!go) {}
            if (kind == "sdot") sink[t] = k_sdot(iters);
            else if (kind == "smmla") sink[t] = k_smmla(iters);
            else if (kind == "fp32") sink[t] = k_fp32(iters);
            else sink[t] = k_fp16(iters);
        });
    while (ready < (int)cpus.size()) {}
    auto t0 = std::chrono::steady_clock::now(); go = true;
    for (auto& x : th) x.join();
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double total_ops = ops_per_instr * NACC * (double)iters * cpus.size();
    double minstr_per_core = (double)NACC * iters / sec / 1e9;
    printf("%s cpus=%s  %.1f GOPS total  (%.2f G instr/s per core)  %.2fs\n",
           argv[1], argv[2], total_ops / sec / 1e9, minstr_per_core, sec);
    return 0;
}