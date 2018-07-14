#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static inline void work(uint64_t N, int16_t * __restrict out1,
        int16_t * __restrict out2, int16_t * __restrict out3, 
        int16_t * __restrict in1, int16_t * __restrict in2,
        int16_t * __restrict in3, int16_t * __restrict in4,
        int16_t * __restrict in5, int16_t * __restrict in6) {
                    
    for (int64_t i = 0; i < N; i++) {
        int16_t e0 = in1[i];
        int16_t e1 = in2[i];
        int16_t e2 = in3[i];
        int16_t e3 = in4[i];
        int16_t e4 = in5[i];
        int16_t e5 = in6[i];
        out1[i] = e0 * e1 * e2 * e3 * e4;
        out2[i] = e1 * e2 * e3 * e4 * e5;
        out3[i] = e0 * e5 + e1 * e4;
    }
}

static inline uint64_t get_cycles() {
  uint64_t a, d;
  __asm volatile("rdtsc" : "=a" (a), "=d" (d));
  return a | (d<<32);
}

int32_t main(int argc, char** argv) {
    uint64_t iters = atoi(argv[1]), N = atoi(argv[2]);
    int16_t *in1 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *in2 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *in3 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *in4 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *in5 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *in6 = (int16_t*)malloc(sizeof(int16_t) * N);
    int32_t res = 0;

    for (int32_t i = 0; i < N; i++) {
        in1[i] = rand() % 42;
        in2[i] = rand() % 42;
        in3[i] = rand() % 42;
        in4[i] = rand() % 42;
        in5[i] = rand() % 42;
        in6[i] = rand() % 42;
    }

    int16_t *out1 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *out2 = (int16_t*)malloc(sizeof(int16_t) * N);
    int16_t *out3 = (int16_t*)malloc(sizeof(int16_t) * N);

    uint64_t total_cycles = 0;
    for (int32_t i = 0; i < iters; i++) {
        uint64_t start = get_cycles();
        work(N, out1, out2, out3, in1, in2, in3, in4, in5, in6);
        total_cycles += (get_cycles() - start);

        // Just making sure the work is not optimized away
        int32_t idx = rand() % N; 
        res += out1[idx] + out2[idx] + out3[idx];
    }

    fprintf(stdout, "%.3g\n", ((double) total_cycles) / (iters*N));

    return res;
}