#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    int32_t a;
    int32_t b;
    int32_t c;
} tuple_t;

typedef struct {
    tuple_t *data;
    uint64_t size;
    uint64_t capacity;
} vector_t;

vector_t *new_vector() {
    vector_t *result = (vector_t *)malloc(sizeof(vector_t));
    result->data = (tuple_t *)malloc(sizeof(tuple_t) * 2);
    result->size = 0;
    result->capacity = 2;
    return result;
}

void work(uint64_t N, vector_t * __restrict out, 
        int32_t * __restrict in1, int32_t * __restrict in2,
        int32_t * __restrict in3, int32_t * __restrict in4,
        int32_t * __restrict in5, int32_t * __restrict in6) {

    for (int64_t i = 0; i < N; i++) {
        if (in1[i] == 42) {
            if (out->size == out->capacity) {
                out->capacity *= 2;
                out->data = (tuple_t *)realloc(out->data, out->capacity * sizeof(tuple_t));
            }
            int32_t e0 = in1[i];
            int32_t e1 = in2[i];
            int32_t e2 = in3[i];
            int32_t e3 = in4[i];
            int32_t e4 = in5[i];
            int32_t e5 = in6[i];
            out->data[out->size].a = e0 * e1 * e2 * e3 * e4;
            out->data[out->size].b = e1 * e2 * e3 * e4 * e5;
            out->data[out->size].c = e0 * e5 + e1 * e4;
            out->size++;
        }
    }
}

static inline uint64_t get_cycles() {
  uint64_t a, d;
  __asm volatile("rdtsc" : "=a" (a), "=d" (d));
  return a | (d<<32);
}

int32_t main(int argc, char** argv) {
    uint64_t iters = atoi(argv[1]), N = atoi(argv[2]);
    int32_t *in1 = (int32_t*)malloc(sizeof(int32_t) * N);
    int32_t *in2 = (int32_t*)malloc(sizeof(int32_t) * N);
    int32_t *in3 = (int32_t*)malloc(sizeof(int32_t) * N);
    int32_t *in4 = (int32_t*)malloc(sizeof(int32_t) * N);
    int32_t *in5 = (int32_t*)malloc(sizeof(int32_t) * N);
    int32_t *in6 = (int32_t*)malloc(sizeof(int32_t) * N);
    uint64_t res = 0;

    for (int32_t i = 0; i < N; i++) {
        in1[i] = 42;
        in2[i] = rand() % 42;
        in3[i] = rand() % 42;
        in4[i] = rand() % 42;
        in5[i] = rand() % 42;
        in6[i] = rand() % 42;
    }

    uint64_t total_cycles = 0;
    int32_t bs = 0;
    for (int32_t i = 0; i < iters; i++) {
        vector_t *out = new_vector();

        uint64_t start = get_cycles();
        work(N, out, in1, in2, in3, in4, in5, in6);
        total_cycles += (get_cycles() - start);

        int32_t bs_idx = rand() % N; 
        bs += out->data[bs_idx].a + out->data[bs_idx].b + out->data[bs_idx].c;
        free(out);
    }

    fprintf(stdout, "%.3g\n", ((double) total_cycles) / (iters*N));

    return bs;
}