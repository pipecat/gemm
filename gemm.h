#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

void Benchmark(const std::vector<int64_t> &dims, std::function<void()> func) {
  const int warmup_times = 5;
  const int infer_times = 20;

  // warmup
  for (int i = 0; i < warmup_times; ++i)
    func();

  // run
  auto start_time = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < infer_times; ++i)
    func();

  // latency
  auto end_time = std::chrono::high_resolution_clock::now();
  double dtime = std::chrono::duration<double>(end_time - start_time).count();

  // compute GLOPs
  double flops = 2.0 *
                 std::accumulate(dims.begin(), dims.end(), 1LL,
                                 std::multiplies<int64_t>()) *
                 1.0e-09;
  flops = flops * infer_times / dtime;

  // print
  std::cout << " GFLOPs: " << flops << std::endl;
}

void Verify(const float *A, const float *B, const int M, const int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      if (std::abs(A[m * N + n] - B[m * N + n]) > 1e-3) {
        std::cout << "Verify failed at " << m << " " << n << std::endl;
        std::cout << "A: " << A[m * N + n] << std::endl;
        std::cout << "B: " << B[m * N + n] << std::endl;
      }
    }
  }
}

void PrintMatrix(const float *A, const int M, const int N) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      std::cout << A[m * N + n] << " ";
    }
    std::cout << std::endl;
  }
}

std::vector<float> Transpose(const float *A, const int M, const int N) {
  std::vector<float> B(N * M);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      B[n * M + m] = A[m * N + n];
    }
  }
  return B;
}

inline void print_float32x4(const float32x4_t &vec) {
  alignas(16) float tmp[4];
  vst1q_f32(tmp, vec);
  std::cout << "[" << tmp[0] << ", " << tmp[1] << ", " << tmp[2] << ", "
            << tmp[3] << "]" << std::endl;
}