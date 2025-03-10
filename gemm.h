#include <sched.h>
#include <unistd.h>

// Standard C++ headers
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <random>

// Required for Benchmark function
#include <chrono>
#include <functional>
#include <numeric>

// Check for NEON instruction set support
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define HAS_NEON 1
#endif

void Benchmark(const std::vector<int64_t> &dims, std::function<void()> func) {
  const int warmup_times = 10; // 增加预热次数
  const int infer_times = 50;  // 增加测试次数以提高准确性

  // 预热阶段
  for (int i = 0; i < warmup_times; ++i) {
    func();
  }

  // 执行测试
  std::vector<double> times;
  times.reserve(infer_times);

  for (int i = 0; i < infer_times; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    times.push_back(std::chrono::duration<double>(end - start).count());
  }

  // 计算统计数据
  std::sort(times.begin(), times.end());
  double total_time = std::accumulate(times.begin(), times.end(), 0.0);
  double avg_time = total_time / infer_times;
  double median_time = times[infer_times / 2];

  // 计算 GFLOPS
  double ops = 2.0 * std::accumulate(dims.begin(), dims.end(), 1LL,
                                     std::multiplies<int64_t>());
  double avg_gflops = (ops * 1.0e-9) / avg_time;
  double median_gflops = (ops * 1.0e-9) / median_time;

  // 美观输出
  std::cout << "┌─────────────────────────────────┐" << std::endl;
  std::cout << "│ 性能测试结果                    │" << std::endl;
  std::cout << "├─────────────────────────────────┤" << std::endl;
  std::cout << "│ 平均 GFLOPS: " << std::fixed << std::setprecision(2)
            << std::setw(16) << avg_gflops << " │" << std::endl;
  std::cout << "│ 中值 GFLOPS: " << std::fixed << std::setprecision(2)
            << std::setw(16) << median_gflops << " │" << std::endl;
  std::cout << "└─────────────────────────────────┘" << std::endl;
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

// CUDA GEMM function declaration
void gemm_cuda(const float* A, const float* B, float* C,
               int M, int N, int K);

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

// CUDA GEMM benchmark
template <int M, int N, int K>
void BenchmarkCUDA() {
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);
  std::vector<float> C(M * N);
  
  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0.0f, 1.0f);
  
  std::generate(A.begin(), A.end(), [&]() { return dis(gen); });
  std::generate(B.begin(), B.end(), [&]() { return dis(gen); });

  std::vector<int64_t> dims = {M, N, K};
  
  Benchmark(dims, [&]() {
    gemm_cuda(A.data(), B.data(), C.data(), M, N, K);
  });
}

#if HAS_NEON
inline void print_float32x4(const float32x4_t &vec) {
  alignas(16) float tmp[4];
  vst1q_f32(tmp, vec);
  std::cout << "[" << tmp[0] << ", " << tmp[1] << ", " << tmp[2] << ", "
            << tmp[3] << "]" << std::endl;
}
#endif
