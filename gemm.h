#include <sched.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
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