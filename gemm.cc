#include "gemm.h"
#include <Eigen/Dense>
#include <arm_neon.h>
#include <vector>

void naive_row_major_sgemm(const float *A, const float *B, float *C,
                           const int M, const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

void naive_mkn_sgemm(const float *A, const float *B, float *C, const int M,
                     const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}
void simd_mkn_sgemm(const float *A, const float *B, float *C, const int M,
                    const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; n += 4) {
        float32x4_t b_vec = vld1q_f32(B + k * N + n);
        float32x4_t c_vec = vld1q_f32(C + m * N + n);
        c_vec = vmlaq_n_f32(c_vec, b_vec, A[m * K + k]);
        vst1q_f32(C + m * N + n, c_vec);
        // C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

void transposed_B_sgemm(const float *A, const float *B, float *C, const int M,
                        const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0;
      for (int k = 0; k < K; k++) {
        acc += A[m * K + k] * B[n * K + k];
      }
      C[m * N + n] += acc;
    }
  }
}

void eigen_sgemm(const float *A, const float *B, float *C, const int M,
                 const int N, const int K) {
  Eigen::Map<const Eigen::MatrixXf> A_mat(A, M, K);
  Eigen::Map<const Eigen::MatrixXf> B_mat(B, K, N);
  Eigen::Map<Eigen::MatrixXf> C_mat(C, M, N);
  C_mat.noalias() = A_mat * B_mat;
}

void simd_transposed_B_sgemm(const float *A, const float *B, float *C,
                             const int M, const int N, const int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float32x4_t acc = vdupq_n_f32(0.0);
      for (int k = 0; k < K; k += 4) {
        float32x4_t a_vec = vld1q_f32(A + m * K + k);
        float32x4_t b_vec = vld1q_f32(B + n * K + k);
        acc = vmlaq_f32(acc, a_vec, b_vec);
      }
      C[m * N + n] += vaddvq_f32(acc);
    }
  }
}

int main() {
  // 定义矩阵维度
  const int M = 128; // 行数
  const int N = 128; // 列数
  const int K = 128; // 内部维度
  std::vector<int64_t> dims = {M, N, K};

  // 初始化矩阵
  std::vector<float> A(M * K);
  std::vector<float> At(K * M);
  std::vector<float> B(K * N);
  std::vector<float> Bt(N * K);
  std::vector<float> C(M * N);
  std::vector<float> Ct(N * M);

  // 随机初始化矩阵
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &elem : A)
    elem = dist(rng);
  for (auto &elem : B)
    elem = dist(rng);
  Bt = Transpose(B.data(), K, N);
  At = Transpose(A.data(), M, K);

  // 封装为 lambda 表达式
  auto test_naive_gemm = [&]() {
    naive_row_major_sgemm(A.data(), B.data(), C.data(), M, N, K);
  };
  auto test_mkn_gemm = [&]() {
    naive_mkn_sgemm(A.data(), B.data(), C.data(), M, N, K);
  };
  auto test_simd_mkn_gemm = [&]() {
    simd_mkn_sgemm(A.data(), B.data(), C.data(), M, N, K);
  };
  auto test_transposed_gemm = [&]() {
    transposed_B_sgemm(A.data(), Bt.data(), C.data(), M, N, K);
  };
  auto test_simd_transposed_gemm = [&]() {
    simd_transposed_B_sgemm(A.data(), Bt.data(), C.data(), M, N, K);
  };
  auto test_eigen_gemm = [&]() {
    eigen_sgemm(At.data(), Bt.data(), C.data(), M, N, K);
  };

  // 运行基准测试
  std::cout << "Naive GEMM";
  std::fill(C.begin(), C.end(), 0.0f);
  Benchmark(dims, test_naive_gemm);
  std::vector<float> Ctrue(C);
  std::cout << "MKN GEMM";
  std::fill(C.begin(), C.end(), 0.0f);
  Benchmark(dims, test_mkn_gemm);
  //   Verify(C.data(), Ctrue.data(), M, N);
  std::cout << "SIMD MKN GEMM";
  std::fill(C.begin(), C.end(), 0.0f);
  Benchmark(dims, test_simd_mkn_gemm);
  //   Verify(C.data(), Ctrue.data(), M, N);
  std::cout << "Transposed GEMM";
  std::fill(C.begin(), C.end(), 0.0f);
  Benchmark(dims, test_transposed_gemm);
  //   Verify(C.data(), Ctrue.data(), M, N);
  std::cout << "SIMD transposed GEMM";
  std::fill(C.begin(), C.end(), 0.0f);
  Benchmark(dims, test_simd_transposed_gemm);
  //   Verify(C.data(), Ctrue.data(), M, N);
  std::cout << "Eigen GEMM";
  std::fill(C.begin(), C.end(), 0.0f);
  Benchmark(dims, test_eigen_gemm);
  Ct = Transpose(C.data(), N, M);
  //   Verify(Ct.data(), Ctrue.data(), M, N);

  return 0;
}