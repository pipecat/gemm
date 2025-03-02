#include "gemm.h"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void gemm_kernel(const float* A, const float* B, float* C, 
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gemm_cuda(const float* A, const float* B, float* C,
               int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float); 
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);
    gemm_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    // 定义矩阵维度
    const int M = 4096; // 行数
    const int N = 4096; // 列数
    const int K = 4096; // 内部维度
    std::vector<int64_t> dims = {M, N, K};

    // 初始化矩阵
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    std::vector<float> Ctrue(M * N);

    // 随机初始化矩阵
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto &elem : A) elem = dist(rng);
    for (auto &elem : B) elem = dist(rng);

    // 封装CUDA GEMM测试
    auto test_cuda_gemm = [&]() {
        gemm_cuda(A.data(), B.data(), C.data(), M, N, K);
    };

    // 运行基准测试
    std::cout << "CUDA GEMM" << std::endl;
    std::fill(C.begin(), C.end(), 0.0f);
    Benchmark(dims, test_cuda_gemm);

    // 验证结果
    // std::fill(Ctrue.begin(), Ctrue.end(), 0.0f);
    // naive_row_major_sgemm(A.data(), B.data(), Ctrue.data(), M, N, K);
    // Verify(C.data(), Ctrue.data(), M, N);

    return 0;
}
