#include <stdio.h>
#include <cuda_runtime.h>

// CUDA计算能力版本到核心数的转换函数
int _ConvertSMVer2Cores(int major, int minor) {
    // 定义不同计算能力版本对应的核心数
    typedef struct {
        int compute;  // 计算能力版本
        int cores;     // 每个SM的核心数
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60,  64},
        {0x61, 128},
        {0x62, 128},
        {0x70,  64},
        {0x72,  64},
        {0x75,  64},
        {0x80,  64},
        {0x86, 128},
        {-1, -1}
    };

    int index = 0;
    while (nGpuArchCoresPerSM[index].compute != -1) {
        if (nGpuArchCoresPerSM[index].compute == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].cores;
        }
        index++;
    }
    return 128; // 默认值
}

// CUDA kernel函数：element-wise乘法
__global__ void elementwiseMul(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

// 使用CUDA进行性能测试
void cuda_flops_test() {
    // 获取设备显存信息
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    
    // 根据可用显存调整数据规模
    const size_t N = (freeMem / 3) / sizeof(float); // 使用1/3显存
    const size_t size = N * sizeof(float);
    
    printf("Using %.1f MB of GPU memory\n", size / (1024.0 * 1024.0));

    // 分配主机内存
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // 初始化数据
    for (size_t i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // 拷贝数据到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 设置线程配置
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    elementwiseMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // 增加测试次数以获得更准确的结果
    const int numRuns = 1000;
    
    // 预热
    for (int i = 0; i < 10; i++) {
        elementwiseMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    cudaDeviceSynchronize();

    // 执行kernel并计时
    cudaEventRecord(start);
    for (int i = 0; i < numRuns; i++) {
        elementwiseMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算性能
    double flops = numRuns * N; // 每次乘法1 FLOP
    double tflops = (flops * 1.0e-12f) / (milliseconds / 1000.0f);

    printf("Data size: %zu elements\n", N);
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f TFLOPS\n", tflops);

    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 获取CUDA设备信息并计算理论FLOPS
void get_device_flops() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    // 计算理论FLOPS
    int cores = prop.multiProcessorCount * 
               _ConvertSMVer2Cores(prop.major, prop.minor);
    float clockRateGHz = prop.clockRate * 1e-6f;
    float flops = cores * clockRateGHz * 2; // 2 FLOPS per cycle per core

    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Theoretical FLOPS: %.2f TFLOPS\n", flops / 1000);
}

int main() {
    // 获取设备信息
    get_device_flops();

    // 运行CUDA性能测试
    // cuda_flops_test();

    return 0;
}
