#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>

double arm_int8_computing() {
  const int64_t N = 1000000000;
  asm volatile("mov x0, #1 \n"
               "dup v0.16b, w0 \n"
               "dup v1.16b, w0 \n"
               "dup v2.16b, w0 \n"
               "dup v3.16b, w0 \n"
               "dup v4.16b, w0 \n"
               "dup v5.16b, w0 \n"
               "dup v6.16b, w0 \n"
               "dup v7.16b, w0 \n"
               "dup v8.16b, w0 \n"
               "dup v9.16b, w0 \n"
               "dup v10.16b, w0 \n"
               "dup v11.16b, w0 \n"
               "dup v12.16b, w0 \n"
               "dup v13.16b, w0 \n"
               "dup v14.16b, w0 \n"
               "dup v15.16b, w0 \n");

  auto start = std::chrono::high_resolution_clock::now();

  for (int64_t i = 0; i < N; i += 16) {
    asm volatile("mla v0.16b, v0.16b, v1.16b \n"
                 "mla v1.16b, v1.16b, v2.16b \n"
                 "mla v2.16b, v2.16b, v3.16b \n"
                 "mla v3.16b, v3.16b, v4.16b \n"
                 "mla v4.16b, v4.16b, v5.16b \n"
                 "mla v5.16b, v5.16b, v6.16b \n"
                 "mla v6.16b, v6.16b, v7.16b \n"
                 "mla v7.16b, v7.16b, v8.16b \n"
                 "mla v8.16b, v8.16b, v9.16b \n"
                 "mla v9.16b, v9.16b, v10.16b \n"
                 "mla v10.16b, v10.16b, v11.16b \n"
                 "mla v11.16b, v11.16b, v12.16b \n"
                 "mla v12.16b, v12.16b, v13.16b \n"
                 "mla v13.16b, v13.16b, v14.16b \n"
                 "mla v14.16b, v14.16b, v15.16b \n"
                 "mla v15.16b, v15.16b, v0.16b \n" ::
                     : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                       "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  double ops = (N / 16) * // 循环次数 (1e9/16)
               16 *       // 每次循环16条MLA指令
               16 *       // 每条指令处理16个int8元素
               2;         // 每个元素2个操作（乘加）
  double seconds = duration.count() * 1e-9;
  return ops / seconds * 1e-9;
}

double arm_fp16_computing() {
  const int64_t N = 1000000000;
  asm volatile("mov x0, #1 \n"
               "dup v0.8h, w0 \n"
               "dup v1.8h, w0 \n"
               "dup v2.8h, w0 \n"
               "dup v3.8h, w0 \n"
               "dup v4.8h, w0 \n"
               "dup v5.8h, w0 \n"
               "dup v6.8h, w0 \n"
               "dup v7.8h, w0 \n"
               "dup v8.8h, w0 \n"
               "dup v9.8h, w0 \n"
               "dup v10.8h, w0 \n"
               "dup v11.8h, w0 \n"
               "dup v12.8h, w0 \n"
               "dup v13.8h, w0 \n"
               "dup v14.8h, w0 \n"
               "dup v15.8h, w0 \n");

  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < N; i += 16) {
    asm volatile("fmla v0.8h, v0.8h, v1.8h \n"
                 "fmla v1.8h, v1.8h, v2.8h \n"
                 "fmla v2.8h, v2.8h, v3.8h \n"
                 "fmla v3.8h, v3.8h, v4.8h \n"
                 "fmla v4.8h, v4.8h, v5.8h \n"
                 "fmla v5.8h, v5.8h, v6.8h \n"
                 "fmla v6.8h, v6.8h, v7.8h \n"
                 "fmla v7.8h, v7.8h, v8.8h \n"
                 "fmla v8.8h, v8.8h, v9.8h \n"
                 "fmla v9.8h, v9.8h, v10.8h \n"
                 "fmla v10.8h, v10.8h, v11.8h \n"
                 "fmla v11.8h, v11.8h, v12.8h \n"
                 "fmla v12.8h, v12.8h, v13.8h \n"
                 "fmla v13.8h, v13.8h, v14.8h \n"
                 "fmla v14.8h, v14.8h, v15.8h \n"
                 "fmla v15.8h, v15.8h, v0.8h \n" ::
                     : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                       "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  double ops = (N / 16) * // 循环次数
               16 *       // 每次循环16条FMLA指令
               8 *        // 每条指令处理8个fp16元素
               2;         // 每个元素2个FLOPs
  double seconds = duration.count() * 1e-9;
  return ops / seconds * 1e-9;
}

double arm_fp32_computing() {
  const int64_t N = 1000000000;
  asm volatile("mov x0, #1 \n"
               "dup v0.4s, w0 \n"
               "dup v1.4s, w0 \n"
               "dup v2.4s, w0 \n"
               "dup v3.4s, w0 \n"
               "dup v4.4s, w0 \n"
               "dup v5.4s, w0 \n"
               "dup v6.4s, w0 \n"
               "dup v7.4s, w0 \n"
               "dup v8.4s, w0 \n"
               "dup v9.4s, w0 \n"
               "dup v10.4s, w0 \n"
               "dup v11.4s, w0 \n"
               "dup v12.4s, w0 \n"
               "dup v13.4s, w0 \n"
               "dup v14.4s, w0 \n"
               "dup v15.4s, w0 \n");

  auto start = std::chrono::high_resolution_clock::now();
  for (int64_t i = 0; i < N; i += 16) {
    asm volatile("fmla v0.4s, v0.4s, v1.4s \n"
                 "fmla v1.4s, v1.4s, v2.4s \n"
                 "fmla v2.4s, v2.4s, v3.4s \n"
                 "fmla v3.4s, v3.4s, v4.4s \n"
                 "fmla v4.4s, v4.4s, v5.4s \n"
                 "fmla v5.4s, v5.4s, v6.4s \n"
                 "fmla v6.4s, v6.4s, v7.4s \n"
                 "fmla v7.4s, v7.4s, v8.4s \n"
                 "fmla v8.4s, v8.4s, v9.4s \n"
                 "fmla v9.4s, v9.4s, v10.4s \n"
                 "fmla v10.4s, v10.4s, v11.4s \n"
                 "fmla v11.4s, v11.4s, v12.4s \n"
                 "fmla v12.4s, v12.4s, v13.4s \n"
                 "fmla v13.4s, v13.4s, v14.4s \n"
                 "fmla v14.4s, v14.4s, v15.4s \n"
                 "fmla v15.4s, v15.4s, v0.4s \n" ::
                     : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8",
                       "v9", "v10", "v11", "v12", "v13", "v14", "v15");
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  double ops = (N / 16) * // 循环次数
               16 *       // 每次循环16条FMLA指令
               4 *        // 每条指令处理4个fp32元素
               2;         // 每个元素2个FLOPs
  double seconds = duration.count() * 1e-9;
  return ops / seconds * 1e-9;
}

int main() {
  // 运行测试
  printf("Testing CPU peak performance...\n");
  printf("arm_int8_computing: %.2f GOPS\n", arm_int8_computing());
  printf("arm_fp16_computing: %.2f GFLOPS\n", arm_fp16_computing());
  printf("arm_fp32_computing: %.2f GFLOPS\n", arm_fp32_computing());

  return 0;
}

#endif
