#include <arm_neon.h>
#include <stdio.h>
namespace mmult {
/* Block sizes */
#define mc 256
#define kc 128

/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

/* Routine for computing C = A * B + C */

void PackMatrixB(int k, float *b, int ldb, float *b_to) {
  int j;
  for (j = 0; j < k; ++j) {
    float *b_ij_pntr = &B(j, 0);
    *b_to++ = b_ij_pntr[0];
    *b_to++ = b_ij_pntr[1];
    *b_to++ = b_ij_pntr[2];
    *b_to++ = b_ij_pntr[3];
  }
}

void PackMatrixA(int k, float *a, int lda, float *a_to) {
  int i;
  float *a_0i_pntr = a, *a_1i_pntr = a + lda, *a_2i_pntr = a + (lda << 1),
        *a_3i_pntr = a + (3 * lda);

  for (i = 0; i < k; ++i) {
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
  }
}

void AddDot4x4(int k, float *a, int lda, float *b, int ldb, float *c, int ldc) {
  float32x4_t c_0p_sum = {0};
  float32x4_t c_1p_sum = {0};
  float32x4_t c_2p_sum = {0};
  float32x4_t c_3p_sum = {0};

  register float a_0p_reg, a_1p_reg, a_2p_reg, a_3p_reg;

  for (int p = 0; p < k; ++p) {
    float32x4_t b_reg = vld1q_f32(b);
    b += 4;

    a_0p_reg = a[0];
    a_1p_reg = a[1];
    a_2p_reg = a[2];
    a_3p_reg = a[3];
    a += 4;

    c_0p_sum = vmlaq_n_f32(c_0p_sum, b_reg, a_0p_reg);
    c_1p_sum = vmlaq_n_f32(c_1p_sum, b_reg, a_1p_reg);
    c_2p_sum = vmlaq_n_f32(c_2p_sum, b_reg, a_2p_reg);
    c_3p_sum = vmlaq_n_f32(c_3p_sum, b_reg, a_3p_reg);
  }

  float *c_pntr = 0;
  c_pntr = &C(0, 0);
  float32x4_t c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_0p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(1, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_1p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(2, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_2p_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(3, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_3p_sum);
  vst1q_f32(c_pntr, c_reg);
}

void InnerKernel(int m, int n, int k, float *a, int lda, float *b, int ldb,
                 float *c, int ldc) {
  int i, j;
  float packedA[m * k];
  float packedB[k * n];

  for (j = 0; j < n; j += 4) { /* Loop over the columns of C, unrolled by 4 */
    PackMatrixB(k, &B(0, j), ldb, packedB + j * k);
    for (i = 0; i < m; i += 4) { /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
          one routine (four inner products) */
      if (0 == j) {
        PackMatrixA(k, &A(i, 0), lda, packedA + i * k);
      }
      AddDot4x4(k, packedA + i * k, k, packedB + j * k, 4, &C(i, j), ldc);
    }
  }
}

void MY_MMult_4x4_13(int m, int n, int k, float *a, int lda, float *b, int ldb,
                     float *c, int ldc) {
  int i, p, pb, ib;
  for (p = 0; p < k; p += kc) {
    pb = min(k - p, kc);
    for (i = 0; i < m; i += mc) {
      ib = min(m - i, mc);
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
    }
  }
}

// 在头文件末尾取消宏定义
#undef A
#undef B
#undef C

#undef mc
#undef kc
} // namespace mmult
