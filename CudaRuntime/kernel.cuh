#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

using ll = long long;

#ifdef __cplusplus
extern "C" {
#endif

	void matrixMultiplyCUDA(float* h_A, float* h_B, float* h_C, ll n, bool useTiled);

#ifdef __cplusplus
}
#endif

#endif  // MATRIX_MUL_H
