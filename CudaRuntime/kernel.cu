#include <string>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <random>
#include <chrono>
#include "kernel.cuh"

#define TILE_WIDTH 32

using namespace std;
using ll = long long;

// Простая версия
__global__ void matrixMultiplyKernelSimple(float* d_A, float* d_B, float* d_C, ll n) {
    ll row = blockIdx.y * blockDim.y + threadIdx.y;
    ll col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float value = 0;
        for (ll k = 0; k < n; k++) {
            value += d_A[row * n + k] * d_B[k * n + col];
        }
        d_C[row * n + col] = value;
    }
}

// Оптимизированная tiled-версия
__global__ void matrixMultiplyKernelTiled(float* d_A, float* d_B, float* d_C, ll n) {
    __shared__ float A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B[TILE_WIDTH][TILE_WIDTH];

    ll row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    ll col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0;

    for (ll t = 0; t < (n + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < n && t * TILE_WIDTH + threadIdx.x < n)
            A[threadIdx.y][threadIdx.x] = d_A[row * n + t * TILE_WIDTH + threadIdx.x];
        else
            A[threadIdx.y][threadIdx.x] = 0;

        if (col < n && t * TILE_WIDTH + threadIdx.y < n)
            B[threadIdx.y][threadIdx.x] = d_B[(t * TILE_WIDTH + threadIdx.y) * n + col];
        else
            B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for (ll k = 0; k < TILE_WIDTH; k++)
            value += A[threadIdx.y][k] * B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < n && col < n)
        d_C[row * n + col] = value;
}

// Выбор ядра и выполнение
void matrixMultiplyCUDA(float* h_A, float* h_B, float* h_C, ll n, bool useTiled) {
    float* d_A, * d_B, * d_C;
    ll size = n * n * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    if (useTiled) {
        matrixMultiplyKernelTiled << <dimGrid, dimBlock >> > (d_A, d_B, d_C, n);
    }
    else {
        matrixMultiplyKernelSimple << <dimGrid, dimBlock >> > (d_A, d_B, d_C, n);
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}




