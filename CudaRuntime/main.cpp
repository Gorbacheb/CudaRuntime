#include <string>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <chrono>
#include "kernel.cuh"

using namespace std;
using ll = long long;

void print_matrix(ll n, float* mx, string name) {
    cout << name << "\n";
    cout << "[\n";
    for (int i = 0; i < n; i++) {
        cout << "   [";
        for (int j = 0; j < n; j++) {
            cout << mx[i * n + j];
            if (j != n - 1) {
                cout << ", ";
            }
        }
        if (i != n - 1) {
            cout << "], ";
        }
        else {
            cout << "]";
        }
        cout << "\n";
    }
    cout << "]\n";
}

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);
std::uniform_int_distribution<int> distribution(1, 100);


void solve(ll n, bool useTiled) {
    float* h_A = (float*)malloc(n * n * sizeof(float));
    float* h_B = (float*)malloc(n * n * sizeof(float));
    float* h_C = (float*)malloc(n * n * sizeof(float));

    for (ll i = 0; i < n * n; i++) {
        h_A[i] = distribution(generator) % 100 / 10.0f;
        h_B[i] = distribution(generator) % 100 / 10.0f;
    }

#ifdef _DEBUG
    print_matrix(n, h_A, "A");
    print_matrix(n, h_B, "B");
#endif 

    clock_t start, end;
    start = clock();

    matrixMultiplyCUDA(h_A, h_B, h_C, n, useTiled);

    end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

#ifdef _DEBUG
    print_matrix(n, h_C, "C");
#endif 

    string name = "output.csv";
    ofstream csvFile(name, std::ios::app);
    if (csvFile.is_open()) {

        csvFile << n << "," << useTiled << "," << time_taken << "\n";

        csvFile.close();
    }
    else
    {
        cout << "Error during writing\n";
    }

    printf("n = %d, %f\n", n, time_taken);

    //для дебага
    //for (int i = 0; i < 5; i++) {
    //    for (int j = 0; j < 5; j++) {
    //        std::cout << h_C[i * n + j] << " ";
    //    }
    //    std::cout << std::endl;
    //}

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    free(h_A);
    free(h_B);
    free(h_C);
}

void checkCudaCapabilities() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        cout << "Device " << i << ": " << prop.name << "\n";
        cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        cout << "  Max Block Dimensions: ("
            << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << ")\n";
        cout << "  Max Grid Dimensions: ("
            << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << ")\n";
        cout << "  Multiprocessor count: " << prop.multiProcessorCount << "\n";
        cout << "  Total Global Memory: " << prop.totalGlobalMem << " bytes\n";
    }
}


int main()
{
    checkCudaCapabilities();
    ll start = 5000;
    ll end = 100000;

    int op_type = -1;

    cout << "Please, input calc type:\n";
    cout << "\tSimple: 1\n\tTilted (optimized): 2 or any simbol\n";
    cin >> op_type;
    solve(40000, op_type != 1);

    

    
    //cout.precision(12);
    //for (ll i = start; i < end; i += 1000) {
    //    solve(i);
    //}
}