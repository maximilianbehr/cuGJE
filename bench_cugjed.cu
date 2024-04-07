/* MIT License
 *
 * Copyright (c) 2024 Maximilian Behr
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <cusolverDn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "cugje.h"

int main(int argc, char **argv) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    int ret = 0;                              // return value
    int n = 1000;                             // size of the input matrix n-by-n
    int m = n;                                // size of the right-hand side matrix n-by-m
    double *A, *RHS;                          // input matrix and right-hand side matrix
    double *dA, *dRHS, *dA2, *dRHS2;          // device matrices
    double *dbuffer = NULL, *hbuffer = NULL;  // buffer for cusolver
    int64_t *d_ipiv = NULL;                   // pivoting sequence
    int *d_info = NULL;                       // error code

    /*-----------------------------------------------------------------------------
     * parse command line arguments
     *-----------------------------------------------------------------------------*/
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        m = atoi(argv[2]);
    }
    if (argc > 3) {
        printf("Usage: %s [n] [m]\n", argv[0]);
        return 1;
    }

    /*-----------------------------------------------------------------------------
     * allocate
     *-----------------------------------------------------------------------------*/
    cudaMallocHost((void **)&A, sizeof(*A) * n * n);
    cudaMallocHost((void **)&RHS, sizeof(*RHS) * n * m);
    cudaMalloc((void **)&dA, sizeof(*dA) * n * n);
    cudaMalloc((void **)&dRHS, sizeof(*dRHS) * n * m);
    cudaMalloc((void **)&dA2, sizeof(*dA2) * n * n);
    cudaMalloc((void **)&dRHS2, sizeof(*dRHS2) * n * m);
    cudaMalloc((void **)&d_ipiv, sizeof(*d_ipiv) * n);
    cudaMalloc((void **)&d_info, sizeof(*d_info));

    /*-----------------------------------------------------------------------------
     * create a random matrix A and RHS
     *-----------------------------------------------------------------------------*/
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i + j * n] = (double)rand() / RAND_MAX;
        }
        for (int j = 0; j < m; ++j) {
            RHS[i + j * n] = (double)rand() / RAND_MAX;
        }
    }

    /*-----------------------------------------------------------------------------
     * copy data to the device
     *-----------------------------------------------------------------------------*/
    cudaMemcpy(dA, A, n * n * sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(dRHS, RHS, n * m * sizeof(*RHS), cudaMemcpyHostToDevice);
    cudaMemcpy(dA2, A, n * n * sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(dRHS2, RHS, n * m * sizeof(*RHS), cudaMemcpyHostToDevice);

    /*-----------------------------------------------------------------------------
     * measure the time and perform Gauss-Jordan Elimination on the device
     *-----------------------------------------------------------------------------*/
    double wtime_cugjed = 0.0;
    {
        auto start = std::chrono::high_resolution_clock::now();
        cugjed(n, m, dA, dRHS);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        wtime_cugjed = elapsed.count();
    }

    /*-----------------------------------------------------------------------------
     * measaure the time and solve the linear system with cusolver
     *-----------------------------------------------------------------------------*/
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);

    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);

    size_t lworkdevice = 0, lworkhost = 0;
    cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, CUDA_R_64F, dA2, n, CUDA_R_64F, &lworkdevice, &lworkhost);

    if (lworkdevice > 0) {
        cudaMalloc((void **)&dbuffer, lworkdevice);
    }

    if (lworkhost > 0) {
        cudaMallocHost((void **)&hbuffer, lworkhost);
    }

    double wtime_cusolver = 0.0;
    {
        auto start = std::chrono::high_resolution_clock::now();
        cusolverDnXgetrf(cusolverH, params, n, n, CUDA_R_64F, dA2, n, d_ipiv, CUDA_R_64F, dbuffer, lworkdevice, hbuffer, lworkhost, d_info);
        cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, m, CUDA_R_64F, dA2, n, d_ipiv, CUDA_R_64F, dRHS2, n, d_info);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        wtime_cusolver = elapsed.count();
    }

    /*-----------------------------------------------------------------------------
     * print results
     *-----------------------------------------------------------------------------*/
    printf("%d,%d,%e,%e\n", n, m, wtime_cugjed, wtime_cusolver);

    /*-----------------------------------------------------------------------------
     * destroy the cusolver handle
     *-----------------------------------------------------------------------------*/
    cusolverDnDestroy(cusolverH);

    /*-----------------------------------------------------------------------------
     * clear memory
     *-----------------------------------------------------------------------------*/
    cudaFreeHost(A);
    cudaFreeHost(RHS);
    cudaFree(dA);
    cudaFree(dRHS);
    cudaFree(dA2);
    cudaFree(dRHS2);
    cudaFree(dbuffer);
    cudaFreeHost(hbuffer);
    cudaFree(d_ipiv);
    cudaFree(d_info);

    return ret;
}
