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

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "cugje.h"

int main(int argc, char **argv) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    int ret = 0;             // return value
    int n = 10;              // size of the input matrix n-by-n
    int m = n;               // size of the right-hand side matrix n-by-m
    cuComplex *A, *RHS, *X;  // input matrix and right-hand side matrix
    cuComplex *dA, *dRHS;    // device

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
    cudaMallocHost((void **)&X, sizeof(*X) * n * m);
    cudaMalloc((void **)&dA, sizeof(*dA) * n * n);
    cudaMalloc((void **)&dRHS, sizeof(*dRHS) * n * m);

    /*-----------------------------------------------------------------------------
     * create a random matrix A and RHS
     *-----------------------------------------------------------------------------*/
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i + j * n] = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
        }
        for (int j = 0; j < m; ++j) {
            RHS[i + j * n] = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
        }
    }

    /*-----------------------------------------------------------------------------
     * copy data to the device
     *-----------------------------------------------------------------------------*/
    cudaMemcpy(dA, A, n * n * sizeof(*A), cudaMemcpyHostToDevice);
    cudaMemcpy(dRHS, RHS, n * m * sizeof(*RHS), cudaMemcpyHostToDevice);

    /*-----------------------------------------------------------------------------
     * measure the time and perform Gauss-Jordan Elimination on the device
     *-----------------------------------------------------------------------------*/
    auto start = std::chrono::high_resolution_clock::now();
    cugjec(n, m, dA, dRHS);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Elapsed time: %e seconds\n", elapsed.count());

    /*-----------------------------------------------------------------------------
     * copy result back to the host
     *-----------------------------------------------------------------------------*/
    cudaMemcpy(X, dRHS, n * m * sizeof(*dRHS), cudaMemcpyDeviceToHost);

    /*-----------------------------------------------------------------------------
     * check the results
     *-----------------------------------------------------------------------------*/
    float fronornmdiff = 0.0, fronormRHS = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cuComplex sum = {0.0, 0.0};
            for (int k = 0; k < n; ++k) {
                sum = cuCaddf(sum, cuCmulf(A[i + k * n], X[k + j * n]));
            }
            float diff = cuCabsf(cuCsubf(sum, RHS[i + j * n]));
            fronornmdiff += diff * diff;
            fronormRHS += cuCrealf(cuCmulf(cuConjf(RHS[i + j * n]), RHS[i + j * n]));
        }
    }
    fronornmdiff = sqrtf(fronornmdiff);
    fronormRHS = sqrtf(fronormRHS);
    printf("rel. error: %e\n", fronornmdiff / fronormRHS);
    if (fronornmdiff / fronormRHS > 1.0e-1) {
        printf("Gauss-Jordan Elimination failed\n");
        ret = 1;
    } else {
        printf("Gauss-Jordan Elimination succeeded\n");
    }

    /*-----------------------------------------------------------------------------
     * clear memory
     *-----------------------------------------------------------------------------*/
    cudaFree(A);
    cudaFree(RHS);
    cudaFree(X);
    cudaFree(dA);
    cudaFree(dRHS);

    return ret;
}
