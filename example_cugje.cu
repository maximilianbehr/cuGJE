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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <checkcuda.h>
#include <stdio.h>
#include <stdlib.h>

#include "cugje.h"

int main(void) {
    /*-----------------------------------------------------------------------------
     * variables
     *-----------------------------------------------------------------------------*/
    int ret = 0;      // return value
    const int n = 4;  // size of the input matrix n-by-n
    double *A;        // input matrix

    /*-----------------------------------------------------------------------------
     * allocate
     *-----------------------------------------------------------------------------*/
    CHECK_CUDA(cudaMallocManaged((void **)&A, sizeof(*A) * n * n));

    /*-----------------------------------------------------------------------------
     * create a random matrix
     *-----------------------------------------------------------------------------*/
    srand(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i + j * n] = (double)rand() / RAND_MAX;
        }
    }

    /*-----------------------------------------------------------------------------
     * create cublas handle
     *-----------------------------------------------------------------------------*/
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * start Gauss-Jordan Elimination elimination of lower triangular part
     *-----------------------------------------------------------------------------*/
    for (int i = 0; i < n; ++i) {
        // find pivot - the maximum element in the i-th column
        int pivot;
        CHECK_CUBLAS(cublasIdamax(cublasH, n - i, &A[i + i * n], 1, &pivot));
        pivot += i - 1;

        // swap rows
        if (pivot != i) {
            CHECK_CUBLAS(cublasDswap(cublasH, n - i, &A[i + i * n], n, &A[pivot + i * n], n));
        }

        // divide the pivot row by the pivot element
        double alpha = 1.0 / A[i + i * n];
        CHECK_CUBLAS(cublasDscal(cublasH, n - i, &alpha, &A[i + i * n], n));

        // eliminate all other elements in the i-th column
        // by subtracting the pivot row multiplied by the corresponding factor
        for (int j = i + 1; j < n; ++j) {
            if (j != i) {
                alpha = -A[j + i * n];
                CHECK_CUBLAS(cublasDaxpy(cublasH, n - i, &alpha, &A[i + i * n], n, &A[i + j * n], n));
            }
        }
    }

    /*-----------------------------------------------------------------------------
     * print the result
     *-----------------------------------------------------------------------------*/
    printf("Gauss-Jordan Elimination of lower triangular part\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", A[i + j * n]);
        }
        printf("\n");
    }

    /*-----------------------------------------------------------------------------
     * destroy cublas handle
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cublasDestroy(cublasH));

    /*-----------------------------------------------------------------------------
     * clear memory
     *-----------------------------------------------------------------------------*/
    cudaFree(A);
    return ret;
}
