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

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "checkcuda.h"
#include "cugje.h"
#include "cugje_traits.h"

__host__ __device__ inline static cuComplex operator*(const cuComplex& a, const cuComplex& b) {
    return cuCmulf(a, b);
}

__host__ __device__ inline static cuDoubleComplex operator*(const cuDoubleComplex& a, const cuDoubleComplex& b) {
    return cuCmul(a, b);
}

__host__ __device__ inline static cuComplex operator-(const cuComplex& a, const cuComplex& b) {
    return cuCsubf(a, b);
}

__host__ __device__ inline static cuDoubleComplex operator-(const cuDoubleComplex& a, const cuDoubleComplex& b) {
    return cuCsub(a, b);
}

__host__ __device__ inline static cuComplex operator/(const cuComplex& a, const cuComplex& b) {
    return cuCdivf(a, b);
}

__host__ __device__ inline static cuDoubleComplex operator/(const cuDoubleComplex& a, const cuDoubleComplex& b) {
    return cuCdiv(a, b);
}

template <typename T>
__global__ void eliminateLower(int n, int m, T* A, T* RHS, int j) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (j < x && x < n && y < m) {
        RHS[x + y * n] = RHS[x + y * n] - A[x + j * n] * RHS[j + y * n];
    }

    if (j < x && x < n && j < y && y < n) {
        A[x + y * n] = A[x + y * n] - A[x + j * n] * A[j + y * n];
    }
}

template <typename T>
__global__ void eliminateUpper(int n, int m, T* A, T* RHS, int j) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= j || x >= n || y >= m) {
        return;
    }
    RHS[x + y * n] = RHS[x + y * n] - A[x + j * n] * RHS[j + y * n];
}

template <typename T>
int cugje(int n, int m, T* A, T* RHS) {
    /*-----------------------------------------------------------------------------
     * threads per block
     *-----------------------------------------------------------------------------*/
    const int blockSize = 16;

    /*-----------------------------------------------------------------------------
     * create cublas handle
     *-----------------------------------------------------------------------------*/
    cublasHandle_t cublasH;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    /*-----------------------------------------------------------------------------
     * Gauss-Jordan Elimination of lower triangular part
     *-----------------------------------------------------------------------------*/
    cudaStream_t streamA, streamRHS;
    CHECK_CUDA(cudaStreamCreate(&streamA));
    CHECK_CUDA(cudaStreamCreate(&streamRHS));
    for (int j = 0; j < n; ++j) {
        // compute maximum element in the j-th column
        int pivot;
        CHECK_CUBLAS(cublasSetStream(cublasH, 0));
        CHECK_CUBLAS(cugje_traits<T>::cublasIxamax(cublasH, n - j, &A[j + j * n], 1, &pivot));
        pivot += j - 1;
        // swap rows of augmented matrix
        if (pivot != j) {
            CHECK_CUBLAS(cublasSetStream(cublasH, streamA));
            CHECK_CUBLAS(cugje_traits<T>::cublasXswap(cublasH, n - j, &A[j + j * n], n, &A[pivot + j * n], n));
            CHECK_CUBLAS(cublasSetStream(cublasH, streamRHS));
            CHECK_CUBLAS(cugje_traits<T>::cublasXswap(cublasH, m, &RHS[j], n, &RHS[pivot], n));
        }

        // get the pivot element
        T elem;
        CHECK_CUDA(cudaMemcpy(&elem, &A[j + j * n], sizeof(elem), cudaMemcpyDeviceToHost));
        // divide the pivot row by the pivot element
        if (cugje_traits<T>::nonzero(elem)) {
            T alpha = cugje_traits<T>::one / elem;
            CHECK_CUBLAS(cublasSetStream(cublasH, streamA));
            CHECK_CUBLAS(cugje_traits<T>::cublasXscal(cublasH, n - j, &alpha, &A[j + j * n], n));
            CHECK_CUBLAS(cublasSetStream(cublasH, streamRHS));
            CHECK_CUBLAS(cugje_traits<T>::cublasXscal(cublasH, m, &alpha, &RHS[j], n));

            // eliminate lower triangular part
            dim3 block(blockSize, blockSize);
            dim3 grid((n + block.x - 1) / block.x, (max(n, m) + block.y - 1) / block.y);
            eliminateLower<<<grid, block>>>(n, m, A, RHS, j);
            CHECK_CUDA(cudaPeekAtLastError());
        }
    }

    /*-----------------------------------------------------------------------------
     *  Gauss-Jordan Elimination of upper triangular part
     *-----------------------------------------------------------------------------*/
    for (int j = n - 1; j >= 0; --j) {
        // eliminate upper triangular part
        dim3 block(blockSize, blockSize);
        dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
        eliminateUpper<<<grid, block>>>(n, m, A, RHS, j);
        CHECK_CUDA(cudaPeekAtLastError());
    }

    /*-----------------------------------------------------------------------------
     * destroy cublas handle
     *-----------------------------------------------------------------------------*/
    CHECK_CUBLAS(cublasDestroy(cublasH));
    CHECK_CUDA(cudaStreamDestroy(streamA));
    CHECK_CUDA(cudaStreamDestroy(streamRHS));

    /*-----------------------------------------------------------------------------
     * return
     *-----------------------------------------------------------------------------*/
    return 0;
}

int cugjes(int n, int m, float* A, float* RHS) {
    return cugje(n, m, A, RHS);
}

int cugjed(int n, int m, double* A, double* RHS) {
    return cugje(n, m, A, RHS);
}

int cugjec(int n, int m, cuComplex* A, cuComplex* RHS) {
    return cugje(n, m, A, RHS);
}

int cugjez(int n, int m, cuDoubleComplex* A, cuDoubleComplex* RHS) {
    return cugje(n, m, A, RHS);
}
