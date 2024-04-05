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

#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>

template <typename T>
struct cugje_traits;

template <>
struct cugje_traits<double> {
    /*-----------------------------------------------------------------------------
     * maximum absolute value of a vector using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
        return cublasIdamax(handle, n, x, incx, result);
    }
};

template <>
struct cugje_traits<float> {
    /*-----------------------------------------------------------------------------
     * maximum absolute value of a vector using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
        return cublasIsamax(handle, n, x, incx, result);
    }
};

template <>
struct cugje_traits<cuDoubleComplex> {
    /*-----------------------------------------------------------------------------
     * maximum absolute value of a vector using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
        return cublasIzamax(handle, n, x, incx, result);
    }
};

template <>
struct cugje_traits<cuComplex> {
    /*-----------------------------------------------------------------------------
     * maximum absolute value of a vector using cuBLAS
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
        return cublasIcamax(handle, n, x, incx, result);
    }
};
