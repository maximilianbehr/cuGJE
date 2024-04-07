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
     * constants
     *-----------------------------------------------------------------------------*/
    inline static constexpr double one = 1.0;

    /*-----------------------------------------------------------------------------
     * nonzero function
     *-----------------------------------------------------------------------------*/
    inline static bool nonzero(const double &a) {
        return a != 0.0;
    }

    /*-----------------------------------------------------------------------------
     * maximum absolute value, scaling, and swap
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const double *x, int incx, int *result) {
        return cublasIdamax(handle, n, x, incx, result);
    }

    inline static cublasStatus_t cublasXscal(cublasHandle_t handle, int n, const double *alpha, double *x, int incx) {
        return cublasDscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXswap(cublasHandle_t handle, int n, double *x, int incx, double *y, int incy) {
        return cublasDswap(handle, n, x, incx, y, incy);
    }
};

template <>
struct cugje_traits<float> {
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    inline static constexpr float one = 1.0f;
    inline static constexpr float zero = 0.0;

    /*-----------------------------------------------------------------------------
     * nonzero function
     *-----------------------------------------------------------------------------*/
    inline static bool nonzero(const float &a) {
        return a != 0.0f;
    }

    /*-----------------------------------------------------------------------------
     * maximum absolute value, scaling, and swap
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const float *x, int incx, int *result) {
        return cublasIsamax(handle, n, x, incx, result);
    }

    inline static cublasStatus_t cublasXscal(cublasHandle_t handle, int n, const float *alpha, float *x, int incx) {
        return cublasSscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXswap(cublasHandle_t handle, int n, float *x, int incx, float *y, int incy) {
        return cublasSswap(handle, n, x, incx, y, incy);
    }
};

template <>
struct cugje_traits<cuDoubleComplex> {
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    inline static constexpr cuDoubleComplex one = cuDoubleComplex{1.0, 0.0};

    /*-----------------------------------------------------------------------------
     * nonzero function
     *-----------------------------------------------------------------------------*/
    inline static bool nonzero(const cuDoubleComplex &a) {
        return a.x != 0.0 || a.y != 0.0;
    }

    /*-----------------------------------------------------------------------------
     * maximum absolute value, scaling, and swap
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const cuDoubleComplex *x, int incx, int *result) {
        return cublasIzamax(handle, n, x, incx, result);
    }

    inline static cublasStatus_t cublasXscal(cublasHandle_t handle, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx) {
        return cublasZscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXswap(cublasHandle_t handle, int n, cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy) {
        return cublasZswap(handle, n, x, incx, y, incy);
    }
};

template <>
struct cugje_traits<cuComplex> {
    /*-----------------------------------------------------------------------------
     * constants
     *-----------------------------------------------------------------------------*/
    inline static constexpr cuComplex one = cuComplex{1.0f, 0.0f};

    /*-----------------------------------------------------------------------------
     * nonzero function
     *-----------------------------------------------------------------------------*/
    inline static bool nonzero(const cuComplex &a) {
        return a.x != 0.0f || a.y != 0.0f;
    }

    /*-----------------------------------------------------------------------------
     * maximum absolute value, scaling, and swap
     *-----------------------------------------------------------------------------*/
    inline static cublasStatus_t cublasIxamax(cublasHandle_t handle, int n, const cuComplex *x, int incx, int *result) {
        return cublasIcamax(handle, n, x, incx, result);
    }

    inline static cublasStatus_t cublasXscal(cublasHandle_t handle, int n, const cuComplex *alpha, cuComplex *x, int incx) {
        return cublasCscal(handle, n, alpha, x, incx);
    }

    inline static cublasStatus_t cublasXswap(cublasHandle_t handle, int n, cuComplex *x, int incx, cuComplex *y, int incy) {
        return cublasCswap(handle, n, x, incx, y, incy);
    }
};
