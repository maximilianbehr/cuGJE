# cuGJE - Matrix Sign Function Approximation using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuGJE)

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuGJE` is a `CUDA` library implementing Gauss-Jordan Eliminiation to solve the linear systems $AX=B$, 
where $A$ is an n-x-n nonsingular matrix and the right-hand side $B$ is of size n-x-m.

**The implementation is just for educational purposes. Use `cuSolver` for a fast linear systems solver.**

`cuGJE` supports real and complex, single and double precision matrices.

## Available Functions


### Single Precision Functions
```C
int cugjes(int n, int m, float* A, float* RHS);
```

### Double Precision Functions
```C
int cugjed(int n, int m, double* A, double* RHS);
```

### Complex Single Precision Functions
```C
int cugjec(int n, int m, cuComplex* A, cuComplex* RHS);
```

### Complex Double Precision Functions
```C
int cugjez(int n, int m, cuDoubleComplex* A, cuDoubleComplex* RHS);

```


## Algorithm

`cuGJE` implements Gauss-Jordan Eliminiation to solve the linear systems $AX=B$, 
where $A$ is an n-x-n nonsingular matrix and the right-hand side $B$ is of size n-x-m.

## Installation

Prerequisites:
 * `CMake >= 3.23`
 * `CUDA >= 11.4.2`

```shell
  mkdir build && cd build
  cmake ..
  make
  make install
```

## Usage and Examples

We provide examples for all supported matrix formats:
  
| File                                     | Data                               |
| -----------------------------------------|------------------------------------|
| [`example_cugjes.cu`](example_cugjes.cu) | real, single precision matrix      |
| [`example_cugjed.cu`](example_cugjed.cu) | real, double precision matrix      |
| [`example_cugjec.cu`](example_cugjec.cu) | complex, single precision matrix   |
| [`example_cugjez.cu`](example_cugjez.cu) | complex, double precision matrix   |
