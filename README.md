# cuGJE - Matrix Sign Function Approximation using CUDA

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
 ![GitHub Release](https://img.shields.io/github/v/release/maximilianbehr/cuGJE)

**Copyright:** Maximilian Behr

**License:** The software is licensed under under MIT. See [`LICENSE`](LICENSE) for details.

`cuGJE` is a `CUDA` library implementing Gauss-Jordan Eliminiation to solve linear systems.

`cuGJE` supports real and complex, single and double precision matrices.

## Available Functions


### Single Precision Functions
```C
```

### Double Precision Functions
```C
```

### Complex Single Precision Functions
```C
```

### Complex Double Precision Functions
```C
```


## Algorithm

`cuGJE` implements Gauss-Jordan Eliminiation to solve linear systems.

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
  
| File                                                       | Data                                |
| -----------------------------------------------------------|-------------------------------------|
| [`example_cusignm_sNewton.cu`](example_cusignm_sNewton.cu) | real, single precision matrix       |
