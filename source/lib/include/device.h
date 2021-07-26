#pragma once
#include <vector>
#include <climits>
#include <stdio.h>
#include <iostream>

#define TPB 256
#define SQRT_2_PI 0.7978845608028654 
typedef unsigned long long uint_64;

#if GOOGLE_CUDA
#include "gpu_cuda.h"
#endif //GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
#include "gpu_rocm.h"
#endif //TENSORFLOW_USE_ROCM

#if GRAPHCORE_POPLAR
#include "ipu_poplar.h"
#endif //GRAPHCORE_POPLAR
