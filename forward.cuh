#include <cuda_runtime.h>

void callForward(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputdim, size_t outputdim);

__global__ void forward(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputdim);

__device__ float sigmoid(float x);