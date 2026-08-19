#include <cuda_runtime.h>

#include "forward.cuh"
#include <cassert>

void callForward(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputdim, size_t outputdim) {
    forward<<<1, outputdim>>>(input, weights, bias, out, activationFunction, inputdim);
}

__global__ void forward(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputdim) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int c = 0; c < inputdim; c++) {
        sum += weights[c + row * inputdim] * input[c];
    }
    sum += *(bias + row);
    switch (activationFunction) {
        case 's':
            *(out + row) = sigmoid(sum);
            break;
        case 'r':
            *(out + row) = relu(sum);
            break;
        case 't':
            *(out + row) = tanhAct(sum);
            break;
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ float tanhAct(float x) {
    return tanhf(x);
}