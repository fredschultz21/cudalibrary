#include <cuda_runtime.h>

#include "forward.cuh"
#include <cassert>

void callBackward(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputdim, size_t outputdim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta) {
    backward<<<1, inputdim>>>(input, weights, bias, out, activationFunction, inputdim, outputdim, isLastLayer, expectedOutput, nextDelta, delta);
}

__global__ void backward(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputdim, size_t outputdim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta) {
    size_t inputIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (isLastLayer) {
        for (int i = 0; i < outputdim; i++) {
            float inputActivation = *(input + inputIndex);
            float activationDerivative = *(out + i) * (1.0f - *(out + i));
            float costDerivative = 2 * (*(out + i) - *(expectedOutput + i));
            *(delta + i) = inputActivation * activationDerivative * costDerivative;
        }
    } else {
        for (int i = 0; i < inputdim; i++) {
            bias + row
        }
    }
}





// float inputActivation = *(input + inputIndex);
//             float activationDerivative = *(out + i) * (1.0f - *(out + i));
//             float costDerivative = 2 * (*(out + i) - *(expectedOutput + i));