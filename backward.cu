#include <cuda_runtime.h>
#include "backward.cuh"
#include "forward.cuh"
#include <cassert>
#include <cstdio>

void callBackwardAdjustWeight(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta) {
    backwardAdjustWeight<<<1, inputDim * outputDim>>>(input, weights, bias, out, activationFunction, inputDim, outputDim, isLastLayer, expectedOutput, nextDelta, delta);
}

void callBackwardAdjustBias(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta) {
    backwardAdjustBias<<<1, outputDim>>>(input, weights, bias, out, activationFunction, inputDim, outputDim, isLastLayer, expectedOutput, nextDelta, delta);
}

void callBackwardAdjustDelta(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta) {
    backwardAdjustDelta<<<1, inputDim>>>(input, weights, bias, out, activationFunction, inputDim, outputDim, isLastLayer, expectedOutput, nextDelta, delta);
}

__global__ void backwardAdjustWeight(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextLayerDelta, float* thisLayerDelta) {
    // inputIndex is equal to the index into the earlier layer's activations
    size_t weightIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (isLastLayer) {
        // outputIndex is row index, and inputIndex is column index.
        size_t inputIndex = weightIndex % inputDim;
        size_t outputIndex = weightIndex / inputDim;
        // These below are the exact equations as in the readme which calculate how much to adjust each weight
        float costDerivative = 2 * (*(out + outputIndex) - *(expectedOutput + outputIndex));
        float activationDerivative = *(out + outputIndex) * (1.0f - *(out + outputIndex));
        float inputActivation = *(input + inputIndex);
        float weightGradient = costDerivative * activationDerivative * inputActivation;
        *(weights + weightIndex) = *(weights + weightIndex) - (.01 * weightGradient);
    } else {
        // outputIndex is row index, and inputIndex is column index.
        size_t inputIndex = weightIndex % inputDim;
        size_t outputIndex = weightIndex / inputDim;
        // These below are the exact equations as in the readme which calculate how much to adjust each weight
        // delta index is the exact same as index for output activation this weight attaches to
        float costDerivative = 0.0f;
        for (int j = 0; j < outputDim; j++) {
            costDerivative += weights[j * inputDim + outputIndex] * nextLayerDelta[j];
        }
        float activationDerivative = *(out + outputIndex) * (1.0f - *(out + outputIndex));
        float inputActivation = *(input + inputIndex);
        float weightGradient = costDerivative * activationDerivative * inputActivation;
        *(weights + weightIndex) = *(weights + weightIndex) - (.01 * weightGradient);
    }
}

__global__ void backwardAdjustBias(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextLayerDelta, float* thisLayerDelta) {
    size_t biasIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (isLastLayer) {
        size_t biasIndex = blockIdx.x * blockDim.x + threadIdx.x;
        float costDerivative = 2 * (*(out + biasIndex) - *(expectedOutput + biasIndex));
        float activationDerivative = *(out + biasIndex) * (1.0f - *(out + biasIndex));
        float biasGradient = costDerivative * activationDerivative;
        *(bias + biasIndex) = *(bias + biasIndex) - (.01 * biasGradient);
    } else {
        size_t biasIndex = blockIdx.x * blockDim.x + threadIdx.x;
        float costDerivative = 0.0f;
        for (int j = 0; j < outputDim; j++) {
            costDerivative += weights[j * inputDim + biasIndex] * nextLayerDelta[j];
        }
        float activationDerivative = *(out + biasIndex) * (1.0f - *(out + biasIndex));
        float biasGradient = costDerivative * activationDerivative;
        *(bias + biasIndex) = *(bias + biasIndex) - (.01 * biasGradient);
    }
}

__global__ void backwardAdjustDelta(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextLayerDelta, float* thisLayerDelta) {
    size_t deltaIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (isLastLayer) {
        float costDerivative = 2 * (*(out + deltaIndex) - *(expectedOutput + deltaIndex));
        printf("Loss at output %zu: %f\n", deltaIndex, (*(out + deltaIndex) - *(expectedOutput + deltaIndex)) * (*(out + deltaIndex) - *(expectedOutput + deltaIndex)));
        float activationDerivative = *(out + deltaIndex) * (1.0f - *(out + deltaIndex));
        *(thisLayerDelta + deltaIndex) = costDerivative * activationDerivative;
    } else {
        float costDerivative = 0.0f;
        for (int j = 0; j < outputDim; j++) {
            costDerivative += weights[j * inputDim + deltaIndex] * nextLayerDelta[j];
        }
        float activationDerivative = *(out + deltaIndex) * (1.0f - *(out + deltaIndex));
        *(thisLayerDelta + deltaIndex) = costDerivative * activationDerivative;
    }
}