#include <cuda_runtime.h>
#include <cstdint>

void callBackwardAdjustWeight(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta);

void callBackwardAdjustBias(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta);

void callBackwardAdjustDelta(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextDelta, float* delta);

__global__ void backwardAdjustWeight(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextLayerDelta, float* thisLayerDelta);

__global__ void backwardAdjustBias(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextLayerDelta, float* thisLayerDelta);

__global__ void backwardAdjustDelta(float* input, float* weights, float* bias, float* out, char activationFunction, size_t inputDim, size_t outputDim, bool isLastLayer, float* expectedOutput, float* nextLayerDelta, float* thisLayerDelta);