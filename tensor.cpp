#include <cstdint>
#include <cuda_runtime.h>
#include "tensor.h"
#include <vector>

tensor::tensor(char type, size_t index, size_t rows, size_t cols, size_t colstride, size_t rowstride) {
    m_type = type;
    m_index = index;
    m_rows = rows;
    m_cols = cols;
    m_colstride = colstride;
    m_rowstride = rowstride;
    cudaMalloc((void**)&data, sizeof(float) * rows * cols);
}

tensor tensor::createInputTensor(std::initializer_list<float> values) {
    tensor t('a', 0, values.size(), 1, 0, 1);
    std::vector<float> hostData(values);
    float* gpuData;
    cudaMalloc((void**)&gpuData, sizeof(float) * hostData.size());
    cudaMemcpy(gpuData, hostData.data(), sizeof(float) * hostData.size(), cudaMemcpyHostToDevice);
    t.setData(gpuData);
    return t;
}

tensor tensor::createOutputTensor(std::initializer_list<float> values) {
    tensor t('a', 0, values.size(), 1, 0, 1);
    std::vector<float> hostData(values);
    float* gpuData;
    cudaMalloc((void**)&gpuData, sizeof(float) * hostData.size());
    cudaMemcpy(gpuData, hostData.data(), sizeof(float) * hostData.size(), cudaMemcpyHostToDevice);
    t.setData(gpuData);
    return t;
}