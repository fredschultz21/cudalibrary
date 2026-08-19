#include <cstdint>
#include <cuda_runtime.h>
#include "tensor.h"

tensor::tensor(char type, size_t index, size_t rows, size_t cols, size_t colstride, size_t rowstride) {
    m_type = type;
    m_index = index;
    m_rows = rows;
    m_cols = cols;
    m_colstride = colstride;
    m_rowstride = rowstride;
    cudaMalloc((void**)&data, sizeof(float) * rows * cols);
}