#pragma once
#include <cstdint>

class tensor{
    public:
        tensor(char type, size_t index, size_t rows, size_t cols, size_t colstride, size_t rowstride);

        char getType() { return m_type; }
        size_t getIndex() { return m_index; }
        void setIndex(size_t index) { m_index = index; }
        size_t getRows() { return m_rows; }
        size_t getCols() { return m_cols; }
        size_t getColstride() { return m_colstride; }
        size_t getRowstide() { return m_rowstride; }
        size_t getSize() { return m_rows * m_cols; }
        float* getData() { return data; }
        static tensor createInputTensor(std::initializer_list<float> values);
        static tensor createOutputTensor(std::initializer_list<float> values);
        void setData(float* newData) { data = newData; }

    private:
        char m_type;
        size_t m_index;
        size_t m_rows;
        size_t m_cols;
        size_t m_colstride;
        size_t m_rowstride;
        float* data;
};