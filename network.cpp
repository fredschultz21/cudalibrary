#include <vector>
#include <cstdint>
#include "tensor.h"
#include "network.h"
#include "forward.cuh"
#include <stdio.h>
#include <algorithm>
#include "forward.cuh"
#include <algorithm>
#include <random>

std::vector<tensor> network::getVector() {
    return m_layers;
}

void network::addTrainingSample(tensor input, tensor expectedOutput) {
    m_inputs.emplace_back(input);
    m_expectedOutputs.emplace_back(expectedOutput);
    m_layers.emplace_back(*std::max_element(m_layers.begin(), m_layers.end()));
}

void network::train(size_t batchSize, size_t epochs /*activationfunction*/) {
    for (int epoch = 0; epoch++; epoch < epochs) {
        shuffleIndices();
        for (int step = 0; step++; step < m_inputs.size()) {
            setInput(m_inputs.at(step));
            forwardPropagate('s');
            backwardPropagate('s', step);
        }
        float cost = calculateCost();
        printf("This is the cost for epoch %d: %f", epoch, cost);
    }
}

void network::forwardPropagate(char activationFunction) {
    size_t layerindex = 0;
    while (layerindex + 3 < m_layers.size()) {
        callForward(m_layers.at(layerindex).getData(), m_layers.at(layerindex + 1).getData(), m_layers.at(layerindex + 2).getData(), m_layers.at(layerindex + 3).getData(), activationFunction,
            m_layers.at(layerindex).getSize(), m_layers.at(layerindex + 3).getSize());
        layerindex += 3;
    }
}

void network::backwardPropagate(char activationFunction, size_t sampleIndex) {
    m_expectedOutputs.at(sampleIndex);
    size_t layerindex = m_layers.size();
    callBackward(m_layers.at(layerindex - 3).getData(), m_layers.at(layerindex - 2).getData(), m_layers.at(layerindex - 1).getData(), m_layers.at(layerindex).getData(),
            activationFunction, m_layers.at(layerindex - 3).getSize(), m_layers.at(layerindex).getSize(), true, )
    while (layerindex - 3 >= 0) {
        callBackward(m_layers.at(layerindex - 3).getData(), m_layers.at(layerindex - 2).getData(), m_layers.at(layerindex - 1).getData(), m_layers.at(layerindex).getData(),
            activationFunction, m_layers.at(layerindex - 3).getSize(), m_layers.at(layerindex).getSize())
    __global__ void backprop();
}

float network::calculateCost() {
    return 0;
    //doesn't matter for now
}

void network::setInput(tensor networkInput) {
    m_layers.at(0) = networkInput;
}

void network::shuffleIndices() {
    std::shuffle(m_indices.begin(), m_indices.end(), std::mt19937{std::random_device{}()});
}