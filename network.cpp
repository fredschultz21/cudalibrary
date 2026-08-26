#include <vector>
#include <cstdint>
#include "tensor.h"
#include "network.h"
#include "forward.cuh"
#include <stdio.h>
#include <algorithm>
#include "backward.cuh"
#include <algorithm>
#include <random>

std::vector<tensor> network::getVector() {
    return m_layers;
}

void network::addTrainingSample(tensor input, tensor expectedOutput) {
    m_inputs.emplace_back(input);
    expectedOutput.setIndex(m_expectedOutputs.size());
    m_expectedOutputs.emplace_back(expectedOutput);
}

void network::train(size_t batchSize, size_t epochs /*activationfunction*/) {
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffleIndices();
        for (int step = 0; step < m_inputs.size(); step++) {
            setInput(m_inputs.at(step));
            forwardPropagate('s');
            backwardPropagate('s', step);
        }
        float cost = calculateCost();
        //printf("This is the cost for epoch %d: %f", epoch, cost);
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
    int layerindex = (int)m_layers.size() - 1; // index of last activation tensor
    float* nextDelta = nullptr;

    while (layerindex - 3 >= 0) {
        float* input = m_layers.at(layerindex - 3).getData();
        float* weights = m_layers.at(layerindex - 2).getData();
        float* bias = m_layers.at(layerindex - 1).getData();
        float* out = m_layers.at(layerindex).getData();
        size_t inputDim = m_layers.at(layerindex - 3).getSize();
        size_t outputDim = m_layers.at(layerindex).getSize();
        bool isLastLayer = (layerindex == (int)m_layers.size() - 1);

        float* thisDelta = m_deltas.at((layerindex - 1) / 3).getData();

        callBackwardAdjustDelta(input, weights, bias, out, activationFunction, inputDim, outputDim, isLastLayer,
            m_expectedOutputs.at(sampleIndex).getData(), nextDelta, thisDelta);

        callBackwardAdjustWeight(input, weights, bias, out, activationFunction, inputDim, outputDim, isLastLayer,
            m_expectedOutputs.at(sampleIndex).getData(), nextDelta, thisDelta);

        callBackwardAdjustBias(input, weights, bias, out, activationFunction, inputDim, outputDim, isLastLayer,
            m_expectedOutputs.at(sampleIndex).getData(), nextDelta, thisDelta);

        nextDelta = thisDelta;
        layerindex -= 3;
    }
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