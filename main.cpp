#include "network.h"
#include "tensor.h"
#include <iostream>

int main() {
    network net({3, 2, 1});

    std::cout << "main" << "\n";

    tensor input1 = tensor::createInputTensor({1.0f, 1.0f, 1.0f});
    tensor expected1 = tensor::createOutputTensor({1.0f});

    net.addTrainingSample(input1, expected1);
    net.train(1, 100);

    return 0;
}