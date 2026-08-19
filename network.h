#include <vector>
#include <cstdint>
#include "tensor.h"

class network {
    public:
        network(std::initializer_list<size_t> args) {
            int prevRowSize = args.begin()[0];
            m_layers.emplace_back(tensor('a', 0, prevRowSize, 1, 0, 1));
            for (size_t i = 1; i < args.size(); i++) {
                int currRowSize = args.begin()[i];
                m_layers.emplace_back(tensor('w', i, currRowSize, prevRowSize, 1, currRowSize));
                m_layers.emplace_back(tensor('b', i, currRowSize, 1, 0, 1));
                m_layers.emplace_back(tensor('a', i, currRowSize, 1, 0, 1));
                m_deltas.emplace_back(tensor('d', i, prevRowSize, 1, 0, 1));
                prevRowSize = currRowSize;
            }
         }

        std::vector<tensor> network::getVector();

        void addTrainingSample(tensor input, tensor expectedOutput);

        void train(size_t batchSize, size_t epochs);

    private:

        void forwardPropagate(char activationFunction);
            // loop, while theres another layer
    
        float calculateCost();

        void backwardPropagate(char activationFunction, size_t sampleIndex);
        
        void setInput(tensor networkInput);

        void shuffleIndices();

        std::vector<tensor> m_layers;
        std::vector<tensor> m_deltas;

        std::vector<tensor> m_inputs;
        std::vector<tensor> m_expectedOutputs;

        std::vector<size_t> m_indices;
};