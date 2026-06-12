#include "layer_input.hpp"

void LayerInput::forward(const Matrix& inputs, bool training){
    outputs = inputs;
}

void LayerInput::backward(const Matrix& dvalues) {}