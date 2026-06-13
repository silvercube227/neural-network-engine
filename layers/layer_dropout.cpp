#include "layer_dropout.hpp"

LayerDropout::LayerDropout(double rate) : rate_(1 - rate){
}

void LayerDropout::forward(const Matrix& inputs, bool training){
    if (not training){
        outputs = inputs.copy();
        return;
    }
    inputs_ = inputs;
    binary_mask = Matrix::bernoulli(inputs.getRows(), inputs.getCols(), rate_).divide(rate_);
    outputs = inputs.multiply(binary_mask);
}

void LayerDropout::backward(const Matrix& dvalues){
    //gradient on values
    //derivative of the bernoulli distribution is 1/1-q if success, 0 if fail
    dinputs = dvalues.multiply(binary_mask);
}