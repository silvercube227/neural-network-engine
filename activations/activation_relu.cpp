#include "activation_relu.hpp"

void ActivationReLU::forward(const Matrix& inputs, bool training){
    inputs_ = inputs;
    outputs = inputs.maximum(0.0);
}

void ActivationReLU::backward(const Matrix& dvalues){
    dinputs = dvalues.multiply(inputs_.greaterThan(0.0));
}

Matrix ActivationReLU::predictions(const Matrix& outputs) const{
    return outputs;
}