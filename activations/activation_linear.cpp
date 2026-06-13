#include "activation_linear.hpp"

void ActivationLinear::forward(const Matrix& inputs, bool training){
    outputs = inputs;
}

void ActivationLinear::backward(const Matrix& dvalues){
    dinputs = dvalues.copy();
}

Matrix ActivationLinear::predictions(const Matrix& outputs) const{
    return outputs;
}