#include "activation_sigmoid.hpp"

void ActivationSigmoid::forward(const Matrix& inputs, bool training){
    Matrix ones = Matrix::ones(inputs.getRows(), inputs.getCols());
    outputs = ones.add(inputs.multiply(-1.0).exp()).reciprocal();
}

void ActivationSigmoid::backward(const Matrix& dvalues){
    Matrix ones = Matrix::ones(outputs.getRows(), outputs.getCols());
    dinputs = dvalues.multiply(ones.add(outputs.multiply(-1.0))).multiply(outputs);
}

Matrix ActivationSigmoid::predictions(const Matrix& outputs) const{
    return outputs.greaterThan(0.5).multiply(1.0);
}