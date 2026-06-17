#include "activation_softmax.hpp"
#include <algorithm>

void ActivationSoftmax::forward(const Matrix& inputs, bool training){
    //exponentiate each value in inputs to get unnormalized probs, remove largest to prevent exploding nums
    Matrix exp_values = inputs.add(inputs.max(1, true).multiply(-1.0)).exp();
    //normalize for each sample
    outputs = exp_values.multiply(exp_values.sum(1, true).reciprocal());
}

void ActivationSoftmax::backward(const Matrix& dvalues){
    dinputs = Matrix::zeros(dvalues.getRows(), dvalues.getCols());
    for(int i = 0; i < outputs.getRows(); i++) {
        Matrix single_output = outputs.row(i);  // get row i
        Matrix single_dvalues = dvalues.row(i); // get row i
        single_output = single_output.reshape(-1 , 1);
        single_dvalues = single_dvalues.reshape(-1, 1);
        // ... compute jacobian and dot
        Matrix jacobian = single_output.diagflat(0).add(single_output.dot(single_output.transpose()).multiply(-1.0));
        dinputs = dinputs.setRow(i, jacobian.dot(single_dvalues).reshape(1, -1));
    }
}

Matrix ActivationSoftmax::predictions(const Matrix& outputs) const{
    return outputs.argmax(1);
}