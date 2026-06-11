#include "layer_dense.hpp"

LayerDense::LayerDense(int n_inputs, int n_neurons, double weight_regularizer_l1,
    double weight_regularizer_l2, double bias_regularizer_l1, double bias_regularizer_l2):
    weights_(Matrix::randn(n_inputs, n_neurons).multiply(0.1)), 
    biases_(Matrix::zeros(1, n_neurons)),
    weight_regularizer_l1_(weight_regularizer_l1), 
    weight_regularizer_l2_(weight_regularizer_l2), 
    bias_regularizer_l1_(bias_regularizer_l1),
    bias_regularizer_l2_(bias_regularizer_l2) {
}

void LayerDense::forward(const Matrix& inputs, bool training){
    inputs_ = inputs;
    outputs = inputs.dot(weights_).add(biases_);
}

void LayerDense::backward(const Matrix& dvalues){
    dweights_ = inputs_.transpose().dot(dvalues);
    dbiases_ = dvalues.sum(0, true);

    if(weight_regularizer_l1_ > 0){
        dweights_ = dweights_.add(weights_.sign().multiply(weight_regularizer_l1_));
    }
    if(bias_regularizer_l1_ > 0){
        dbiases_ = dbiases_.add(biases_.sign().multiply(bias_regularizer_l1_));
    }
    if(weight_regularizer_l2_ > 0){
        dweights_ = dweights_.add(weights_.multiply(weight_regularizer_l2_ * 2.0));
    }
    if(bias_regularizer_l2_ > 0){
        dbiases_ = dbiases_.add(biases_.multiply(bias_regularizer_l2_ * 2.0));
    }
    dinputs = dvalues.dot(weights_.transpose());
}