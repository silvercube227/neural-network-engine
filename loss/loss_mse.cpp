#include "loss_mse.hpp"

Matrix Loss_MSE::forward(const Matrix& y_pred, const Matrix& y_true){
    Matrix sample_losses = y_pred.multiply(-1.0).add(y_true);
    sample_losses = sample_losses.multiply(sample_losses).sum(1).divide(y_pred.getCols());
    return sample_losses;
}

void Loss_MSE::backward(const Matrix& dvalues, const Matrix& y_true){
    int samples = dvalues.getRows();
    int outputs = dvalues.getCols();
    dinputs = dvalues.multiply(-1.0).add(y_true).multiply(-2.0).divide(outputs);
    dinputs = dinputs.divide(samples);
}