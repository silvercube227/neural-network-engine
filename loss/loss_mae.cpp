#include "loss_mae.hpp"

Matrix Loss_MAE::forward(const Matrix& y_pred, const Matrix& y_true){
    Matrix sample_losses = y_true.add(y_pred.multiply(-1.0)).abs().sum(1).divide(y_pred.getCols());
    return sample_losses;
}

void Loss_MAE::backward(const Matrix& dvalues, const Matrix& y_true){
    int samples = dvalues.getRows();
    int outputs = dvalues.getCols();

}

