#include "loss_binary_crossentropy.hpp"

Matrix LossBinaryCrossentropy::forward(const Matrix& y_pred, const Matrix& y_true){
    //clip data to prevent division by 0 
    //clip both sides to not drag mean toward any value
    Matrix y_pred_clipped = y_pred.clip(1e-7, 1 - 1e-7);
    //sample-wide loss
    Matrix ones_true = Matrix::ones(y_true.getRows(), y_true.getCols());
    Matrix ones_pred = Matrix::ones(y_pred.getRows(), y_pred.getCols());
    Matrix sample_loss = (y_true.multiply(y_pred_clipped.log())).multiply(-1.0).add(
        (ones_true.add(y_true.multiply(-1.0))).multiply(ones_pred.add(y_pred_clipped.multiply(-1.0)).log())
    );
    Matrix sample_losses = sample_loss.sum(1).divide(y_pred.getCols());
    return sample_losses;
}

void LossBinaryCrossentropy::backward(const Matrix& dvalues, const Matrix& y_true){
    int samples = dvalues.getRows();
    int outputs = dvalues.getCols();
    Matrix dvalues_clipped = dvalues.clip(1e-7, 1 - 1e-7);
    //calc gradient
    Matrix ones_true = Matrix::ones(y_true.getRows(), y_true.getCols());
    dinputs = y_true.multiply(dvalues_clipped.reciprocal()).add(
        ones_true.add(y_true.multiply(-1.0)).multiply(ones_true.add(dvalues_clipped.multiply(-1.0)).reciprocal()).multiply(-1.0)
    ).multiply(-1.0).divide(outputs).divide(samples);
}

