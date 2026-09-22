#include "loss_categorical_crossentropy.hpp"

Matrix Loss_CategoricalCrossentropy::forward(const Matrix& y_pred, const Matrix& y_true){
    int samples = y_pred.getRows();
    Matrix y_pred_clipped = y_pred.clip(1e-7, 1 - 1e-7);
    Matrix correct_confidences(samples, 1);

    if(y_true.getCols() == 1){
        for(int i = 0; i < samples; i++){
            int label = (int)y_true.at(i, 0);
            correct_confidences.at(i, 0) = y_pred_clipped.at(i, label);
        }
    }
    else if(y_true.getCols() > 1){
        correct_confidences = y_pred_clipped.multiply(y_true).sum(1);
    }

    Matrix negative_log_likelihoods = correct_confidences.log().multiply(-1.0);
    return negative_log_likelihoods;
}

void Loss_CategoricalCrossentropy::backward(const Matrix& dvalues, const Matrix& y_true){
    int samples = dvalues.getRows();
    int labels = dvalues.getCols();
    
    Matrix labels_onehot = Matrix::zeros(samples, labels);
    if(y_true.getCols() == 1){
        for(int i = 0; i < samples; i++){
            labels_onehot.at(i, (int)y_true.at(i, 0)) = 1.0;
        }
    } else {
        labels_onehot = y_true;
    }
    
    dinputs = labels_onehot.multiply(-1.0).multiply(dvalues.reciprocal()).divide(samples);
}