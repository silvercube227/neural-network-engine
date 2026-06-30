#include "activation_softmax_categoricalcrossentropy.hpp"

void Activation_SoftmaxCategoricalCrossentropy::backward(const Matrix& dvalues, const Matrix& y_true){
    int samples = dvalues.getRows();
    Matrix labels = (y_true.getCols() > 1) ? y_true.argmax(1) : y_true;
    Matrix m = dvalues.copy();
    for(int i = 0; i < samples; ++i){
        m.at(i, (int)labels.at(i, 0)) -= 1.0;
    }
    dinputs = m.divide(samples);
}