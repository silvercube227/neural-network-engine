#pragma once
#include "../core/matrix.hpp"

class Activation_SoftmaxCategoricalCrossentropy{
    public:
        Matrix dinputs;
        void backward(const Matrix& dvalues, const Matrix& y_true);
};