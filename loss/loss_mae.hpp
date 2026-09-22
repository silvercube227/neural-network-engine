#pragma once
#include "loss_base.hpp"

class Loss_MAE : public LossBase {
    public:
        Matrix forward(const Matrix& y_pred, const Matrix& y_true) override;
        void backward(const Matrix& dvalues, const Matrix& y_true) override;
};