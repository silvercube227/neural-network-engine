#pragma once
#include "activation_base.hpp"

class ActivationSigmoid : public ActivationBase { 
    public:
        void forward(const Matrix& inputs, bool training) override;
        void backward(const Matrix& dvalues) override;
        Matrix predictions(const Matrix& outputs) const override;
};