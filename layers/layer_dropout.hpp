#pragma once
#include "layer_base.hpp"

class LayerDropout : public LayerBase{
    public:
        LayerDropout(double rate); 
        void forward(const Matrix& inputs, bool training) override;
        void backward(const Matrix& dvalues) override;

    private:
        double rate_;
        Matrix binary_mask;
        Matrix inputs_;

};