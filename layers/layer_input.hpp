#pragma once
#include "layer_base.hpp"

//generic layer that contains trainging to be used as previous layers for first pass to dense
class LayerInput : public LayerBase{
    public:
        void forward(const Matrix& inputs, bool training) override;
        void backward(const Matrix& dvalues) override;
};