#pragma once
#include "../layers/layer_base.hpp"

class ActivationBase : public LayerBase{
    public:
        virtual Matrix predictions(const Matrix& outputs) const = 0;
};