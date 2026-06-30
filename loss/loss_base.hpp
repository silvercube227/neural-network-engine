#pragma once
#include "../core/matrix.hpp"
#include <vector>
#include "../layers/layer_base.hpp"
#include <tuple>

class LossBase{
    public:
        Matrix dinputs;
        virtual Matrix forward(const Matrix& y_pred, const Matrix& y_true) = 0;
        virtual void backward(const Matrix& dvalues, const Matrix& y_true) = 0;
        std::tuple<double, double> calculate(const Matrix& outputs, const Matrix& y, bool include_regularization=false);
        double regularization_loss();
        void remember_trainable_layers(std::vector<LayerBase*> trainable_layers);

    private:
        std::vector<LayerBase*> trainable_layers_;

};