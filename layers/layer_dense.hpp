#pragma once
#include "layer_base.hpp"

class LayerDense : public LayerBase{
    public:
        LayerDense(int n_inputs, int n_neurons, double weight_regularizer_l1 = 0.0, 
            double weight_regularizer_l2 = 0.0, double bias_regularizer_l1 = 0.0, double bias_regularizer_l2 = 0.0);
        void forward(const Matrix& inputs, bool training) override;
        void backward(const Matrix& dvalues) override;
    private:
        Matrix weights_;
        Matrix biases_;
        Matrix inputs_;
        Matrix dweights_;
        Matrix dbiases_;
        double weight_regularizer_l1_, weight_regularizer_l2_, bias_regularizer_l1_, bias_regularizer_l2_;

};