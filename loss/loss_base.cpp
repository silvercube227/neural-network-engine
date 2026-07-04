#include "loss_base.hpp"


void LossBase::remember_trainable_layers(std::vector<LayerBase*> trainable_layers){
    trainable_layers_ = trainable_layers;
}

//regularization loss calc
//L1 = lambda * sum abs weights
//l2 = lambda * sum weights **2
double LossBase::regularization_loss(){
    double regularization_loss = 0.0;
    for(int i = 0; i < trainable_layers_.size(); ++i){
        regularization_loss += trainable_layers_[i]->regularization_loss();

    }
    return regularization_loss;
}


std::tuple<double, double> LossBase::calculate(const Matrix& outputs, const Matrix& y, bool include_regularization){
    Matrix sample_losses = forward(outputs, y);
    double data_loss = sample_losses.mean();
    if(not include_regularization){
        std::tuple<double, double> loss = std::make_tuple(data_loss, 0.0);
        return loss;
    }
    std::tuple<double, double> loss = std::make_tuple(data_loss, regularization_loss());
    return loss;
}
