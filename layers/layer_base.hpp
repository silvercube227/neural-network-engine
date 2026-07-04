#pragma once
#include "../core/matrix.hpp"

class LayerBase{
    public:
        Matrix outputs;
        Matrix dinputs;
        
        //virtual - subclass override, = 0 -- pure virtual must override
        virtual void forward(const Matrix& inputs, bool training) = 0;
        virtual void backward(const Matrix& dvalues) = 0;

        //virtual destructor needed for inheritance, = default means use default constructor
        virtual ~LayerBase() = default;
        virtual double regularization_loss() { return 0.0; }
};