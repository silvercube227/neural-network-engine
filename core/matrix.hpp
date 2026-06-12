#pragma once
#include <vector>
#include <stdexcept>

class Matrix{
    public:
        Matrix();
        Matrix(int rows, int cols, double fill_value = 0.0);
        static Matrix zeros(int rows, int cols);
        static Matrix ones(int rows, int cols);
        static Matrix randn(int rows, int cols);
        static Matrix eye(int size);
        //first const means matrix is read only, & passes ref to avoid unnecessary duplication
        //second const means method doesnt modify object its called on
        Matrix dot(const Matrix& other) const;
        Matrix add(const Matrix& other) const;
        Matrix sum(int axis, bool keepdims = false) const;
        Matrix max(int axis, bool keepdims = false) const;
        Matrix transpose() const;
        int getRows() const;
        int getCols() const;
        double& at(int row, int col);
        double at(int row, int col) const; //read only
        const std::vector<double> getData() const;
        Matrix exp() const; // exponentiation
        Matrix copy() const;
        Matrix multiply(const Matrix& other) const; //element wize multiplication
        Matrix multiply(double scalar) const; //scalar multiplication
        Matrix divide(double scalar) const; //scalar division
        Matrix subtract(double scalar) const; //scalar subtract
        Matrix clip(double min, double max) const; //limit values in range
        Matrix argmax(int axis) const; //return indices of max in array
        void print() const;
        Matrix sign() const;
        static Matrix bernoulli(int rows, int cols, double p);
    private:
        int rows_, cols_;
        std::vector<double> data;
};