#include "matrix.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>


Matrix::Matrix() : rows_(0), cols_(0), data(0) {}

Matrix::Matrix(int rows, int cols, double fill_value): rows_(rows), cols_(cols), data(rows*cols, fill_value){

}

Matrix Matrix::zeros(int rows, int cols){
    return Matrix(rows, cols);
}

Matrix Matrix::ones(int rows, int cols){
    return Matrix(rows, cols, 1.0);
}

//generate a random number using gaussian dist
Matrix Matrix::randn(int rows, int cols){
    Matrix m(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    for(int i = 0; i < rows * cols; i++){
        m.data[i] = dist(gen);
    }
    return m;
}

//identity is square so single input
Matrix Matrix::eye(int size){
    Matrix m(size, size);
    for(int i = 0; i < size; i++){
        m.data[i * size + i] = 1.0;
    }
    return m;
}

double& Matrix::at(int row, int col){
    if(row >= rows_ || row < 0 || col >= cols_ || col < 0){
        throw std::out_of_range("invalid input");
    }
    return data[row*cols_+col];
}

double Matrix::at(int row, int col) const{
    if(row >= rows_ || row < 0 || col >= cols_ || col < 0){
        throw std::out_of_range("invalid input");
    }
    return data[row*cols_+col];
}

Matrix Matrix::transpose() const{
    Matrix m(cols_, rows_);
    for(int i = 0; i < cols_; i++){
        for(int j = 0; j < rows_; j++){
            m.data[i * rows_ + j] = at(j, i);
        }
    }
    return m;
}

Matrix Matrix::dot(const Matrix& other) const{
    if(cols_ != other.rows_){
        throw std::invalid_argument("shape mismatch for dot product");
    }
    Matrix m(rows_, other.cols_);
    for(int i = 0; i < rows_; i++){
        for(int j = 0; j < other.cols_; j++){
            double value = 0.0;
            for(int k = 0; k < cols_; k++){
                value += at(i, k) * other.at(k, j);
            }
            m.data[i * other.cols_ + j] = value;
        }
    }
    return m;
}

Matrix Matrix::add(const Matrix& other) const{
    if(cols_ != other.cols_){
        throw std::invalid_argument("shape mismatch for add");
    }
    Matrix m(rows_, cols_); //take shape of larger matrix
    for(int i = 0; i < rows_; i++){
        for(int j = 0; j < cols_; j++){
            // i % other.rows_ handles broadcasting: if other has 1 row, always reads row 0
            // if shapes match, i % i == i so it behaves like normal addition
            m.data[i * cols_ + j] = at(i, j) + other.at(i % other.rows_, j);
        }
    }
    return m;
}

Matrix Matrix::sum(int axis, bool keepdims) const {
    //sum along rows, 1 * n
    if(axis == 0){
        Matrix m(1, cols_);
        for(int i = 0; i < rows_; i++){
            for(int j = 0; j < cols_; j++){
                m.data[j] += at(i, j);
            }
        }
        return m;
    }
    //sum along cols, m * 1
    else{
        Matrix m(rows_, 1);
        for(int i = 0; i < rows_; i++){
            for(int j = 0; j < cols_; j++){
                m.data[i] += at(i, j);
            }
        }
        return m;
    }
}

Matrix Matrix::max(int axis, bool keepdims) const {
    if(axis == 0){
        Matrix m(1, cols_);
        for(int j = 0; j < cols_; j++){
            double largest = std::numeric_limits<double>::lowest();
            for(int i = 0; i < rows_; i++){
                if(at(i, j) > largest){
                    largest = at(i, j);
                    m.data[j] = largest;
                }
            }
        }
        return m;
    }

    else{
        Matrix m(rows_, 1);
        for(int i = 0; i < rows_; i++){
            double largest = std::numeric_limits<double>::lowest();
            for(int j = 0; j < cols_; j++){
                if(at(i, j) > largest){
                    largest = at(i, j);
                    m.data[i] = largest;
                }
            }
        }
        return m;
    }
}

int Matrix::getRows() const{
    return rows_;
}

int Matrix::getCols() const{
    return cols_;
}

const std::vector<double> Matrix::getData() const{
    return data;
}

Matrix Matrix::multiply(double scalar) const{
    Matrix m(rows_, cols_);
    for(int i = 0; i < rows_*cols_; i++){
        m.data[i] = data[i] * scalar;
    }
    return m;
}

Matrix Matrix::divide(double scalar) const{
    if(scalar == 0.0){
        throw std::invalid_argument("division by zero");
    }
    Matrix m(rows_, cols_);
    for(int i = 0; i < rows_*cols_; i++){
        m.data[i] = data[i] / scalar;
    }
    return m;
}

Matrix Matrix::subtract(double scalar) const{
    Matrix m(rows_, cols_);
    for(int i = 0; i < rows_*cols_; i++){
        m.data[i] = data[i] - scalar;
    }
    return m;
}

Matrix Matrix::exp() const{
    Matrix m(rows_, cols_);
    for(int i = 0; i < rows_*cols_; i++){
        m.data[i] = std::exp(data[i]);
    }
    return m;
}


Matrix Matrix::copy() const{
    return *this;
}

Matrix Matrix::clip(double min, double max) const{
    Matrix m(rows_, cols_);
    for(int i = 0; i < rows_*cols_; i++){
        if(data[i] < min){
            m.data[i] = min;
        }
        else if(data[i] > max){
            m.data[i] = max;
        }
        else{
            m.data[i] = data[i];
        }
    }
    return m;
}

Matrix Matrix::multiply(const Matrix& other) const{
    if(cols_ != other.cols_){
        throw std::invalid_argument("shape mismatch for element wise multiplication");
    }
    Matrix m(rows_, cols_);
    for(int i = 0; i < rows_*cols_; i++){
        m.data[i] = data[i] * other.data[i];
    }
    return m;
}

Matrix Matrix::argmax(int axis) const {
    if(axis == 0){
        Matrix m(1, cols_);
        for(int j = 0; j < cols_; j++){
            double largest = std::numeric_limits<double>::lowest();
            for(int i = 0; i < rows_; i++){
                if(at(i, j) > largest){
                    largest = at(i, j);
                    m.data[j] = i; //store row index
                }
            }
        }
        return m;
    }

    else{
        Matrix m(rows_, 1);
        for(int i = 0; i < rows_; i++){
            double largest = std::numeric_limits<double>::lowest();
            for(int j = 0; j < cols_; j++){
                if(at(i, j) > largest){
                    largest = at(i, j);
                    m.data[i] = j; //store column index
                }
            }
        }
        return m;
    }
}

void Matrix::print() const{
    for(int i = 0; i < rows_; i++){
        for(int j = 0; j < cols_; j++){
            std::cout << data[i * cols_ + j] << " ";
        }
        std::cout << "" << std::endl;
    }
}