#include <iostream>
#include "matrix.hpp"

int main() {
    std::cout << "--- zeros(3,3) ---" << std::endl;
    Matrix z = Matrix::zeros(3, 3);
    z.print();

    std::cout << "--- ones(3,3) ---" << std::endl;
    Matrix o = Matrix::ones(3, 3);
    o.print();

    std::cout << "--- randn(3,3) ---" << std::endl;
    Matrix r = Matrix::randn(3, 3);
    r.print();

    std::cout << "--- eye(4) ---" << std::endl;
    Matrix e = Matrix::eye(4);
    e.print();

    std::cout << "--- dot test ---" << std::endl;
    Matrix a = Matrix::ones(2, 3);
    Matrix b = Matrix::ones(3, 2);
    a.dot(b).print();

    std::cout << "--- add with broadcasting ---" << std::endl;
    Matrix x = Matrix::ones(3, 3).multiply(2.0);
    Matrix bias = Matrix::ones(1, 3);
    x.add(bias).print();

    std::cout << "--- transpose ---" << std::endl;
    Matrix t = Matrix::randn(2, 3);
    std::cout << "original:" << std::endl;
    t.print();
    std::cout << "transposed:" << std::endl;
    t.transpose().print();

    std::cout << "--- sum axis=0 ---" << std::endl;
    Matrix s = Matrix::ones(3, 3).multiply(2.0);
    s.sum(0).print();

    std::cout << "--- sum axis=1 ---" << std::endl;
    s.sum(1).print();

    std::cout << "--- clip ---" << std::endl;
    Matrix c = Matrix::randn(3, 3);
    c.print();
    c.clip(-0.5, 0.5).print();

    std::cout << "--- argmax axis=1 ---" << std::endl;
    Matrix am = Matrix::randn(3, 4);
    am.print();
    am.argmax(1).print();

    return 0;
}