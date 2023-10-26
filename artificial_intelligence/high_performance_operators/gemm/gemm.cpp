#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>

template<class type>
struct matrix_2d {
    matrix_2d(std::vector<std::vector<type>> &data) {
        this->data = data;
        h = data.size();
        w = data[0].size();
    }
    int h;
    int w;
    std::vector<std::vector<type>> data;
};

template<class type>
type & inner_product() {
}

template<class type>
matrix_2d<type> & gemm2d(matrix_2d<type> & a, matrix_2d<type> & b, matrix_2d<type> & result) {

};

int main() {
    std::vector<std::vector<float>> tmp(2, std::vector<float>{1, 1});
    auto a = matrix_2d<float>(tmp);
    std::cout << a.h << std::endl;
    return 0;
}
