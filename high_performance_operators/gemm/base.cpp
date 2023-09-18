#include <iostream>
#include <vector>
#include <cassert>
#include <sycl/sycl.hpp>

template<class type>
struct matrix_2d {
    matrix_2d(std::vector<std::vector<type>> &data) {
        this->data = data;
        h = data.size();
        w = data[0].size();
    }

    std::vector<type> & operator[](int i) {
        return data[i];
    }

    friend std::ostream & operator<<(std::ostream & o, matrix_2d<type> & m) {
        for (int i = 0; i < m.h; ++i) {
            for (int j = 0; j < m.w; ++j) o << m[i][j] << " ";
            o << std::endl;
        }
        return o;
    }

    int h;
    int w;
    std::vector<std::vector<type>> data;
};

template<class type>
matrix_2d<type> & gemm2d(matrix_2d<type> & a, matrix_2d<type> & b, matrix_2d<type> & result) {
    assert(a.w == b.h);
    assert(result.w == a.w && result.h == b.h);

    for (int i = 0; i < result.h; ++i) {
        for (int j = 0; j < result.w; ++j) {
            type tmp = 0;
            for (int k = 0; k < a.w; ++k) {
                tmp += a[i][k] * b[k][j];
            }
            result[i][j] = tmp;
        }
    }
    return result;
};

int main() {
    std::vector<std::vector<float>> tmp(2, std::vector<float>{1, 1});
    auto a = matrix_2d<float>(tmp);
    auto b = a;
    auto c = a;

    gemm2d<float>(a, b, c);
    std::cout << a << b << c << std::endl;

    return 0;
}
