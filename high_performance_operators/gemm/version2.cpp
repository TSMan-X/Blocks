#include <iostream>
#include <vector>
#include <cassert>
#include <sycl/sycl.hpp>

// Memory bandwidth

// one should always start with memory before optimizing arithmetic, as it is much more likely to be the bottleneck.

template<class type>
struct matrix_2d {
    matrix_2d(std::vector<std::vector<type>> &data) {
        this->data = data;
        h = data.size();
        w = data[0].size();
    }

    matrix_2d(int high, int width, type fill = 0) {
        data.resize(high, std::vector<type>(width, fill));
        h = high;
        w = width;
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
void transposition(matrix_2d<type> & old, matrix_2d<type> & trans) {
    assert(old.h == trans.w && old.w == trans.h);

    for (int i = 0; i < old.h; ++i) {
        for (int j = 0; j < old.w; ++j) {
            trans[j][i] = old[i][j];
        }
    }
    return;
}

template<class type>
matrix_2d<type> & gemm2d_version1(matrix_2d<type> & a, matrix_2d<type> & b, matrix_2d<type> & result) {
    assert(a.w == b.h);
    assert(result.h == a.h && result.w == b.w);

    auto trans_b = matrix_2d<type>(b.w, b.h);
    transposition<type>(b, trans_b);

    for (int i = 0; i < result.h; ++i) {
        for (int j = 0; j < result.w; ++j) {
            type tmp = 0;
            for (int k = 0; k < a.w; ++k) {
                tmp += a[i][k] * trans_b[j][k];
            }
            result[i][j] = tmp;
        }
    }
    return result;
};

int main() {
    auto a = matrix_2d<float>(4, 3, 2);
    auto b = matrix_2d<float>(3, 5, 1.5);
    auto c = matrix_2d<float>(4, 5);

    gemm2d_version1<float>(a, b, c);
    std::cout << a << b << c << std::endl;

    return 0;
}
