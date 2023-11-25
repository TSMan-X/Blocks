#include <iostream>
#include "tensor.h"

template<class type>
struct matrix_2d {
public:
    matrix_2d() {}

    matrix_2d(std::vector<std::vector<type>> data) {}

    matrix_2d(int high, int width, type fill = 0) {}

    matrix_2d(matrix_2d<type> & other) {}

    void lazy_init(std::vector<std::vector<type>> data) {}

    void lazy_init(int high, int width, type fill = 0) {}

    void lazy_init(matrix_2d<type> & other) {}

    type * data_ptr() {}

    type * operator[](int i) {}

    friend std::ostream & operator<<(std::ostream & o, matrix_2d<type> & m) {}

    int h;
    int w;

private:
    int stride[2];
    type * data;
};




