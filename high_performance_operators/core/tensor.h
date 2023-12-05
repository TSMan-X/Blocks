#include <iostream>
#include <vector>
#include <cassert>
#include <initializer_list>

template<class T>
struct tensor {
public:
    tensor();
    tensor(std::vector<int> shape, T fill);
    tensor(const tensor<T> & other);
    tensor(T * data, std::vector<int> shape, std::vector<int> stride, int size);

    void lazy_init(std::vector<int> shape, T fill);
    void lazy_init(const tensor<T> & other);
    void lazy_init(T * data, std::vector<int> shape, std::vector<int> stride, int size);

    T * get_data();
    bool empty();
    T & operator[](const std::vector<int> index);
    T & operator[](const std::initializer_list<int> index);

    std::vector<int> shape;

private:
    std::vector<int> stride;
    int size;
    T * data;
};

template<class T>
struct matrix_2d {
public:
    matrix_2d();
    matrix_2d(std::vector<std::vector<type>> data);
    matrix_2d(int high, int width, type fill = 0);
    matrix_2d(matrix_2d<type> & other);

    void lazy_init(std::vector<std::vector<type>> data);
    void lazy_init(int high, int width, type fill = 0);
    void lazy_init(matrix_2d<type> & other);

    type * data_ptr();
    type * operator[](int i);
    friend std::ostream & operator<<(std::ostream & o, matrix_2d<type> & m);

    int h;
    int w;

private:
    int stride[2];
    type * data;
};
