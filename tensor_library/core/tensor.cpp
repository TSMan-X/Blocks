#include <iostream>
#include <vector>
#include "tensor.h"

// tensor
template<class T>
tensor<T>::tensor() {
  this->date = nullptr;
}

template<class T>
tensor<T>::tensor(std::vector<int> shape, T fill) {
  this->shape = shape;
  (this->stride).resize(shape.size(), 1);

  this->size = shape[0];
  for (int i = shape.size() - 1; i > 0; --i) {
    stride[i - 1] = stride[i] * shape[i];  
    this->size *= shape[i];
  }

  this->data = (T *)malloc(this->size * sizeof(T));
  std::fill(this->data, this->data + size, fill);
}

template<class T>
tensor<T>::tensor(const tensor<T> & other) {
  this->shape = other.shape;
  this->stride = other.stride;
  this->size = other.size;

  this->data = (T *)malloc(size * sizeof(T));
  memcpy(this->data, other.data(), size);
}

template<class T>
tensor<T>::tensor(T * data, std::vector<int> shape, std::vector<int> stride, int size) {
  this->shape = shape;
  this->stride = stride;
  this->size = size;
  this->data = data;
}

template<class T>
void tensor<T>::lazy_init(std::vector<int> shape, T fill) {
  this->shape = shape;
  (this->stride).resize(shape.size(), 1);

  this->size = shape[0];
  for (int i = shape.size() - 1; i > 0; --i) {
    stride[i - 1] = stride[i] * shape[i];  
    this->size *= shape[i];
  }

  this->data = (T *)malloc(this->size * sizeof(T));
  std::fill(this->data, this->data + size, fill);
}

template<class T>
void tensor<T>::lazy_init(const tensor<T> & other) {
  this->shape = other.shape;
  this->stride = other.stride;
  this->size = other.size;

  this->data = (T *)malloc(size * sizeof(T));
  memcpy(this->data, other.data(), size);
}

template<class T>
void tensor<T>::lazy_init(T * data, std::vector<int> shape, std::vector<int> stride, int size) {
  this->shape = shape;
  this->stride = stride;
  this->size = size;
  this->data = data;
}

template<class T>
T * tensor<T>::get_data() {
  return this->data;
}

template<class T>
bool tensor<T>::empty() {
  return (this->data == nullptr);
}

template<class T>
T & tensor<T>::operator[](const std::vector<int> index) {
  assert(index.size() == (this->stride).size() && "Index length is not correct!");
  int position = 1;
  for (int i = 1; i < index.size(); ++i) {
    position += (index[i] * stride[i]);
  }

  return *((this->data) + position);
}

template<class T>
T & tensor<T>::operator[](const std::initializer_list<int> index) {
  assert(index.size() == (this->stride).size() && "Index length is not correct!");
  int position = 1;
  int j = 0;
  for (auto i:index) {
    position += (i * stride[j++]);
  }

  return *((this->data) + position);
}

// matrix_2d
template<class T>
matrix_2d<T>::matrix_2d() {
  h = 0;
  w = 0;
  stride[0] = 0;
  stride[1] = 0;
  this->date = nullptr;
}

template<class T>
matrix_2d<T>::matrix_2d(std::vector<std::vector<type>> data) {
  h = data.size();
  w = data[0].size();
  stride[0] = w;
  stride[1] = 1;
  
  this->data = (type *)malloc(h * w * sizeof(type));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      (this->data)[i * stride[0] + j * stride[1]] = data[i][j];  
    }
  }
}

template<class T>
matrix_2d<T>::matrix_2d(int high, int width, type fill = 0) {
  h = high;
  w = width;
  stride[0] = w;
  stride[1] = 1;

  this->data = (type *)malloc(h * w * sizeof(type));
  std::fill(this->data, this->data + h * w, fill);
}

template<class T>
matrix_2d<T>::matrix_2d(matrix_2d<type> & other) {
  h = other.h;
  w = other.w;
  stride[0] = w;
  stride[1] = 1;

  this->data = (type *)malloc(h * w * sizeof(type));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      (this->data)[i * stride[0] + j * stride[1]] = other[i][j];  
    }
  }
}

template<class T>
void matrix_2d<T>::lazy_init(std::vector<std::vector<type>> data) {
  h = data.size();
  w = data[0].size();
  stride[0] = w;
  stride[1] = 1;
  
  type * old_data = this->data;
  delete old_data;

  this->data = (type *)malloc(h * w * sizeof(type));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      (this->data)[i * stride[0] + j * stride[1]] = data[i][j];  
    }
  }
}

template<class T>
void matrix_2d<T>::lazy_init(int high, int width, type fill = 0) {
  h = high;
  w = width;
  stride[0] = w;
  stride[1] = 1;

  type * old_data = this->data;
  delete old_data;

  this->data = (type *)malloc(h * w * sizeof(type));
  std::fill(this->data, this->data + h * w, fill);
}

template<class T>
void matrix_2d<T>::lazy_init(matrix_2d<type> & other) {
  h = other.h;
  w = other.w;
  stride[0] = w;
  stride[1] = 1;

  type * old_data = this->data;
  delete old_data;

  this->data = (type *)malloc(h * w * sizeof(type));
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      (this->data)[i * stride[0] + j * stride[1]] = other[i][j];  
    }
  }
}

template<class T>
type * matrix_2d<T>::data_ptr() {
  return data;
}

template<class T>
type * matrix_2d<T>::operator[](int i) {
  return (this->data) + i * stride[0];
}

template<class T>
friend std::ostream & matrix_2d<T>::operator<<(std::ostream & o, matrix_2d<type> & m) {
  for (int i = 0; i < m.h; ++i) {
      for (int j = 0; j < m.w; ++j) o << m[i][j] << " ";
      o << std::endl;
  }
  return o;
}
