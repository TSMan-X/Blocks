#include <iostream>
#include <vector>
#include <cassert>
#include <limits>

#include <sycl/sycl.hpp>

// Sycl version

// Maximize the parallization of the computing.

template<class type>
struct matrix_2d {

public:
    matrix_2d(std::vector<std::vector<type>> data) {
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

    matrix_2d(int high, int width, type fill = 0) {
      h = high;
      w = width;
      stride[0] = w;
      stride[1] = 1;

      this->data = (type *)malloc(h * w * sizeof(type));
      std::fill(this->data, this->data + h * w, fill);
    }

    type * data_ptr() {
      return data;
    }

    type * operator[](int i) {
      return (this->data) + i * stride[0];
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

private:
    int stride[2];
    type * data;
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
matrix_2d<type> & gemm2d_version2(matrix_2d<type> & a, matrix_2d<type> & b, matrix_2d<type> & result) {
    assert(a.w == b.h);
    assert(result.h == a.h && result.w == b.w);

    try {
      sycl::queue q(sycl::default_selector_v);
      std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

      sycl::buffer a_buf(a.data_ptr(), sycl::range(a.h, a.w));
      sycl::buffer b_buf(b.data_ptr(), sycl::range(b.h, b.w));
      sycl::buffer re_buf(result.data_ptr(), sycl::range(result.h, result.w));
      
      q.submit([&](auto &h) {
        sycl::accessor a(a_buf, h, sycl::read_only);  
        sycl::accessor b(b_buf, h, sycl::read_only); 
        sycl::accessor re(re_buf, h, sycl::write_only);

        int a_w = a.get_range()[1];

        h.parallel_for(sycl::range(re.get_range()[0], re.get_range()[1]), [=](auto index){
          int row_number = index[0];
          int col_number = index[1];

          type sum = 0;

          for (int i = 0; i < a_w; ++i) {
            sum += a[row_number][i] * b[i][col_number];
          }

          re[row_number][col_number] = sum;
        });

      });

    } catch (sycl::exception const & e){
      std::cout << "Sycl level error happens!" << std::endl;
      std::terminate();
    }

    return result;

};

int main() {
    auto a = matrix_2d<float>(4, 3, 2);
    auto b = matrix_2d<float>(3, 5, 1.5);
    auto c = matrix_2d<float>(4, 5);

    gemm2d_version2<float>(a, b, c);
    std::cout << a << b << c << std::endl;

    return 0;
}
