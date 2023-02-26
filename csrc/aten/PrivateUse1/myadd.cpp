#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/extension.h>
#include <torch/script.h>

/*
Generally, speaking, the structure of your registrations will look like this:

1. A single TORCH_LIBRARY that lists every custom operator in your namespace in a centralized place.

2. A TORCH_LIBRARY_IMPL per dispatch key that registers implementations for that key (e.g., CPU or CUDA). If you like, you can further subdivide TORCH_LIBRARY_IMPL blocks into a block per operator. This is convenient if you have a separate file per operator implementation, but don’t want to expose the operators in a header; you can just put the registration in the cpp file that defines your operator.
*/



// we need to actually provide some implementations of this operator. For concreteness, here is a really simple implementation of addition on CPU

Tensor myadd_cpu(const Tensor& self_, const Tensor& other_) {
  TORCH_CHECK(self_.sizes() == other_.sizes());
  TORCH_INTERNAL_ASSERT(self_.device().type() == DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(other_.device().type() == DeviceType::CPU);
  Tensor self = self_.contiguous();
  Tensor other = other_.contiguous();
  Tensor result = torch::empty(self.sizes(), self.options());
  const float* self_ptr = self.data_ptr<float>();
  const float* other_ptr = other.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i];
  }
  return result;
}

// we just provide a schema string specifying the type signature of the operator that all of our other kernels will abide by

TORCH_LIBRARY(myops, m) {
  m.def("myadd(Tensor self, Tensor other) -> Tensor");
}

// the simple way of registering it (def("myadd", myadd_cpu)) would register the kernel to run in all cases, even if the tensor is not a CPU tensor! (Internally, we refer to these as “catch-all” kernels, since they catch all cases.) To ensure that myadd_cpu is only run for CPU tensors, we can use the TORCH_LIBRARY_IMPL macro

TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("myadd", myadd_cpu);
}

/*
TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("myadd", myadd_cuda);
}
*/
