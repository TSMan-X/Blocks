#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor warp_perspective(torch::Tensor image, torch::Tensor warp) {
  // BEGIN image_mat
  cv::Mat image_mat(/*rows=*/image.size(0),
                    /*cols=*/image.size(1),
                    /*type=*/CV_32FC1,
                    /*data=*/image.data_ptr<float>());
  // END image_mat

  // BEGIN warp_mat
  cv::Mat warp_mat(/*rows=*/warp.size(0),
                   /*cols=*/warp.size(1),
                   /*type=*/CV_32FC1,
                   /*data=*/warp.data_ptr<float>());
  // END warp_mat

  // BEGIN output_mat
  cv::Mat output_mat;
  cv::warpPerspective(image_mat, output_mat, warp_mat, /*dsize=*/{8, 8});
  // END output_mat

  // BEGIN output_tensor
  torch::Tensor output = torch::from_blob(output_mat.ptr<float>(), /*sizes=*/{8, 8});
  return output.clone();
  // END output_tensor
}

/*
somewhere at the top level of our op.cpp file. The TORCH_LIBRARY macro creates a function that will be called when your program starts. The name of your library (my_ops) is given as the first argument (it should not be in quotes). The second argument (m) defines a variable of type torch::Library which is the main interface to register your operators. The method Library::def actually creates an operator named warp_perspective, exposing it to both Python and TorchScript. You can define as many operators as you like by making multiple calls to def.
*/

TORCH_LIBRARY(my_ops, m) {
  m.def("warp_perspective", warp_perspective);
}
