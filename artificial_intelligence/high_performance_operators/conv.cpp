#include <iostream>
#include <vector>

std::vector<std::vector<float> >& convolution_2d(std::vector<std::vector<float> > input, std::vector<std::vector<float> > weight) {
    int h_input = input.size();
    int w_input = input[0].size();
    int h_weight = weight.size();
    int w_weight = weight[0].size();
    int w_pad = (w_weight / 2);
    int h_pad = (h_weight / 2)
    int h_output = h_input - 2 * h_pad;
    int w_output = w_input - 2 * w_pad;

    std::vector<std::vector<float> > output(h_output, std::vector<float>(w_output, 0));
    for (int i(0); i < h_output; ++i) {
        for (int j(0); j < w_output; ++j) {
            // filter
            for (int wi(0); wi < h_weight; ++wi) {
                for (int wj(0); wj < w_weight; ++wj) {
                    output[i][j] += weight[wi][wj] * input[wi + i][wj + j];
                }
            }
        }
    }
}

int main() {
}
