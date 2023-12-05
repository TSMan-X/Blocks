#include <iostream>
#include <limits>
#include "all.h"

template<class T>
struct WordPieceTokenlizer {
public:
  WordPieceTokenlizer() = default;

  template<class T>
  tensor<T> encoder(const std::string & input) {
    tensor<T> token(std::vector{input.size() + 2}, 0);
    token[{0}] = static_cast<T>('^');
    for (int i = 0; i < input.size(); ++i) {
      token[{i + 1}] = static_cast<T>(input[i]);
    }
    token[{input.size() + 1}] = static_cast<T>("$");
    return token;
  }

  template<class T>
  std::string decoder(const tensor<T> & token) {
    std::string output = "";
    for (int i = 0; i < token.size; ++i) {
      auto c = static_cast<char>(token[{i}]);
      if (c == '^') continue;
      if (c == '$') break;
      output.push_back(c);
    }
    return output;
  }

}

template<class T>
struct Enbedding {
public:
  Embedding() = default;
  Embedding(int vocal_length = 1024, int embedding_dim) {
    this->weight = new tensor<T>(std::vector({vocal_length, embedding_dim)})) // vocal_length * embedding_dim
    this->embedding_dim = embedding_dim;
    this->vocal_length = vocal_length;
  }

  tensor<T> forward(tensor<T> & token) {
    auto shape = token.shape; // batchsize * token_length
    shape.push_back(embedding_dim);
    tensor<T> output(shape, 0); // batchsize * token_length * embedding_dim

    for (int n = 0; n < shape[0]; ++n) {
      for (int i = 0; i < shape[1]; ++i) {
        memcpy(&(output[{n, i, 0}]), &weight[{token[{n, i}], 0}], embedding_dims * sizeof(T));
      }
    }

    return output;
  }

private:
  tensor<T> * weight;
  int embedding_dim;
  int vocal_length;
}

template<class T>
struct Linear {
public:
  Linear() = default;

  Linear(int inFeature, int outFeature) {
    in_feature = inFeature;
    out_feature = outFeature;
    weight.lazy_init({in_feature, out_feature}, 1);
  }

  tensor<T> forward(const tensor<T> & input) {
    tensor<T> output({input.shape[0], out_feature});

    auto in_ptr = input.get_data();
    auto out_ptr = output.get_data();
    auto wei_ptr = weight.get_data();

    hpdnn::gemm(in_ptr,
                out_ptr,
                wei_ptr,
                input.shape.data(),
                output.shape.data(),
                weight.shape.data());
    return output;
  }

  void forward(const tensor<T> & input, tensor<T> & output) {
    auto in_ptr = input.get_data();
    auto out_ptr = output.get_data();
    auto wei_ptr = weight.get_data();

    hpdnn::gemm(in_ptr,
                out_ptr,
                wei_ptr,
                input.shape.data(),
                output.shape.data(),
                weight.shape.data());
    return;
  }
private:
  int in_feature;
  int out_feature;
  tensor<T> weight;
}


template<class T>
struct MultiHeadAttention {
public:
  MultiHeadAttention(int model_dim, int head_num) {
    this->n_head = head_num;
    this->q_linear = new linear(model_dim, model_dim * head_num); 
    this->k_linear = new linear(model_dim, model_dim * head_num); 
    this->v_linear = new linear(model_dim, model_dim * head_num); 
    this->fc = new linear(model_dim * head_num, model_dim); 
  }

  tensor<T> forward(tensor<T> query, tensor<T> key, tensor<T> value, tensor<T> mask = tensor<T>()) {
    auto shape = query.shape; // batchsize * token_length * model_dim

    auto q = q_linear->forward(query); // batchsize * token_length * (model_dim * head_num)
    auto k = k_linear->forward(key);
    auto v = v_linear->forward(value);

    q.view(shape[0], -1, n_head, model_dim); // batchsize * token_length * head_num * model_dim
    k.view(shape[0], -1, n_head, model_dim);
    v.view(shape[0], -1, n_head, model_dim);

    q.permute(0, 2, 1, 3).contigous(); // batchsize * head_num * token_length * model_dim
    k.permute(0, 2, 3, 1).contigous(); // batchsize * head_num * model_dim * token_length
    v.permute(0, 2, 1, 3).contigous();

    scale = model_dim;
    auto energy = blocks::matmul(q, k) / scale; // batchsize * head_num * token_length * token_length

    if (!mask.empty()) {
      blocks::masked_fill(energy, mask, std::numeric_limits<T>::min()); // set value of energy to minimun value while corresponding position of mask has value of 1
    }

    auto attention = blocks::softmax(energy, -1); // batchsize * head_num * token_length * token_length
    auto output = blocks::matmul(attention, v); // batchsize * head_num * token_length * model_dim

    output.permute(0, 2, 1, 3); // batchsize * token_length * head_num * model_dim
    output.contigous().view(shape[0], -1, (n_head * model_dim)); // batchsize * token_length * (head_num * model_dim);

    output = fc->forward(output); // batchsize * token_length * model_dim
    return output;
  }

  void forward(tensor<T> query, tensor<T> key, tensor<T> value, tensor<T> mask = tensor<T>(), tensor<T> & output) {
    auto shape = query.shape; // batchsize * token_length * model_dim

    auto q = q_linear->forward(query); // batchsize * token_length * (model_dim * head_num)
    auto k = k_linear->forward(key);
    auto v = v_linear->forward(value);

    q.view(shape[0], -1, n_head, model_dim); // batchsize * token_length * head_num * model_dim
    k.view(shape[0], -1, n_head, model_dim);
    v.view(shape[0], -1, n_head, model_dim);

    q.permute(0, 2, 1, 3).contigous(); // batchsize * head_num * token_length * model_dim
    k.permute(0, 2, 3, 1).contigous(); // batchsize * head_num * model_dim * token_length
    v.permute(0, 2, 1, 3).contigous();

    scale = model_dim;
    auto energy = blocks::matmul(q, k) / scale; // batchsize * head_num * token_length * token_length

    if (!mask.empty()) {
      blocks::masked_fill(energy, mask, std::numeric_limits<T>::min()); // set value of energy to minimun value while corresponding position of mask has value of 1
    }

    auto attention = blocks::softmax(energy, -1); // batchsize * head_num * token_length * token_length
    output = blocks::matmul(attention, v); // batchsize * head_num * token_length * model_dim

    output.permute(0, 2, 1, 3); // batchsize * token_length * head_num * model_dim
    output.contigous().view(shape[0], -1, (n_head * model_dim)); // batchsize * token_length * (head_num * model_dim);

    output = fc->forward(output); // batchsize * token_length * model_dim
    return;
  }
  
private:
  int n_head;
  linear<T> * q_linear;
  linear<T> * k_linear;
  linear<T> * v_linear;
  linear<T> * fc;
}

template<class T> 
struct FeedForward {
public:
  FeedForward(int model_dim, int hidden_dim) {
    this->ff1 = new linear(model_dim, hidden_dim);
    this->ff1 = new linear(hidden_dim, model_dim);
  }

  tensor<T> forward(tensor<T> & input) {
    auto hidden_output = ff1->forward(input); // batchsize * token_length * hidden_dim
    auto output = ff2->forward(hidden_output); // batchsize * token_length * model_dim
    return output;
  }

  void forward(tensor<T> & input, tensor<T> & output) {
    auto hidden_output = ff1->forward(input); // batchsize * token_length * hidden_dim
    ff2->forward(hidden_output, output); // batchsize * token_length * model_dim
    return;
  }

private:
  linear<T> * ff1;
  linear<T> * ff2;
}

template<class T> 
struct LayerNorm {
public:
  LayerNorm(T ga, T be):gamma(ga), beta(be) {};
  tensor<T> forward(tensor<T> input) {}
private:
  T gamma;
  T beta;
}

template<class T>
struct EncoderBlock {
public:
  EncoderBlock(int model_dim, int head_num, int hidden_dim) {
    this->mha = new MultiHeadAttention(model_dim, head_num);
    this->mha_norm = new LayerNorm(1, 0);
    this->ff = new FeedForward(model_dim, hidden_dim);
    this->ff_norm = new LayerNorm(1, 0);
  }

  tensor<T> forward(tesnor<T> input, tensor<T> src_mask = tensor<T>()) {
    auto mid_res = this->mha->forward(input, input, input, src_mask); // batchsize * token_length * model_dim
    input = this->mha_norm->forward(input + mid_res); // batchsize * token_length * model_dim

    mid_res = this->ff->forward(input); // batchsize * token_length * model_dim
    input = this->ff_norm->forward(input + mid_res); // batchsize * token_length * model_dim
    return input;
  }
private:
  MultiHeadAttention * mha;
  LayerNorm * mha_norm;
  FeedForward * ff;
  LayerNorm * ff_norm;
}


template<class T>
struct Encoder {
public:
  Encoder(int model_dim, int head_num, int hidden_dim, int block_num) {
    this->block_num = block_num;
    this->encode_embedding = new Embedding(1024, model_dim);
    for (int i = 0; i < block_num) {
      encoder_blocks.push_back(EncoderBlock(model_dim, head_num, hidden_dim));
    } 
  }

  tensor<T> forward(tesnor<T> input, tensor<T> src_mask = tensor<T>()) {
    // input: batchsize * token_length
    input = encode_embedding(input);
    for(int i = 0; i < block_num; i++) {
      input = encoder_blockers[i].forward(input, src_mask); // batchsize * token_length * model_dim
    }
    return input;
  }
private:
  Embedding * encode_embedding;
  std::vector<EncoderBlock> encoder_blocks;
  int block_num;
}

template<class T>
struct DecoderBlock {
public:
  DecoderBlock(int model_dim, int head_num, int hidden_dim, int block_num) {
    this->decoder_mha = new MultiHeadAttention(model_dim, head_num);
    this->dm_norm = new LayerNorm(1, 0);
    this->encoder_mha = new MultiHeadAttention(model_dim, head_num);
    this->em_norm = new LayerNorm(1, 0);
    this->ff = new FeedForward(model_dim, hidden_dim);
    this->ff_norm = new LayerNorm(1, 0);
  }

  tensor<T> forward(tensor<T> target, tensor<T> encoded_src, tensor<T> src_mask, tensor<T> tar_mask) {
    auto mid_res = this->decoder_mha->forward(target, target, target, tar_mask); // batchsize * token_length * model_dim
    target = this->dm_norm->forward(target + mid_res); // batchsize * token_length * model_dim

    mid_res = this->encoder_mha->forward(target, encoder_src, encoder_src, src_mask); // batchsize * token_length * model_dim
    target = this->em_norm->forward(target + mid_res); // batchsize * token_length * model_dim
    
    mid_res = this->ff->forward(target); // batchsize * token_length * model_dim
    target = this->ff_norm->forward(target + mid_res); // batchsize * token_length * model_dim
    return target;
  }

private:
  MultiHeadAttention * decoder_mha;
  LayerNorm * dm_norm;
  MultiHeadAttention * encoder_mha;
  LayerNorm * em_norm;
  FeedForward * ff;
  LayerNorm * ff_norm;
}

template<class T>
struct Decoder {
public:
  Decoder(int model_dim, int head_num, int hidden_dim, int block_num) {
    this->block_num = block_num;
    this->decode_embedding = new Embedding(1024, model_dim);
    for (int i = 0; i < block_num) {
      decoder_blocks.push_back(DecoderBlock(model_dim, head_num, hidden_dim));
    } 
  }

  tensor<T> forward(tensor<T> target, tensor<T> encoded_src, tensor<T> src_mask, tensor<T> tar_mask) {
    // target: batchsize * token_length
    target = decode_embedding(target);
    for(int i = 0; i < block_num; i++) {
      target = decoder_blockers[i].forward(target, encoded_src, src_mask, tar_mask); // batchsize * token_length * model_dim
    }
    return input;
  }
private:
  Embedding * decode_embedding;
  std::vector<DecoderBlock> decoder_blocks;
  int block_num;
}


template<class T>
struct TransformerTraining {
public:
  TransformerTraining(int model_dim, int head_num, int hidden_dim, int block_num) {
    this->model_dim = model_dim;
    this->head_num = head_num;
    this->hidden_dim = hidden_dim;
    this->block_num = block_num;

    this->encoder = new encoder(model_dim, head_num, hidden_dim, block_num);
    this->decoder = new decoder(model_dim, head_num, hidden_dim, block_num);
  }

  tensor<T> forward(tensor<T> src, tensor<T> tar, tensor<T> src_mask, tensor<T> tar_mask) {
    // src: batchsize * token_length
    // tar: batchsize * token_length
    // src_mask: token_length * token_length
    // tar_mask: token_length * token_length

    auto encoded_src = this->encoder->forward(src, src_mask); // batchsize * token_length * model_dim
    auto output = this->decoder->forward(tar, encoded_src, src_mask, tar_mask); // batchsize * token_length * model_dim

    return output;
  }

private:
  Encoder * encoder;
  Decoder * decoder;
  int model_dim
  int head_num
  int hidden_dim
  int block_num
}

int main() {
}
