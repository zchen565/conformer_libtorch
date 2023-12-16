#pragma once

#include <torch/torch.h>

using namespace torch::indexing;
class PositionalEncodingImpl : public torch::nn::Module {
public:
    PositionalEncodingImpl(int64_t d_model = 512, int64_t max_len = 1600) {
        auto pe = torch::zeros({max_len, d_model}); // no need for grad
        auto position = torch::arange(0, max_len, torch::kFloat).unsqueeze(1);
        auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::kFloat) * (-std::log(10000.0) / d_model));
        pe.index_put_({Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
        pe.index_put_({Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));
        pe = pe.unsqueeze(0);
        std::cout << pe.sizes() << std::endl;
        register_buffer("pe", pe);
        std::cout << pe.device() << std::endl;
    }

    torch::Tensor forward(int64_t length) {
        std::cout << "forwarding" << std::endl;
        std::cout << pe.values() << std::endl;
        std::cout << pe.device() << std::endl;
        return pe.index({Slice(), Slice(0, length)});
    }

private:
    torch::Tensor pe;
};
TORCH_MODULE(PositionalEncoding);


// the Relative attention in Conformer
// Note : this is not the original MHSA in pytroch or libtorch, it is different in Conformer paper

class RelativeMultiHeadAttentionImpl : public torch::nn::Module {
public:
    RelativeMultiHeadAttentionImpl(int64_t d_model = 512, int64_t num_heads = 16, double dropout_p = 0.1) {
        TORCH_CHECK(d_model % num_heads == 0, "d_model % num_heads should be zero.");
        this->num_heads = num_heads;
        d_head = d_model / num_heads;
        sqrt_dim = std::sqrt(d_model);

        query_proj = torch::nn::Linear(d_model, d_model);
        key_proj = torch::nn::Linear(d_model, d_model);
        value_proj = torch::nn::Linear(d_model, d_model);
        pos_proj = torch::nn::Linear(d_model, d_model);
        pos_proj->options.bias(0); // check the api file

        dropout = torch::nn::Dropout(dropout_p);
        u_bias = register_parameter("u_bias", torch::randn({num_heads, d_head}));
        v_bias = register_parameter("v_bias", torch::randn({num_heads, d_head}));

        out_proj = torch::nn::Linear(d_model, d_model);

        register_module("query_proj", query_proj);
        register_module("key_proj", key_proj);
        register_module("value_proj", value_proj);
        register_module("pos_proj", pos_proj);
        register_module("dropout", dropout);
        register_module("out_proj", out_proj);
    }

    torch::Tensor forward(torch::Tensor query, torch::Tensor key, torch::Tensor value, torch::Tensor pos_embedding, torch::Tensor mask = torch::Tensor()) {
        auto batch_size = value.size(0);

        // Processing inputs
        query = query_proj->forward(query).view({batch_size, -1, num_heads, d_head});
        key = key_proj->forward(key).view({batch_size, -1, num_heads, d_head}).permute({0, 2, 1, 3});
        value = value_proj->forward(value).view({batch_size, -1, num_heads, d_head}).permute({0, 2, 1, 3});
        pos_embedding = pos_proj->forward(pos_embedding).view({batch_size, -1, num_heads, d_head});

        // Calculating scores
        auto content_score = torch::matmul((query + u_bias).transpose(1, 2), key.transpose(2, 3));
        auto pos_score = torch::matmul((query + v_bias).transpose(1, 2), pos_embedding.permute({0, 2, 3, 1}));
        pos_score = _relative_shift(pos_score);

        auto score = (content_score + pos_score) / sqrt_dim;

        if (mask.defined()) {
            mask = mask.unsqueeze(1);
            score.masked_fill_(mask, -1e9);
        }

        auto attn = torch::softmax(score, -1);
        attn = dropout->forward(attn);

        auto context = torch::matmul(attn, value).transpose(1, 2);
        context = context.view({batch_size, -1, d_head * num_heads});

        return out_proj->forward(context);
    }

private:
    torch::nn::Linear query_proj{nullptr}, key_proj{nullptr}, value_proj{nullptr}
                , pos_proj{nullptr}, out_proj{nullptr};
    torch::nn::Dropout dropout;
    torch::Tensor u_bias, v_bias;
    int64_t d_head, num_heads;
    double sqrt_dim;

    torch::Tensor _relative_shift(torch::Tensor pos_score) {
        auto shape = pos_score.sizes();
        auto batch_size = shape[0];
        auto num_heads = shape[1];
        auto seq_length1 = shape[2];
        auto seq_length2 = shape[3];
        auto zeros = torch::zeros({batch_size, num_heads, seq_length1, 1}, pos_score.options());
        auto padded_pos_score = torch::cat({zeros, pos_score}, -1);
        padded_pos_score = padded_pos_score.view({batch_size, num_heads, seq_length2 + 1, seq_length1});
        pos_score = padded_pos_score.slice(2, 1).view_as(pos_score);
        return pos_score;
    }
};

TORCH_MODULE(RelativeMultiHeadAttention);

class MultiHeadedSelfAttentionModuleImpl : public torch::nn::Module {
public:
    MultiHeadedSelfAttentionModuleImpl(int64_t d_model, int64_t num_heads, double dropout_p = 0.1)
        : positional_encoding(d_model),
          layer_norm(torch::nn::LayerNormOptions({d_model})),
          attention(d_model, num_heads, dropout_p),
          dropout(torch::nn::DropoutOptions(dropout_p)) {
        // layer_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model}));
        register_module("positional_encoding", positional_encoding);
        register_module("layer_norm", layer_norm);
        register_module("attention", attention);
        register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor inputs, torch::Tensor mask = torch::Tensor()) {
        int64_t batch_size = inputs.size(0);
        int64_t seq_length = inputs.size(1);
        int64_t d_model = inputs.size(2);

        torch::Tensor pos_embedding = positional_encoding(seq_length);
        pos_embedding = pos_embedding.repeat({batch_size, 1, 1});

        inputs = layer_norm->forward(inputs);
        torch::Tensor outputs = attention->forward(inputs, inputs, inputs, pos_embedding, mask);

        return dropout->forward(outputs);
    }

private:
    PositionalEncoding positional_encoding;
    torch::nn::LayerNorm layer_norm;
    RelativeMultiHeadAttention attention;
    torch::nn::Dropout dropout;
};

TORCH_MODULE(MultiHeadedSelfAttentionModule);