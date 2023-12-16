#pragma once
#include <torch/torch.h>
#include <iostream>
#include <optional>
#include <tuple>

class FeedForwardModuleImpl : public torch::nn::Module {
public:
    FeedForwardModuleImpl(int64_t input_dim = 512, int64_t hidden_dim = 4, double dropout = 0.1) {
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));
        linear1 = register_module("linear1", torch::nn::Linear(input_dim, hidden_dim));
        dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
        linear2 = register_module("linear2", torch::nn::Linear(hidden_dim, input_dim));
        dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
    }

    torch::Tensor forward(torch::Tensor input) {
        input = layer_norm->forward(input);
        input = linear1->forward(input);
        input = torch::silu(input);
        input = dropout1->forward(input);
        input = linear2->forward(input);
        input = dropout2->forward(input);
        return input;
    }

private:
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Linear linear1{nullptr}, linear2{nullptr};
    torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
};

TORCH_MODULE(FeedForwardModule); // for further use