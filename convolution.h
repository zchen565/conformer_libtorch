#pragma once
#include <torch/torch.h>

#include "basic.h"
class DepthwiseConv1dImpl : public torch::nn::Module {
public:
    DepthwiseConv1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
                        int64_t stride = 1, int64_t padding = 0, bool bias = false) {
        // Ensure the out_channels is a multiple of in_channels
        TORCH_CHECK(out_channels % in_channels == 0, "out_channels should be a constant multiple of in_channels");

        conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, kernel_size)
                                 .stride(stride)
                                 .padding(padding)
                                 .bias(bias)
                                 .groups(in_channels));
        register_module("conv", conv);
    }

    torch::Tensor forward(torch::Tensor x) {
        return conv->forward(x);
    }
private:
    torch::nn::Conv1d conv{nullptr};
};
TORCH_MODULE(DepthwiseConv1d); // Wrapper for shared pointer

class PointwiseConv1dImpl : public torch::nn::Module {
public:
    PointwiseConv1dImpl(int64_t in_channels, int64_t out_channels, 
                        int64_t stride = 1, int64_t padding = 0, bool bias = true) {
        conv = torch::nn::Conv1d(torch::nn::Conv1dOptions(in_channels, out_channels, 1)
                                 .stride(stride)
                                 .padding(padding)
                                 .bias(bias));
        register_module("conv", conv);
    }

    torch::Tensor forward(torch::Tensor x) {
        return conv->forward(x);
    }
private:
    torch::nn::Conv1d conv{nullptr};
};
TORCH_MODULE(PointwiseConv1d); //


class ConformerConvModuleImpl : public torch::nn::Module {
private:
    torch::nn::LayerNorm layer_norm{nullptr};
    PointwiseConv1d pointwise_conv1;
    GLU glu;
    DepthwiseConv1d depthwise_conv;
    torch::nn::BatchNorm1d batch_norm;
    SILU silu;
    PointwiseConv1d pointwise_conv2;
    torch::nn::Dropout dropout;
    Transpose transpose_;

public:
    ConformerConvModuleImpl(int64_t in_channels, int64_t kernel_size = 31, 
                            float dropout_p = 0.1, int64_t expansion_factor = 2)
    : layer_norm(torch::nn::LayerNormOptions({in_channels})),
      pointwise_conv1(in_channels, in_channels * expansion_factor, 1),
      glu(1), 
      depthwise_conv(in_channels, in_channels, kernel_size),
      batch_norm(in_channels),
      silu(), 
      pointwise_conv2(in_channels, in_channels, 1),
      dropout(dropout_p),
      transpose_(1,2) {
        // Check conditions
        TORCH_CHECK((kernel_size - 1) % 2 == 0, "kernel_size should be an odd number for 'SAME' padding");
        TORCH_CHECK(expansion_factor == 2, "Currently, only supports expansion_factor 2");

        // Register modules
        register_module("layer_norm", layer_norm);
        register_module("transpose", transpose_);
        register_module("pointwise_conv1", pointwise_conv1);
        register_module("glu", glu);
        register_module("depthwise_conv", depthwise_conv);
        register_module("batch_norm", batch_norm);
        register_module("silu", silu);
        register_module("pointwise_conv2", pointwise_conv2);
        register_module("dropout", dropout);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = layer_norm->forward(x);
        // std::cout << "line 90 : " << x.sizes() << std::endl;
        x = transpose_->forward(x);
        // std::cout << "line 92 : " << x.sizes() << std::endl;
        x = pointwise_conv1->forward(x);
        // std::cout << "line 94 : " << x.sizes() << std::endl;
        x = glu->forward(x);
        // std::cout << "line 96 : " << x.sizes() << std::endl;
        x = depthwise_conv->forward(x);
        std::cout << "line 98 : " << x.sizes() << std::endl;
        x = batch_norm->forward(x);
        // std::cout << "line 100 : " << x.sizes() << std::endl;
        x = silu->forward(x);
        // std::cout << "line 102 : " << x.sizes() << std::endl;
        x = pointwise_conv2->forward(x);
        // std::cout << "line 104 : " << x.sizes() << std::endl;
        x = dropout->forward(x);
        // std::cout << "line 98 : " << x.sizes() << std::endl;
        return x.transpose(1, 2);
    }
};
TORCH_MODULE(ConformerConvModule);


class Conv2dSubsamplingImpl : public torch::nn::Module {
public:
    Conv2dSubsamplingImpl(int64_t in_channels, int64_t out_channels)
    : conv1(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(2)),
      conv2(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(2)),
      relu() {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("relu", relu);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor inputs, torch::Tensor input_lengths) {
        inputs = inputs.unsqueeze(1);
        inputs = relu(conv1->forward(inputs));
        inputs = relu(conv2->forward(inputs));

        auto batch_size = inputs.size(0);
        auto channels = inputs.size(1);
        auto subsampled_lengths = inputs.size(2);
        auto sumsampled_dim = inputs.size(3);

        inputs = inputs.permute({0, 2, 1, 3}).contiguous().view({batch_size, subsampled_lengths, channels * sumsampled_dim});

        torch::Tensor output_lengths = input_lengths.div(4, "floor").sub(1);

        return std::make_tuple(inputs, output_lengths);
    }
private:
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};
    torch::nn::ReLU relu;
};

TORCH_MODULE(Conv2dSubsampling);