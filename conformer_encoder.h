/**
 * @file conformer_encoder.h
 * this file containes ConformerBlock and ConformerEncoder
 * @note the encoder contains the convolution subsampling
 * @note decoder is in another file
 * @note the ast model is in conformer_asr
 */
#pragma once
#include <torch/torch.h>
#include "basic.h"
#include "convolution.h"
#include "attention.h"
#include "feedforward.h"
// Conformer Block

class ConformerBlockImpl : public torch::nn::Module {
public:
    ConformerBlockImpl(
        int64_t encoder_dim = 512,
        int64_t num_attention_heads = 8,
        int64_t feed_forward_expansion_factor = 4,
        int64_t conv_expansion_factor = 2,
        double feed_forward_dropout_p = 0.1,
        double attention_dropout_p = 0.1,
        double conv_dropout_p = 0.1,
        int64_t conv_kernel_size = 31,
        bool half_step_residual = true
    ) {
        if (half_step_residual) {
            feed_forward_residual_factor = 0.5;
        } else {
            feed_forward_residual_factor = 1;
        }

        sequential = torch::nn::Sequential(
            ResidualConnectionModule(
                torch::nn::AnyModule(
                    FeedForwardModule(
                        encoder_dim,
                        feed_forward_expansion_factor,
                        feed_forward_dropout_p
                    )
                ),
                feed_forward_residual_factor
            ),
            ResidualConnectionModule(
                torch::nn::AnyModule(
                    MultiHeadedSelfAttentionModule(
                        encoder_dim,
                        num_attention_heads,
                        attention_dropout_p
                    )
                )
            ),
            ResidualConnectionModule(
                torch::nn::AnyModule(
                    ConformerConvModule(
                        encoder_dim,
                        conv_kernel_size,
                        conv_expansion_factor,
                        conv_dropout_p
                    )
                )
            ),
            ResidualConnectionModule(
                torch::nn::AnyModule(
                    FeedForwardModule(
                        encoder_dim,
                        feed_forward_expansion_factor,
                        feed_forward_dropout_p
                    )
                )
                ,
                feed_forward_residual_factor
            ),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({encoder_dim}))
        );

        register_module("sequential", sequential);
    }

    torch::Tensor forward(torch::Tensor inputs) {
        return sequential->forward(inputs);
    }

private:
    torch::nn::Sequential sequential;
    double feed_forward_residual_factor;
};
TORCH_MODULE(ConformerBlock);

// Conformer Encoder
class ConformerEncoderImpl : public torch::nn::Module {
public:
    ConformerEncoderImpl(
        int64_t input_dim = 80,
        int64_t encoder_dim = 512,
        int64_t num_layers = 6, // this should be small
        int64_t num_attention_heads = 8,
        int64_t feed_forward_expansion_factor = 4,
        int64_t conv_expansion_factor = 2,
        double input_dropout_p = 0.1,
        double feed_forward_dropout_p = 0.1,
        double attention_dropout_p = 0.1,
        double conv_dropout_p = 0.1,
        int64_t conv_kernel_size = 31,
        bool half_step_residual = true
    ): conv_subsample(1, encoder_dim),
    input_projection(torch::nn::Linear(input_dim * (((encoder_dim - 1) / 2 - 1) / 2), encoder_dim),
                    torch::nn::Dropout(input_dropout_p)),
    conformer_layers(torch::nn::Sequential()) { // Initialize conformer_layers here 

        register_module("conv_subsample", conv_subsample);
        register_module("input_projection", input_projection);

        for (int64_t i = 0; i < num_layers; ++i) {
            conformer_layers->push_back(register_module("conformer_block_" + std::to_string(i),
                ConformerBlock(
                    encoder_dim,
                    num_attention_heads,
                    feed_forward_expansion_factor,
                    conv_expansion_factor,
                    feed_forward_dropout_p,
                    attention_dropout_p,
                    conv_dropout_p,
                    conv_kernel_size,
                    half_step_residual)));
        }
        register_module("conformer_layers", conformer_layers);
    }

    torch::Tensor forward(torch::Tensor inputs, torch::Tensor input_lengths) {
        std::tuple<torch::Tensor, torch::Tensor> subsample_output = conv_subsample->forward(inputs, input_lengths);
        torch::Tensor outputs = input_projection->forward(std::get<0>(subsample_output));

        for (auto& layer : *conformer_layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

private:
    Conv2dSubsampling conv_subsample;
    torch::nn::Sequential input_projection;
    torch::nn::Sequential conformer_layers;
};

TORCH_MODULE(ConformerEncoder);