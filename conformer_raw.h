/**
 * @brief NOTE this is a simiplified version of Conformer, the original (complete) version is not in this file
 * this file is only for writing test and exercise libtorch
 */
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


// // class MyBigModelImpl : public torch::nn::Module {
// //  public:
// //     MyBigModelImpl(int64_t input_dim, int64_t hidden_dim, double dropout = 0.0) {
// //         // check FFM
// //         feed_forward_module = register_module("feed_forward_module", FeedForwardModule(input_dim, hidden_dim, dropout));
        
// //         // Register others
// //     }

// //     torch::Tensor forward(torch::Tensor input) {
// //         auto output = feed_forward_module->forward(input);

// //         // rest of the forward shit

// //         return output;
// //     }

// //  private:
// //     FeedForwardModule feed_forward_module{nullptr};

// //     // remaining
// // };

// // TORCH_MODULE(MyBigModel);

class ConvolutionModuleImpl : public torch::nn::Module {
public:
/**
 * @brief Construct a new Convolution Module Impl object
 * note the input dim should be (B,T,D) , (batch, sequence length, feature dimension)
 * 
 * @param input_dim 
 * @param num_channels 
 * @param depthwise_kernel_size 
 * @param dropout 
 * @param bias 
 * @param use_group_norm 
 */
    ConvolutionModuleImpl(int64_t input_dim, int64_t num_channels, int64_t depthwise_kernel_size, double dropout = 0.0, bool bias = false, bool use_group_norm = false) {
        if ((depthwise_kernel_size - 1) % 2 != 0) {
            throw std::invalid_argument("depthwise_kernel_size must be odd");
        }

        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));

        torch::nn::Sequential seq;
        auto conv1 = torch::nn::Conv1d(torch::nn::Conv1dOptions(input_dim, 2 * num_channels, 1).stride(1).padding(0).bias(bias));
        seq->push_back(conv1);
        auto glu = torch::nn::Functional(torch::glu, 1);
        seq->push_back(glu);

        auto conv2 = torch::nn::Conv1d(torch::nn::Conv1dOptions(num_channels, num_channels, depthwise_kernel_size).stride(1).padding((depthwise_kernel_size - 1) / 2).groups(num_channels).bias(bias));
        seq->push_back(conv2);

        if (use_group_norm) {
            seq->push_back(torch::nn::GroupNorm(1, num_channels));
        } else {
            seq->push_back(torch::nn::BatchNorm1d(num_channels));
        }

        seq->push_back(torch::nn::SiLU());
        seq->push_back(torch::nn::Conv1d(torch::nn::Conv1dOptions(num_channels, input_dim, 1).stride(1).padding(0).bias(bias)));
        seq->push_back(torch::nn::Dropout(dropout));

        sequential = register_module("sequential", seq);
    }

    torch::Tensor forward(torch::Tensor input) {
        input = layer_norm->forward(input);
        input = input.transpose(1, 2);
        input = sequential->forward(input);
        return input.transpose(1, 2);
    }

private:
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Sequential sequential{nullptr};
};

TORCH_MODULE(ConvolutionModule);


/**
 * @brief 
 * 
 */
class ConformerBlockImpl : public torch::nn::Module {
public:
    ConformerBlockImpl(int64_t input_dim, int64_t ffn_dim, int64_t num_attention_heads, int64_t depthwise_conv_kernel_size, double dropout = 0.0, bool use_group_norm = false, bool convolution_first = false) {
        
        ffn1 = register_module("ffn1", FeedForwardModule(input_dim, ffn_dim, dropout));
        // ffn1 = register_module("ffn1", std::make_shared<FeedForwardModuleImpl>(input_dim, ffn_dim, dropout));


        self_attn_layer_norm = register_module("self_attn_layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));
        self_attn = register_module("self_attn", torch::nn::MultiheadAttention(torch::nn::MultiheadAttentionOptions(input_dim, num_attention_heads).dropout(dropout)));
        self_attn_dropout = register_module("self_attn_dropout", torch::nn::Dropout(dropout));

        conv_module = register_module("conv_module", ConvolutionModule(input_dim, input_dim, depthwise_conv_kernel_size, dropout, true, use_group_norm));
        // conv_module = register_module("conv_module", std::make_shared<ConvolutionModule>(input_dim, input_dim, depthwise_conv_kernel_size, dropout, true, use_group_norm));

        ffn2 = register_module("ffn2", FeedForwardModule(input_dim, ffn_dim, dropout));
        // ffn2 = register_module("ffn2", std::make_shared<FeedForwardModuleImpl>(input_dim, ffn_dim, dropout));
        final_layer_norm = register_module("final_layer_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));
        this->convolution_first = convolution_first;
    }

    torch::Tensor forward(torch::Tensor input, std::optional<torch::Tensor> key_padding_mask = std::nullopt) {
        auto residual = input;
        auto x = ffn1->forward(input);
        x = x * 0.5 + residual;

        if (convolution_first) {
            x = apply_convolution(x);
        }

        residual = x;
        x = self_attn_layer_norm->forward(x);
        // std::tie(x, std::ignore) = self_attn->forward(x, x, x, key_padding_mask);
        if (key_padding_mask.has_value()) {
            // 如果 key_padding_mask 包含值，则提取它并使用
            std::tie(x, std::ignore) = self_attn->forward(x, x, x, key_padding_mask.value(), false);
        } else {
            // 如果 key_padding_mask 没有值，则传递一个默认张量
            std::tie(x, std::ignore) = self_attn->forward(x, x, x, at::Tensor(), false);
        }
        x = self_attn_dropout->forward(x);
        x = x + residual;

        if (!convolution_first) {
            x = apply_convolution(x);
        }

        residual = x;
        x = ffn2->forward(x);
        x = x * 0.5 + residual;

        x = final_layer_norm->forward(x);
        return x;
    }

private:
    // torch::nn::ModuleHolder<FeedForwardModule> ffn1;
    FeedForwardModule ffn1{nullptr};
    torch::nn::LayerNorm self_attn_layer_norm{nullptr};
    torch::nn::MultiheadAttention self_attn{nullptr};
    torch::nn::Dropout self_attn_dropout{nullptr};
    // torch::nn::ModuleHolder<ConvolutionModule> conv_module;
    ConvolutionModule conv_module{nullptr};
    // torch::nn::ModuleHolder<FeedForwardModule> ffn2;
    FeedForwardModule ffn2{nullptr};
    torch::nn::LayerNorm final_layer_norm{nullptr};
    bool convolution_first;

    torch::Tensor apply_convolution(torch::Tensor input) {
        auto residual = input;
        input = input.transpose(0, 1);
        input = conv_module->forward(input);
        input = input.transpose(0, 1);
        input = residual + input;
        return input;
    }
};
TORCH_MODULE(ConformerBlock);




class ConformerImpl : public torch::nn::Module {
public:
    ConformerImpl(
        int64_t input_dim,
        int64_t num_heads,
        int64_t ffn_dim,
        int64_t num_layers,
        int64_t depthwise_conv_kernel_size,
        double dropout = 0.0,
        bool use_group_norm = false,
        bool convolution_first = false
    ) {

        for (int64_t i = 0; i < num_layers; ++i) {
            // Create a ConformerBlock and give it a unique name
            conformer_layers->push_back(register_module("conformer_block_" + std::to_string(i),
            
                ConformerBlock(input_dim, ffn_dim, num_heads, depthwise_conv_kernel_size, dropout, use_group_norm, convolution_first)));
        }
        register_module("conformer_layers", conformer_layers);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor lengths) {


        torch::Tensor encoder_padding_mask = LengthsToPaddingMask(lengths);

        auto x = input.transpose(0, 1);
        for (auto& layer : *conformer_layers) { // *conformer_layers
            x = layer.forward(x, std::optional<torch::Tensor>(encoder_padding_mask));
        }
        return std::make_tuple(x.transpose(0, 1), lengths);
    }
private:
    torch::nn::Sequential conformer_layers;
};
TORCH_MODULE(Conformer);



