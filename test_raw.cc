#include <torch/torch.h>
#include <iostream>
#include "conformer.h"

// feed forward test
void test_FeedForwardModule() {
    int64_t input_dim = 256; 
    int64_t hidden_dim = 512; 
    double dropout = 0.1; 

    FeedForwardModule feed_forward_module(input_dim, hidden_dim, dropout);

    int64_t B = 2; 
    auto input = torch::randn({B, input_dim});

    auto output = feed_forward_module->forward(input);
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;
    std::cout << "Feed Forward Module success" << std::endl;
}


// convolution test
void test_ConvolutionModule() {
    
    // conformer block test
    torch::manual_seed(0);

    // 创建模块实例
    int64_t input_dim = 256; 
    int64_t num_channels = 32; // 示例通道数
    int64_t depthwise_kernel_size = 3; // 示例深度卷积核大小
    double dropout = 0.1; // 示例 dropout 概率
    bool bias = true; // 启用或禁用偏置
    bool use_group_norm = true; // 使用 GroupNorm 或 BatchNorm

    ConvolutionModule conv_module(input_dim, num_channels, depthwise_kernel_size, dropout, bias, use_group_norm);

    // 创建一个假的输入张量
    int64_t batch_size = 10; // 示例批次大小
    int64_t seq_length = 100; // 示例序列长度
    torch::Tensor input = torch::randn({batch_size, seq_length, input_dim});

    // 执行前向传播
    
    torch::Tensor output = conv_module->forward(input);

    // 打印输出的形状
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;
    std::cout << "Convolution Module success" << std::endl;
}


void test_ConformerBlock() {
    int64_t batch_size = 2;
    int64_t seq_len = 128;
    int64_t input_dim = 512;
    int64_t ffn_dim = 2048;
    int64_t num_attention_heads = 8;
    int64_t depthwise_conv_kernel_size = 31;
    double dropout = 0.1;
    bool use_group_norm = true; // should also test false
    bool convolution_first = false;

    ConformerBlockImpl conformer_block(input_dim, ffn_dim, num_attention_heads, depthwise_conv_kernel_size, dropout, use_group_norm, convolution_first);

    torch::Tensor input = torch::randn({batch_size, seq_len, input_dim});

    torch::Tensor output = conformer_block.forward(input);

    assert(output.sizes() == std::vector<int64_t>({batch_size, seq_len, input_dim}));
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;
    std::cout << "Conformer Block Module passed." << std::endl;
}


void test_Conformer() {
    int64_t batch_size = 2;
    int64_t seq_len = 128;
    int64_t input_dim = 512;
    int64_t num_heads = 8;
    int64_t ffn_dim = 2048;
    int64_t num_layers = 6;
    int64_t depthwise_conv_kernel_size = 31;
    double dropout = 0.1;
    bool use_group_norm = true;
    bool convolution_first = false;

    // 创建 Conformer 模型实例
    ConformerImpl conformer(input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, dropout, use_group_norm, convolution_first);

    // 创建虚拟输入和长度向量
    torch::Tensor input = torch::randn({batch_size, seq_len, input_dim});
    torch::Tensor lengths = torch::tensor({seq_len, seq_len});  // 假设所有序列长度相同

    // 应用模型
    auto [output, _] = conformer.forward(input, lengths);

    // 检查输出形状是否正确
    assert(output.sizes() == std::vector<int64_t>({batch_size, seq_len, input_dim}));
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;
    std::cout << "Conformer passed." << std::endl;
}
int main() {

    test_FeedForwardModule();

    test_ConvolutionModule();

    test_ConformerBlock();

    test_Conformer();

    return 0;
}