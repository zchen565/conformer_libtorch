#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <optional>
#include <tuple>

#include "basic.h"
#include "feedforward.h"
#include "convolution.h"
#include "attention.h"
#include "conformer_encoder.h"

// basic.h
void test_ResidualConnectionModule() {
    int64_t input_features = 10;
    int64_t output_features = 10;
    int64_t batch_size = 5;

    // 创建一个 ResidualConnectionModule 实例
    // 以一个线性层作为内部模块
    auto linear = torch::nn::Linear(input_features, output_features);
    auto residual_module = ResidualConnectionModule(torch::nn::AnyModule(linear), 1.0, 1.0);

    // 创建一个随机输入 Tensor
    torch::Tensor input = torch::randn({batch_size, input_features});

    // 将输入传递给残差模块并获取输出
    torch::Tensor output = residual_module->forward(input);

    // 打印输出的形状以验证结果
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;
    std::cout << "ResidualConnectionModule Success" << std::endl; 
}

void test_Linear() {
    torch::manual_seed(0); // For reproducible results

    // Create a Linear layer
    Linear linear(10, 5); // Example sizes for in and out features

    // Create a dummy input tensor
    torch::Tensor x = torch::randn({1, 10}); // Batch size of 1 and 10 features

    // Forward pass
    torch::Tensor y = linear->forward(x);

    // Print output
    std::cout << "Output Tensor Shape: " << y.sizes() << std::endl;
    std::cout << "Linear success" << std::endl;
}

void test_View_Transpose() {
    torch::manual_seed(0); // For reproducible results

    // Create a View module to reshape a tensor to 2x5
    View view(std::vector<int64_t>{2, 5}, false);

    // Create a Transpose module to transpose dimensions 0 and 1
    Transpose transpose(0, 1);

    // Create a dummy input tensor of size 1x10
    torch::Tensor x = torch::randn({1, 10});

    // Apply View
    torch::Tensor y_view = view->forward(x);
    std::cout << "After View: " << y_view.sizes() << std::endl;

    // Apply Transpose
    torch::Tensor y_transpose = transpose->forward(y_view);
    std::cout << "After Transpose: " << y_transpose.sizes() << std::endl;

    std::cout << "View and Transpose success! " << std::endl;

}
// feedforward.h
void test_FeedForwardModule() {
    int64_t input_dim = 256; 
    int64_t hidden_dim = 512; 
    double dropout = 0.1; 

    FeedForwardModule feed_forward_module(input_dim, hidden_dim, dropout);

    int64_t B = 2; 
    auto input = torch::randn({B, input_dim});

    auto output = feed_forward_module->forward(input);

    // 输出结果的形状
    std::cout << "Output Tensor Shape: " << output.sizes() << std::endl;
    std::cout << "Feed Forward Module success" << std::endl;
}

// convolution.h

void test_basic_convolution() {

    // Create a random input tensor
    auto input = torch::rand({1, 16, 50}); // (batch, channels, length)

    // Create the DepthwiseConv1d and PointwiseConv1d modules
    auto depthwise = DepthwiseConv1d(16, 32, 3); // Example parameters
    auto pointwise = PointwiseConv1d(32, 64); // Example parameters

    // Apply the convolutions
    auto output_depthwise = depthwise->forward(input);
    auto output_pointwise = pointwise->forward(output_depthwise);

    // Print output sizes
    std::cout << "Output size after DepthwiseConv1d: " << output_depthwise.sizes() << std::endl;
    std::cout << "Output size after PointwiseConv1d: " << output_pointwise.sizes() << std::endl;

}

void test_DepthConvolution() {
    int64_t in_channels = 16; // 例如
    int64_t out_channels = 32; // 应该是 in_channels 的倍数
    int64_t kernel_size = 31;
    int64_t stride = 1;
    int64_t padding = 0; // 'SAME' padding
    bool bias = false;

    // 创建 DepthwiseConv1d 模块实例
    DepthwiseConv1d depthwiseConv1d(in_channels, out_channels, kernel_size, stride, padding, bias);

    // 创建一个测试输入 Tensor
    torch::Tensor input = torch::rand({1, in_channels, 50}); // 假设 batch_size=1, in_channels=16, length=50
    std::cout << "Input size: " << input.sizes() << std::endl;
    // 运行模块
    torch::Tensor output = depthwiseConv1d->forward(input);

    // 计算输出 Tensor 的预期维度
    int64_t expected_length = (50 + 2 * padding - kernel_size) / stride + 1;

    // 打印输出维度
    std::cout << "Output size: " << output.sizes() << std::endl;

    // 检查输出维度是否符合预期
    if (output.size(0) == 1 && output.size(1) == out_channels && output.size(2) == expected_length) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed: Incorrect output dimensions." << std::endl;
    }
}
void test_ConformerConvolution() {
    // auto input = torch::rand({1, 50, 16}); // (batch, time, channels)

    // // Create the ConformerConvModule
    // ConformerConvModule conv_module(16); // Example in_channels

    // // Apply the module
    // auto output = conv_module->forward(input);

    // // Print output sizes
    // std::cout << "check what happend" << std::endl;
    // std::cout << "Output size after ConformerConvModule: " << output.sizes() << std::endl;
    // 设置输入参数
    int64_t in_channels = 64; // 比如
    // int64_t kernel_size = 31;

    float dropout_p = 0.1;
    int64_t expansion_factor = 2;

    // 创建模块实例
    ConformerConvModule module(in_channels);

    // 创建一个测试输入 Tensor
    torch::Tensor input = torch::rand({16, 100, in_channels}); // 假设 batch_size=1, seq_len=50
    std::cout << "Input size: " << input.sizes() << std::endl;
    // 运行模块
    torch::Tensor output = module->forward(input);

    // 打印输出维度
    std::cout << "Output size: " << output.sizes() << std::endl;

    // 检查输出维度是否正确（根据您的模型设计）
    if (output.size(0) == 1 && output.size(1) == 50 && output.size(2) == in_channels) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed: Incorrect output dimensions." << std::endl;
    }
}

void test_Convolution2dSampling() {
    int64_t in_channels = 1; // 例如，对于灰度图像
    int64_t out_channels = 32;
    int64_t batch_size = 1;
    int64_t height = 128; // 输入图像的高度
    int64_t width = 128; // 输入图像的宽度

    // 创建 Conv2dSubsampling 模块实例
    Conv2dSubsampling subsampling(in_channels, out_channels);

    // 创建一个测试输入 Tensor
    torch::Tensor input = torch::rand({batch_size, height, width});

    // 创建输入长度 Tensor
    torch::Tensor input_lengths = torch::full({batch_size}, width, torch::dtype(torch::kInt32));

    // 运行模块
    auto outputs = subsampling->forward(input, input_lengths);
    torch::Tensor subsampled_inputs = std::get<0>(outputs);
    torch::Tensor output_lengths = std::get<1>(outputs);

    // 打印输出维度
    std::cout << "Subsampled inputs size: " << subsampled_inputs.sizes() << std::endl;
    std::cout << "Output lengths: " << output_lengths << std::endl;

    // 验证输出维度
    int64_t expected_height = (height - 1) / 2 / 2; // 两次卷积层，每次 stride=2
    int64_t expected_width = (width - 1) / 2 / 2;
    int64_t expected_channels = out_channels;

    if (subsampled_inputs.size(1) == expected_height &&
        subsampled_inputs.size(2) == expected_channels * expected_width &&
        output_lengths[0].item<int>() == (width / 4 - 1)) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed: Incorrect output dimensions." << std::endl;
    }
}
// attention.h

void test_PositionalEncoding() {
    PositionalEncoding pe(64, 20);
    
    std::cout << "init success "<< std::endl;
    
    // Test for a specific length
    int64_t test_length = 10;
    torch::Tensor output = pe->forward(test_length);

    std::cout << "forward success "<< std::endl;
    // Print out the output dimensions
    std::cout << "Output dimensions: " << output.sizes() << std::endl;

    // Print out a part of the tensor to visually inspect
    // std::cout << "Output tensor slice: " << output.index({0, Slice(None, 5), Slice(None, 10)}) << std::endl;

}

void test_RelativeMultiHeadAttention(){
    RelativeMultiHeadAttention attention_module(512,16);

    // 设置示例输入
    int64_t batch_size = 4;
    int64_t seq_length = 10;
    int64_t d_model = 512;
    int64_t num_heads = 16;

    torch::Tensor query = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor key = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor value = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor pos_embedding = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor mask = torch::randn({batch_size, seq_length, seq_length}).ge(0.5); // 生成示例的掩码
    std::cout << "init success "<< std::endl;
    // 调用 forward 函数获取输出
    torch::Tensor output = attention_module->forward(query, key, value, pos_embedding, mask);
std::cout << "forward success "<< std::endl;
    // 打印输出的维度
    std::cout << "Output dimensions: " << output.sizes() << std::endl;
}


void test_RelativeMultiHeadAttention_emptymask(){
    RelativeMultiHeadAttention attention_module(512,16);

    // 设置示例输入
    int64_t batch_size = 4;
    int64_t seq_length = 10;
    int64_t d_model = 512;
    int64_t num_heads = 16;

    torch::Tensor query = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor key = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor value = torch::randn({batch_size, seq_length, d_model});
    torch::Tensor pos_embedding = torch::randn({batch_size, seq_length, d_model});
    // torch::Tensor mask = torch::randn({batch_size, seq_length, seq_length}).ge(0.5); // 生成示例的掩码
    std::cout << "init success "<< std::endl;
    // 调用 forward 函数获取输出
    torch::Tensor output = attention_module->forward(query, key, value, pos_embedding);
std::cout << "forward success "<< std::endl;
    // 打印输出的维度
    std::cout << "Output dimensions: " << output.sizes() << std::endl;

}

void test_MHSA(){
    int64_t d_model = 512;  // Model dimension
    int64_t num_heads = 8;  // Number of attention heads
    double dropout_p = 0.1; // Dropout probability
    int64_t batch_size = 10; // Batch size
    int64_t seq_length = 20; // Sequence length

    // Initialize the module
    MultiHeadedSelfAttentionModule attention_module(d_model, num_heads, dropout_p);

    // Generate random inputs
    torch::Tensor inputs = torch::rand({batch_size, seq_length, d_model});
    torch::Tensor mask; // Empty mask for this test

    // Forward pass
    torch::Tensor outputs = attention_module->forward(inputs, mask);

    // Check output dimensions
    std::cout << "Output dimensions: " << outputs.sizes() << std::endl;

    // Verify if output dimensions match the expected dimensions
    // if (outputs.size(0) == batch_size && outputs.size(1) == seq_length && outputs.size(2) == d_model) {
    //     std::cout << "Test passed: Output dimensions are correct." << std::endl;
    // } else {
    //     std::cout << "Test failed: Output dimensions are incorrect." << std::endl;
    // }
}

void test_MHSA_emptymask(){
    int64_t d_model = 512;  // Model dimension
    int64_t num_heads = 8;  // Number of attention heads
    double dropout_p = 0.1; // Dropout probability
    int64_t batch_size = 10; // Batch size
    int64_t seq_length = 20; // Sequence length

    // Initialize the module
    MultiHeadedSelfAttentionModule attention_module(d_model, num_heads, dropout_p);

    // Generate random inputs
    torch::Tensor inputs = torch::rand({batch_size, seq_length, d_model});
    // Forward pass
    std::cout << "start forward\n";
    torch::Tensor outputs = attention_module->forward(inputs);
    std::cout << "start forward\n";
    // Check output dimensions
    std::cout << "Output dimensions: " << outputs.sizes() << std::endl;

    // Verify if output dimensions match the expected dimensions
    if (outputs.size(0) == batch_size && outputs.size(1) == seq_length && outputs.size(2) == d_model) {
        std::cout << "Test passed: Output dimensions are correct." << std::endl;
    } else {
        std::cout << "Test failed: Output dimensions are incorrect." << std::endl;
    }
}
// conformer_encoder.h

void test_ConformerBlock(){

    // Create a ConformerBlock instance
    ConformerBlock conformerBlock(512); // default encoder_dim = 512

    // Create a dummy input tensor of shape (batch_size, time, dim)
    int64_t batch_size = 1;
    int64_t time = 100;
    int64_t dim = 512; // This should match the encoder_dim of ConformerBlock
    auto input = torch::randn({batch_size, time, dim});

    // Forward pass

    std::cout << "test_305" << std::endl;
    torch::Tensor output = conformerBlock->forward(input);
    
    std::cout << "test_308" << std::endl;
    // Check the output dimensions
    if (output.sizes() == input.sizes()) {
        std::cout << "Test passed! Output dimensions match the input dimensions." << std::endl;
    } else {
        std::cerr << "Test failed! Output dimensions do not match the expected dimensions." << std::endl;
    }
}

void test_ConformerEncoder() {
    int64_t batch_size = 4;  // 批量大小
    int64_t input_dim = 80; // 输入维度

    // 创建 ConformerEncoder 实例
    ConformerEncoder encoder(input_dim);

    // 创建模拟输入数据
    torch::Tensor inputs = torch::randn({batch_size, input_dim, 100}); // 假设输入序列长度为 100
    std::cout << "Input shape: " << inputs.sizes() << std::endl;
    // 前向传递
    torch::Tensor input_lengths = torch::full({batch_size}, 100, torch::kInt64); //?
    std::cout <<"test shape : " << input_lengths.sizes() << std::endl;
    torch::Tensor outputs = encoder->forward(inputs, input_lengths);

    // 打印输出的形状
    std::cout << "Output shape: " << outputs.sizes() << std::endl;

}
int main(int argc, char const *argv[])
{
    // basic.h
    test_ResidualConnectionModule();
    test_Linear();
    test_View_Transpose();
    test_FeedForwardModule();

    // convolution.h
    test_basic_convolution();
    test_DepthConvolution();
    test_ConformerConvolution(); // note line 98 why [1,16,50]->[1,16,20] // this is correct !
    test_Convolution2dSampling();

    // attention.h
    test_PositionalEncoding();
    test_RelativeMultiHeadAttention();
    test_RelativeMultiHeadAttention_emptymask();
    test_MHSA();
    test_MHSA_emptymask();
    
    // conformer_encoder.h
    test_ConformerBlock();

    // test_ConformerEncoder(); 
    // note we need to specify the exact input size to fit the model for test (including the test case!)
    // this is corrent
    return 0;
}
