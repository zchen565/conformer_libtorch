#pragma once
#include <torch/torch.h>

/**
 * @brief this is the function for padding for the final conformer model
 * 
 * @param lengths 
 * @return torch::Tensor 
 */

torch::Tensor LengthsToPaddingMask(const torch::Tensor& lengths) {
    auto batch_size = lengths.size(0);
    auto max_length = lengths.max().item<int64_t>();
    auto arange = torch::arange(max_length, lengths.options()).expand({batch_size, -1});
    auto padding_mask = arange >= lengths.unsqueeze(1);
    return padding_mask;
}

// template<typename T>
class ResidualConnectionModuleImpl : public torch::nn::Module {
public:
    // Constructor
    ResidualConnectionModuleImpl(torch::nn::AnyModule module,
                                 float module_factor = 1.0,
                                 float input_factor = 1.0)
        : module_(std::move(module)), 
          module_factor_(module_factor), 
          input_factor_(input_factor) {}

    // Forward function
    torch::Tensor forward(torch::Tensor inputs) {
        return module_.forward(inputs).mul(module_factor_) + inputs.mul(input_factor_);
    }

private:
    torch::nn::AnyModule module_;
    float module_factor_;
    float input_factor_;
};

TORCH_MODULE(ResidualConnectionModule);


class LinearImpl : public torch::nn::Module {
public:
    LinearImpl(int64_t in_features, int64_t out_features, bool bias = true) {
        linear = torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(bias));
        
        // Register the module
        register_module("linear", linear);

        // Xavier Uniform Initialization
        torch::nn::init::xavier_uniform_(linear->weight);
        
        // Zero Initialization for Bias
        if (bias) {
            torch::nn::init::zeros_(linear->bias);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        return linear->forward(x);
    }

private:
    torch::nn::Linear linear{nullptr};
};
TORCH_MODULE(Linear);

class ViewImpl : torch::nn::Module {
public:
    ViewImpl(std::vector<int64_t> shape, bool contiguous = false)
    : shape_(std::move(shape)), contiguous_(contiguous) {}

    torch::Tensor forward(torch::Tensor x) {
        if (contiguous_) {
            x = x.contiguous();
        }
        return x.view(shape_);
    }

private:
    std::vector<int64_t> shape_;
    bool contiguous_;
};
TORCH_MODULE(View);

class TransposeImpl : public torch::nn::Module {
public:
    TransposeImpl(int64_t dim1, int64_t dim2)
    : dim1_(dim1), dim2_(dim2) {}

    torch::Tensor forward(torch::Tensor x) {
        return x.transpose(dim1_, dim2_);
    }

private:
    int64_t dim1_, dim2_;
};

TORCH_MODULE(Transpose); 


class GLUImpl : public torch::nn::Module {
public:
    GLUImpl(int64_t _dim):dim(_dim) {}
    torch::Tensor forward(torch::Tensor x) {
        return torch::glu(x, dim);
    }
private:
    int64_t dim;
};

TORCH_MODULE(GLU); 

// Swish in pytorch
class SILUImpl : public torch::nn::Module {
public:
    SILUImpl() {}

    torch::Tensor forward(torch::Tensor x) {
        return torch::silu(x);
    }
};
TORCH_MODULE(SILU); // 用于创建模块的便捷包装器



