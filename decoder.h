#include <torch/torch.h>

// one lstm decoder , this is not the transducer version
class DecoderImpl : public torch::nn::Module {
public:
    DecoderImpl(int enc_dim, int vocab_size)
        : lstm(torch::nn::LSTMOptions(enc_dim, vocab_size).batch_first(true)) {
        register_module("lstm", lstm);
    }

    torch::Tensor forward(torch::Tensor inp) {
        torch::Tensor output;
        std::tie(output, std::ignore) = lstm(inp);
        return output;
    }

private:
    torch::nn::LSTM lstm;
};

TORCH_MODULE(Decoder);
