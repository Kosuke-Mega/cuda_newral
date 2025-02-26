#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

#define INPUT_DIM 300
#define OUTPUT_DIM 10

// CUDAエラーをチェックするためのマクロ
#define CUDA_CHECK(err) do { \
    if(err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 全結合層（FC層）のCUDAカーネル
// in: 入力ベクトル（長さ INPUT_DIM）
// weights: 重み行列 (OUTPUT_DIM x INPUT_DIM) ※行優先
// bias: バイアス（長さ OUTPUT_DIM）
// out: 出力ベクトル（長さ OUTPUT_DIM）
__global__ void fc_kernel(const float *in, const float *weights, const float *bias, float *out, int input_dim, int output_dim) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < output_dim) {
        float sum = 0.0f;
        for(int j = 0; j < input_dim; j++){
            sum += in[j] * weights[i * input_dim + j];
        }
        out[i] = sum + bias[i];
    }
}

// テキスト埋め込みを取得する関数（ダミー実装）
std::vector<float> get_text_embedding(const std::string& text) {
    std::vector<float> embedding(INPUT_DIM, 0.0f);
    // 簡単な例として、文字列の長さを使って埋め込みを生成
    for (size_t i = 0; i < text.length() && i < INPUT_DIM; ++i) {
        embedding[i] = static_cast<float>(text[i]) / 255.0f;
    }
    return embedding;
}

// 重みを初期化する関数
std::vector<float> initialize_weights(int output_dim, int input_dim) {
    std::vector<float> weights(output_dim * input_dim);
    for (auto& w : weights) {
        w = static_cast<float>(rand()) / RAND_MAX * 0.1f;
    }
    return weights;
}

// バイアスを初期化する関数
std::vector<float> initialize_bias(int output_dim) {
    return std::vector<float>(output_dim, 0.0f);
}

// 応答を生成する関数
std::string generate_response(int predicted_class) {
    switch (predicted_class) {
        case 0: return "Hello!";
        case 1: return "I'm good!";
        default: return "Sorry, I do not understand.";
    }
}

int main() {
    // 1. ユーザからの入力を取得
    std::cout << "Question: ";
    std::string question;
    std::getline(std::cin, question);
    
    // 2. テキスト埋め込みを使用して入力ベクトルを作成
    std::vector<float> h_input = get_text_embedding(question);
    
    // 3. モデルパラメータ：重みとバイアスの初期化（より複雑なモデルとして実装）
    std::vector<float> h_weights = initialize_weights(OUTPUT_DIM, INPUT_DIM);
    std::vector<float> h_bias = initialize_bias(OUTPUT_DIM);
    
    // 4. GPUメモリの確保
    float *d_input, *d_weights, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, OUTPUT_DIM * INPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias, OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, OUTPUT_DIM * sizeof(float)));
    
    // 5. ホストからデバイスへデータ転送
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), OUTPUT_DIM * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), OUTPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    
    // 6. 全結合層カーネルの実行
    int block = 256;
    int grid = (OUTPUT_DIM + block - 1) / block;
    fc_kernel<<<grid, block>>>(d_input, d_weights, d_bias, d_output, INPUT_DIM, OUTPUT_DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 7. 結果をデバイスからホストへ転送
    std::vector<float> h_output(OUTPUT_DIM, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, OUTPUT_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    
    // （オプション）出力に対してソフトマックス処理（今回は各出力はそのままで十分ですが、下記は例です）
    float max_val = h_output[0];
    for (int i = 1; i < OUTPUT_DIM; i++) {
        if (h_output[i] > max_val) max_val = h_output[i];
    }
    float sum = 0.0f;
    std::vector<float> softmax_output(OUTPUT_DIM, 0.0f);
    for (int i = 0; i < OUTPUT_DIM; i++) {
        softmax_output[i] = exp(h_output[i] - max_val);
        sum += softmax_output[i];
    }
    for (int i = 0; i < OUTPUT_DIM; i++) {
        softmax_output[i] /= sum;
    }
    
    // 8. 最も高い確率のクラスを予測（argmax）
    int predicted_class = 0;
    float highest_prob = softmax_output[0];
    for (int i = 1; i < OUTPUT_DIM; i++) {
        if (softmax_output[i] > highest_prob) {
            highest_prob = softmax_output[i];
            predicted_class = i;
        }
    }
    
    // 9. 出力クラスに対応する応答を決定（より多様な応答をサポート）
    std::string answer = generate_response(predicted_class);
    
    std::cout << "LLM Response: " << answer << std::endl;
    
    // 10. GPUメモリの解放
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
    
    return 0;
}
