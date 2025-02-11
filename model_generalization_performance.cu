#include <iostream>
#include <cuda_runtime.h>

// sigmoid関数の実装
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// sigmoid関数の導関数
__device__ float d_sigmoid(float x) {
    return x * (1.0f - x);
}

// 順伝播関数
__device__ void forward(float* input, float* output, float* weights, float* biases, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float hidden = sigmoid(input[idx * 2] * weights[0] + input[idx * 2 + 1] * weights[1] + biases[0]);
        output[idx] = sigmoid(hidden * weights[2] + biases[1]);
    }
}

// 逆伝播関数
__device__ void backward(float* input, float* output, float* target, float* weights, float* biases, float* delta_weights, float* delta_biases, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        float error = target[idx] - output[idx];
        float delta_output = error * d_sigmoid(output[idx]);
        float hidden = sigmoid(input[idx * 2] * weights[0] + input[idx * 2 + 1] * weights[1] + biases[0]);
        float delta_hidden = delta_output * weights[2] * d_sigmoid(hidden);

        delta_weights[0] += delta_hidden * input[idx * 2];
        delta_weights[1] += delta_hidden * input[idx * 2 + 1];
        delta_weights[2] += delta_output * hidden;
        delta_biases[0] += delta_hidden;
        delta_biases[1] += delta_output;
    }
}

// 学習関数
__global__ void train(float* input, float* output, float* target, float* weights, float* biases, float* delta_weights, float* delta_biases, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        forward(input, output, weights, biases, size);
        backward(input, output, target, weights, biases, delta_weights, delta_biases, size);
    }
}

int main() {
    int size = 4; // トレーニングデータの数

    float* input, *output, *target, *weights, *biases, *delta_weights, *delta_biases;
    cudaMalloc((void**)&input, size * 2 * sizeof(float)); // 入力値
    cudaMalloc((void**)&output, size * sizeof(float)); // 出力値
    cudaMalloc((void**)&target, size * sizeof(float)); // 目標値
    cudaMalloc((void**)&weights, 3 * sizeof(float)); // 重み
    cudaMalloc((void**)&biases, 2 * sizeof(float)); // バイアス
    cudaMalloc((void**)&delta_weights, 3 * sizeof(float)); // 重みの更新量
    cudaMalloc((void**)&delta_biases, 2 * sizeof(float)); // バイアスの更新量

    // 入力値と出力値の初期化
    float* input_host = new float[size * 2];
    float* target_host = new float[size];
    for (int i = 0; i < size; i++) {
        input_host[i * 2] = i == 0 || i == 3 ? 0.0f : 1.0f;
        input_host[i * 2 + 1] = i == 0 || i == 1 ? 0.0f : 1.0f;
        target_host[i] = i == 0 || i == 3 ? 0.0f : 1.0f;
    }
    cudaMemcpy(input, input_host, size * 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(target, target_host, size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] input_host;
    delete[] target_host;

    // 重みとバイアスの初期化
    float* weights_host = new float[3];
    float* biases_host = new float[2];
    for (int i = 0; i < 3; i++) {
        weights_host[i] = 0.1f;
    }
    for (int i = 0; i < 2; i++) {
        biases_host[i] = 0.1f;
    }
    cudaMemcpy(weights, weights_host, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(biases, biases_host, 2 * sizeof(float), cudaMemcpyHostToDevice);
    delete[] weights_host;
    delete[] biases_host;

    // 学習
    int epoch = 10000;
    for (int i = 0; i < epoch; i++) {
        train<<<1, 1>>>(input, output, target, weights, biases, delta_weights, delta_biases, size);
    }

    // 出力値の取得
    float* output_host = new float[size];
    cudaMemcpy(output_host, output, size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
        std::cout << "output[" << i << "] = " << output_host[i] << std::endl;
    }
    delete[] output_host;

    cudaFree(input);
    cudaFree(output);
    cudaFree(target);
    cudaFree(weights);
    cudaFree(biases);
    cudaFree(delta_weights);
    cudaFree(delta_biases);
    return 0;
}