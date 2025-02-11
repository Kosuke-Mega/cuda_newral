#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

// 入力画像サイズとチャネル数
#define INPUT_WIDTH 416
#define INPUT_HEIGHT 416
#define CHANNELS 3

// -------------------------------------------------------------------------------
// カーネル関数: 畳み込み (conv2d) forward
// ※ ReLU活性化を内部に持つ
// -------------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(const float* in, const float* weight, const float* bias,
                                      float* out,
                                      int inC, int inH, int inW,
                                      int outC, int outH, int outW,
                                      int kernelH, int kernelW,
                                      int stride, int pad) {
    int oc = blockIdx.z;  // 出力チャネル
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc < outC && oy < outH && ox < outW) {
        float sum = 0.0f;
        for (int ic = 0; ic < inC; ic++) {
            for (int ky = 0; ky < kernelH; ky++) {
                for (int kx = 0; kx < kernelW; kx++) {
                    int iy = oy * stride - pad + ky;
                    int ix = ox * stride - pad + kx;
                    if (iy >= 0 && ix >= 0 && iy < inH && ix < inW) {
                        float v = in[(ic * inH + iy) * inW + ix];
                        float w = weight[(((oc * inC) + ic) * kernelH + ky) * kernelW + kx];
                        sum += v * w;
                    }
                }
            }
        }
        if (bias != nullptr)
            sum += bias[oc];
        // ReLU活性化
        if (sum < 0) sum = 0;
        out[(oc * outH + oy) * outW + ox] = sum;
    }
}

// -------------------------------------------------------------------------------
// カーネル関数: MaxPooling forward (2x2, stride = 2)
// -------------------------------------------------------------------------------
__global__ void maxpool_forward_kernel(const float* in, float* out,
                                       int C, int inH, int inW,
                                       int outH, int outW,
                                       int kernel, int stride) {
    int c = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C && oy < outH && ox < outW) {
        float max_val = -1e30f;
        for (int ky = 0; ky < kernel; ky++) {
            for (int kx = 0; kx < kernel; kx++) {
                int iy = oy * stride + ky;
                int ix = ox * stride + kx;
                if (iy < inH && ix < inW) {
                    float val = in[(c * inH + iy) * inW + ix];
                    if (val > max_val)
                        max_val = val;
                }
            }
        }
        out[(c * outH + oy) * outW + ox] = max_val;
    }
}

// -------------------------------------------------------------------------------
// カーネル関数: 人間特徴検出カーネル
// このカーネルは、検出ヘッドから得られた「人間検出確率マップ」を解析し、
// 各グリッドセルごとに閾値 (threshold) を用いて、1.0 (人間あり) か 0.0 (なし) を出力します。
// -------------------------------------------------------------------------------
__global__ void humanFeatureDetectionKernel(const float* featureMap, float* humanMap,
                                              int gridH, int gridW, int channels, float threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 列（グリッドセル）
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 行（グリッドセル）
    if (x < gridW && y < gridH) {
        float sum = 0.0f;
        // 複数チャネルの場合は各チャネルの値を平均
        for (int c = 0; c < channels; c++) {
            int idx = c * gridH * gridW + y * gridW + x;
            sum += featureMap[idx];
        }
        float avg = sum / channels;
        humanMap[y * gridW + x] = (avg > threshold ? 1.0f : 0.0f);
    }
}

// -------------------------------------------------------------------------------
// ホスト側メイン関数: YOLO向けCNNと人間特徴検出
// -------------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. 画像読み込み
    string imagePath = "img/person.jpg"; // 対象画像パス（人間が写っている画像）
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "画像の読み込みに失敗: " << imagePath << endl;
        return -1;
    }
    // 入力サイズにリサイズ
    Mat resized;
    resize(image, resized, Size(INPUT_WIDTH, INPUT_HEIGHT));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // 2. HWC (OpenCV)→CHW 形式に変換
    int inputSize = INPUT_WIDTH * INPUT_HEIGHT * CHANNELS;
    float* h_input = new float[inputSize];
    for (int c = 0; c < CHANNELS; c++) {
        for (int y = 0; y < INPUT_HEIGHT; y++) {
            for (int x = 0; x < INPUT_WIDTH; x++) {
                h_input[c * (INPUT_WIDTH * INPUT_HEIGHT) + y * INPUT_WIDTH + x] =
                    resized.at<Vec3f>(y, x)[c];
            }
        }
    }

    // GPU用入力メモリ
    float* d_input;
    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // ------------------------------
    // レイヤー1: 畳み込み (Conv1)
    // 入: (3,416,416)  → 出: (16,416,416)
    // カーネル: 3x3, stride=1, pad=1
    // ------------------------------
    int conv1_outC = 16;
    int conv1_kernel = 3;
    int conv1_stride = 1;
    int conv1_pad = 1;
    int conv1_outH = INPUT_HEIGHT; // 416
    int conv1_outW = INPUT_WIDTH;  // 416
    int conv1_weight_size = conv1_outC * CHANNELS * conv1_kernel * conv1_kernel;
    float* h_conv1_weight = new float[conv1_weight_size];
    for (int i = 0; i < conv1_weight_size; i++) {
        h_conv1_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    float* d_conv1_weight;
    cudaMalloc(&d_conv1_weight, conv1_weight_size * sizeof(float));
    cudaMemcpy(d_conv1_weight, h_conv1_weight, conv1_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    float* h_conv1_bias = new float[conv1_outC];
    memset(h_conv1_bias, 0, conv1_outC * sizeof(float));
    float* d_conv1_bias;
    cudaMalloc(&d_conv1_bias, conv1_outC * sizeof(float));
    cudaMemcpy(d_conv1_bias, h_conv1_bias, conv1_outC * sizeof(float), cudaMemcpyHostToDevice);
    float* d_conv1_out;
    int conv1_outSize = conv1_outC * conv1_outH * conv1_outW;
    cudaMalloc(&d_conv1_out, conv1_outSize * sizeof(float));

    dim3 blockDimConv(16, 16);
    dim3 gridDimConv((conv1_outW + blockDimConv.x - 1) / blockDimConv.x,
                     (conv1_outH + blockDimConv.y - 1) / blockDimConv.y,
                     conv1_outC);
    conv2d_forward_kernel<<<gridDimConv, blockDimConv>>>(d_input, d_conv1_weight, d_conv1_bias, d_conv1_out,
                                                          CHANNELS, INPUT_HEIGHT, INPUT_WIDTH,
                                                          conv1_outC, conv1_outH, conv1_outW,
                                                          conv1_kernel, conv1_kernel,
                                                          conv1_stride, conv1_pad);
    cudaDeviceSynchronize();

    // ------------------------------
    // レイヤー1: Max Pooling (2x2, stride=2)
    // 出: (16,208,208)
    // ------------------------------
    int pool1_kernel = 2, pool1_stride = 2;
    int pool1_outH = conv1_outH / 2; // 208
    int pool1_outW = conv1_outW / 2; // 208
    int pool1_outC = conv1_outC;
    int pool1_outSize = pool1_outC * pool1_outH * pool1_outW;
    float* d_pool1_out;
    cudaMalloc(&d_pool1_out, pool1_outSize * sizeof(float));
    dim3 blockDimPool(16, 16);
    dim3 gridDimPool((pool1_outW + blockDimPool.x - 1) / blockDimPool.x,
                     (pool1_outH + blockDimPool.y - 1) / blockDimPool.y,
                     pool1_outC);
    maxpool_forward_kernel<<<gridDimPool, blockDimPool>>>(d_conv1_out, d_pool1_out,
                                                           pool1_outC, conv1_outH, conv1_outW,
                                                           pool1_outH, pool1_outW,
                                                           pool1_kernel, pool1_stride);
    cudaDeviceSynchronize();

    // ------------------------------
    // レイヤー2: 畳み込み (Conv2)
    // 入: (16,208,208) → 出: (32,208,208)
    // カーネル: 3x3, stride=1, pad=1
    // ------------------------------
    int conv2_inC = pool1_outC;
    int conv2_inH = pool1_outH;
    int conv2_inW = pool1_outW;
    int conv2_outC = 32;
    int conv2_kernel = 3, conv2_stride = 1, conv2_pad = 1;
    int conv2_outH = conv2_inH;
    int conv2_outW = conv2_inW;
    int conv2_weight_size = conv2_outC * conv2_inC * conv2_kernel * conv2_kernel;
    float* h_conv2_weight = new float[conv2_weight_size];
    for (int i = 0; i < conv2_weight_size; i++) {
        h_conv2_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    float* d_conv2_weight;
    cudaMalloc(&d_conv2_weight, conv2_weight_size * sizeof(float));
    cudaMemcpy(d_conv2_weight, h_conv2_weight, conv2_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    float* h_conv2_bias = new float[conv2_outC];
    memset(h_conv2_bias, 0, conv2_outC * sizeof(float));
    float* d_conv2_bias;
    cudaMalloc(&d_conv2_bias, conv2_outC * sizeof(float));
    cudaMemcpy(d_conv2_bias, h_conv2_bias, conv2_outC * sizeof(float), cudaMemcpyHostToDevice);
    float* d_conv2_out;
    int conv2_outSize = conv2_outC * conv2_outH * conv2_outW;
    cudaMalloc(&d_conv2_out, conv2_outSize * sizeof(float));
    dim3 blockDimConv2(16, 16);
    dim3 gridDimConv2((conv2_outW + blockDimConv2.x - 1) / blockDimConv2.x,
                      (conv2_outH + blockDimConv2.y - 1) / blockDimConv2.y,
                      conv2_outC);
    conv2d_forward_kernel<<<gridDimConv2, blockDimConv2>>>(d_pool1_out, d_conv2_weight, d_conv2_bias, d_conv2_out,
                                                           conv2_inC, conv2_inH, conv2_inW,
                                                           conv2_outC, conv2_outH, conv2_outW,
                                                           conv2_kernel, conv2_kernel,
                                                           conv2_stride, conv2_pad);
    cudaDeviceSynchronize();

    // ------------------------------
    // レイヤー2: Max Pooling (2x2, stride=2)
    // 出: (32,104,104)
    // ------------------------------
    int pool2_kernel = 2, pool2_stride = 2;
    int pool2_outH = conv2_outH / 2; // 104
    int pool2_outW = conv2_outW / 2; // 104
    int pool2_outC = conv2_outC;
    int pool2_outSize = pool2_outC * pool2_outH * pool2_outW;
    float* d_pool2_out;
    cudaMalloc(&d_pool2_out, pool2_outSize * sizeof(float));
    dim3 blockDimPool2(16, 16);
    dim3 gridDimPool2((pool2_outW + blockDimPool2.x - 1) / blockDimPool2.x,
                      (pool2_outH + blockDimPool2.y - 1) / blockDimPool2.y,
                      pool2_outC);
    maxpool_forward_kernel<<<gridDimPool2, blockDimPool2>>>(d_conv2_out, d_pool2_out,
                                                             pool2_outC, conv2_outH, conv2_outW,
                                                             pool2_outH, pool2_outW,
                                                             pool2_kernel, pool2_stride);
    cudaDeviceSynchronize();

    // ------------------------------
    // レイヤー3: 検出ヘッド (Detection Head)
    // 入: (32,104,104) → 出: (1,104,104)
    // 1x1 畳み込み, stride=1, パディングなし
    // ------------------------------
    int conv3_inC = pool2_outC;
    int conv3_inH = pool2_outH;
    int conv3_inW = pool2_outW;
    int conv3_outC = 1;  // 人間検出の確率マップ
    int conv3_kernel = 1, conv3_stride = 1, conv3_pad = 0;
    int conv3_outH = conv3_inH;
    int conv3_outW = conv3_inW;
    int conv3_weight_size = conv3_outC * conv3_inC * conv3_kernel * conv3_kernel;
    float* h_conv3_weight = new float[conv3_weight_size];
    for (int i = 0; i < conv3_weight_size; i++) {
        h_conv3_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    float* d_conv3_weight;
    cudaMalloc(&d_conv3_weight, conv3_weight_size * sizeof(float));
    cudaMemcpy(d_conv3_weight, h_conv3_weight, conv3_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    float* h_conv3_bias = new float[conv3_outC];
    h_conv3_bias[0] = 0;
    float* d_conv3_bias;
    cudaMalloc(&d_conv3_bias, conv3_outC * sizeof(float));
    cudaMemcpy(d_conv3_bias, h_conv3_bias, conv3_outC * sizeof(float), cudaMemcpyHostToDevice);
    float* d_conv3_out;
    int conv3_outSize = conv3_outC * conv3_outH * conv3_outW;
    cudaMalloc(&d_conv3_out, conv3_outSize * sizeof(float));
    dim3 blockDimConv3(16, 16);
    dim3 gridDimConv3((conv3_outW + blockDimConv3.x - 1) / blockDimConv3.x,
                      (conv3_outH + blockDimConv3.y - 1) / blockDimConv3.y,
                      conv3_outC);
    conv2d_forward_kernel<<<gridDimConv3, blockDimConv3>>>(d_pool2_out, d_conv3_weight, d_conv3_bias, d_conv3_out,
                                                          conv3_inC, conv3_inH, conv3_inW,
                                                          conv3_outC, conv3_outH, conv3_outW,
                                                          conv3_kernel, conv3_kernel,
                                                          conv3_stride, conv3_pad);
    cudaDeviceSynchronize();

    // ------------------------------
    // 人間特徴検出カーネルの適用
    // conv3_out (shape: (1,104,104)) に対して閾値判定を実施
    // ------------------------------
    int gridH = conv3_outH;
    int gridW = conv3_outW;
    int detection_channels = conv3_outC; // 1
    float threshold = 0.5f;
    float* d_humanMap;
    int humanMapSize = gridH * gridW;
    cudaMalloc(&d_humanMap, humanMapSize * sizeof(float));
    dim3 blockDimHuman(16, 16);
    dim3 gridDimHuman((gridW + blockDimHuman.x - 1) / blockDimHuman.x,
                      (gridH + blockDimHuman.y - 1) / blockDimHuman.y);
    humanFeatureDetectionKernel<<<gridDimHuman, blockDimHuman>>>(d_conv3_out, d_humanMap,
                                                                   gridH, gridW, detection_channels, threshold);
    cudaDeviceSynchronize();

    // 結果をホスト側へ転送：humanMap は各グリッドセルが 1.0 (検出あり) か 0.0 (検出なし)
    float* h_humanMap = new float[humanMapSize];
    cudaMemcpy(h_humanMap, d_humanMap, humanMapSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 結果表示: detectionMap をリサイズして元画像にオーバーレイ
    Mat detectionMap(gridH, gridW, CV_32F, h_humanMap);
    Mat detectionMapResized;
    resize(detectionMap, detectionMapResized, image.size());
    Mat overlay = image.clone();
    for (int y = 0; y < overlay.rows; y++) {
        for (int x = 0; x < overlay.cols; x++) {
            float val = detectionMapResized.at<float>(y, x);
            if (val > 0.5f) {
                // 赤色オーバーレイ
                overlay.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
        }
    }

    imshow("Original Image", image);
    imshow("Detection Map", detectionMapResized);
    imshow("Overlay", overlay);
    waitKey(0);

    // クリーンアップ
    delete[] h_input;
    delete[] h_humanMap;
    delete[] h_conv1_weight;
    delete[] h_conv1_bias;
    delete[] h_conv2_weight;
    delete[] h_conv2_bias;
    delete[] h_conv3_weight;
    delete[] h_conv3_bias;
    cudaFree(d_input);
    cudaFree(d_conv1_weight);
    cudaFree(d_conv1_bias);
    cudaFree(d_conv1_out);
    cudaFree(d_pool1_out);
    cudaFree(d_conv2_weight);
    cudaFree(d_conv2_bias);
    cudaFree(d_conv2_out);
    cudaFree(d_pool2_out);
    cudaFree(d_conv3_weight);
    cudaFree(d_conv3_bias);
    cudaFree(d_conv3_out);
    cudaFree(d_humanMap);

    return 0;
} 