#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

#define INPUT_SIZE 64    // 入力画像サイズ（64x64）
#define CHANNELS 3       // カラー画像 (RGB)
#define CONV_OUT_CH 8    // 畳み込み層の出力チャネル数
#define FC_OUT_SIZE 2    // 全結合層出力（性別分類：[男性, 女性]の確率）

// max pooling の設定 (2x2, stride=2)
#define POOL_KERNEL 2
#define POOL_STRIDE 2
#define POOL_OUT_SIZE (INPUT_SIZE / POOL_STRIDE) // 64/2 = 32

// -------------------------------------------------------------------------------
// [CUDA Kernel] 畳み込み層 (Conv2D) forward - ReLU活性化内部含む
// -------------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(const float* in, const float* weight, const float* bias,
                                      float* out,
                                      int inC, int inH, int inW,
                                      int outC, int outH, int outW,
                                      int kernelH, int kernelW,
                                      int stride, int pad)
{
    int oc = blockIdx.z;  // 出力チャネル
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc < outC && oy < outH && ox < outW) {
        float sum = 0.0f;
        for (int ic = 0; ic < inC; ic++){
            for (int ky = 0; ky < kernelH; ky++){
                for (int kx = 0; kx < kernelW; kx++){
                    int iy = oy * stride - pad + ky;
                    int ix = ox * stride - pad + kx;
                    if (iy >= 0 && ix >= 0 && iy < inH && ix < inW){
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
// [CUDA Kernel] MaxPooling forward (2x2, stride=2)
// -------------------------------------------------------------------------------
__global__ void maxpool_forward_kernel(const float* in, float* out,
                                       int C, int inH, int inW,
                                       int outH, int outW,
                                       int kernel, int stride)
{
    int c = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C && oy < outH && ox < outW) {
        float max_val = -1e30f;
        for (int ky = 0; ky < kernel; ky++){
            for (int kx = 0; kx < kernel; kx++){
                int iy = oy * stride + ky;
                int ix = ox * stride + kx;
                if (iy < inH && ix < inW) {
                    float v = in[(c * inH + iy) * inW + ix];
                    if (v > max_val)
                        max_val = v;
                }
            }
        }
        out[(c * outH + oy) * outW + ox] = max_val;
    }
}

// -------------------------------------------------------------------------------
// [CUDA Kernel] 全結合層 Forward
// -------------------------------------------------------------------------------
__global__ void fc_forward_kernel(const float* in, const float* weight, const float* bias, 
                                  float* out, int in_size, int out_size)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o < out_size) {
        float sum = 0.0f;
        for (int i = 0; i < in_size; i++){
            sum += in[i] * weight[i * out_size + o];
        }
        sum += bias[o];
        out[o] = sum;
    }
}

// -------------------------------------------------------------------------------
// [CUDA Kernel] Sigmoid活性化
// -------------------------------------------------------------------------------
__global__ void sigmoid_kernel(const float* in, float* out, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size){
        float x = in[idx];
        out[idx] = 1.0f / (1.0f + expf(-x));
    }
}

// -------------------------------------------------------------------------------
// [CUDA Kernel] 全結合層のバックプロパゲーション更新 (fc層のみ更新)
// -------------------------------------------------------------------------------
__global__ void fc_backward_update_kernel(const float* fc_in, float* fc_weight, float* fc_bias,
                                            float error, float learning_rate, int in_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_size) {
        fc_weight[i] -= learning_rate * error * fc_in[i];
    }
    if (i == 0) {
        fc_bias[0] -= learning_rate * error;
    }
}

// -------------------------------------------------------------------------------
// [CUDA Kernel] 全結合層のバックプロパゲーション更新 (マルチクラス用)
// 更新: fc_weightとfc_biasを、pred - targetを用いて更新 (ワンホット表現)
// ※ ここでは、fc層の各出力ニューロンに対して誤差 (prediction - target) を加味して更新します。
__global__ void fc_backward_update_kernel_multi(const float* fc_in, float* fc_weight, float* fc_bias,
                                                  const float* error, float learning_rate, int in_size, int out_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_size) {
        for (int o = 0; o < out_size; o++) {
            fc_weight[i * out_size + o] -= learning_rate * error[o] * fc_in[i];
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int o = 0; o < out_size; o++) {
            fc_bias[o] -= learning_rate * error[o];
        }
    }
}

// -------------------------------------------------------------------------------
// [CUDA Kernel] Softmax活性化 (全結合層出力を正規化して各クラスの確率に)
// ※ 簡易実装（出力サイズが小さいことを前提としています）
__global__ void softmax_kernel(const float* in, float* out, int size)
{
    // このカーネルは1ブロック1スレッドで実行される前提です
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_val = in[0];
        for (int i = 1; i < size; i++) {
            if (in[i] > max_val)
                max_val = in[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++){
            float e = expf(in[i] - max_val);
            out[i] = e; // 一時保存
            sum += e;
        }
        for (int i = 0; i < size; i++){
            out[i] /= sum;
        }
    }
}

int main(int argc, char** argv)
{
    // OpenCVでカメラ画像の取得
    VideoCapture cap(0);
    if (!cap.isOpened()){
        cerr << "カメラがオープンできません" << endl;
        return -1;
    }

    // CNNモデルのパラメータ設定
    // --- 畳み込み層 ---
    // 入力: (3,64,64), 出力: (CONV_OUT_CH,64,64)
    int conv_inC = CHANNELS;
    int conv_inH = INPUT_SIZE;
    int conv_inW = INPUT_SIZE;
    int conv_outC = CONV_OUT_CH;
    int conv_kernel = 3;
    int conv_stride = 1;
    int conv_pad = 1;
    int conv_outH = conv_inH;
    int conv_outW = conv_inW;
    int conv_weight_size = conv_outC * conv_inC * conv_kernel * conv_kernel;
    float* h_conv_weight = new float[conv_weight_size];
    for (int i = 0; i < conv_weight_size; i++){
        h_conv_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    float* h_conv_bias = new float[conv_outC];
    memset(h_conv_bias, 0, conv_outC * sizeof(float));

    // --- Fully Connected層 ---
    // MaxPooling後のサイズ: (CONV_OUT_CH, INPUT_SIZE/2, INPUT_SIZE/2) = (CONV_OUT_CH,32,32)
    int pool_outH = conv_outH / POOL_STRIDE; // 64/2 = 32
    int pool_outW = conv_outW / POOL_STRIDE; // 32
    int fc_in_size = conv_outC * pool_outH * pool_outW;
    int fc_out_size = FC_OUT_SIZE;
    int fc_weight_size = fc_in_size * fc_out_size;
    float* h_fc_weight = new float[fc_weight_size];
    for (int i = 0; i < fc_weight_size; i++){
        h_fc_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }
    float* h_fc_bias = new float[fc_out_size];
    h_fc_bias[0] = 0.0f;

    // GPUメモリの確保とモデルパラメータの転送（畳み込み層 & fc層）
    float *d_conv_weight, *d_conv_bias;
    cudaMalloc(&d_conv_weight, conv_weight_size * sizeof(float));
    cudaMalloc(&d_conv_bias, conv_outC * sizeof(float));
    cudaMemcpy(d_conv_weight, h_conv_weight, conv_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_bias, h_conv_bias, conv_outC * sizeof(float), cudaMemcpyHostToDevice);

    float *d_fc_weight, *d_fc_bias;
    cudaMalloc(&d_fc_weight, fc_weight_size * sizeof(float));
    cudaMalloc(&d_fc_bias, fc_out_size * sizeof(float));
    cudaMemcpy(d_fc_weight, h_fc_weight, fc_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_bias, h_fc_bias, fc_out_size * sizeof(float), cudaMemcpyHostToDevice);

    // 各中間変数用GPUメモリ確保
    int input_size = CHANNELS * INPUT_SIZE * INPUT_SIZE;
    int conv_out_size = conv_outC * conv_outH * conv_outW;
    int pool_out_size = conv_out_size / (POOL_STRIDE * POOL_STRIDE);
    float *d_input, *d_conv_out, *d_pool_out, *d_fc_out, *d_sigmoid_out, *d_loss;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_conv_out, conv_out_size * sizeof(float));
    cudaMalloc(&d_pool_out, pool_out_size * sizeof(float));
    cudaMalloc(&d_fc_out, fc_out_size * sizeof(float));
    cudaMalloc(&d_sigmoid_out, fc_out_size * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));

    // トレーニング設定
    const int num_iterations = 5000;
    float learning_rate = 0.001f;
    float target_label = 1.0f; // 仮の正解ラベル（例：常に「人が写っている」と仮定）

    // ----- YOLO Training開始 (学習データ：男性写真10枚 + 女性写真10枚 計20枚) -------------
    vector<string> trainFiles;
    for (int i = 1; i <= 100; i++){
        trainFiles.push_back("./train/male" + to_string(i) + ".jpg");
    }
    for (int i = 1; i <= 100; i++){
        trainFiles.push_back("./train/female" + to_string(i) + ".jpg");
    }

    // d_error用のGPUメモリ確保（fc層出力サイズ分：2要素）
    float *d_error;
    cudaMalloc(&d_error, fc_out_size * sizeof(float));

    int num_epochs = 100; // 学習エポック数（例）
    cout << "YOLO Training start..." << endl;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (const auto &filepath : trainFiles) {
            Mat train_img = imread(filepath);
            if (train_img.empty()){
                cout << "Could not open " << filepath << endl;
                continue;
            }

            // 前処理：リサイズ、float変換＋正規化
            Mat train_resized;
            resize(train_img, train_resized, Size(INPUT_SIZE, INPUT_SIZE));
            train_resized.convertTo(train_resized, CV_32F, 1.0/255.0);

            // HWC→CHW形式変換
            float* train_h_frame = new float[input_size];
            for (int c = 0; c < CHANNELS; c++){
                for (int y = 0; y < INPUT_SIZE; y++){
                    for (int x = 0; x < INPUT_SIZE; x++){
                        train_h_frame[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                            train_resized.at<Vec3f>(y, x)[c];
                    }
                }
            }
            cudaMemcpy(d_input, train_h_frame, input_size*sizeof(float), cudaMemcpyHostToDevice);
            delete[] train_h_frame;

            // Forward Pass: 畳み込み層
            dim3 t_blockDimConv(16, 16);
            dim3 t_gridDimConv((conv_outW + t_blockDimConv.x - 1) / t_blockDimConv.x,
                               (conv_outH + t_blockDimConv.y - 1) / t_blockDimConv.y,
                               conv_outC);
            conv2d_forward_kernel<<<t_gridDimConv, t_blockDimConv>>>(d_input, d_conv_weight, d_conv_bias, d_conv_out,
                                                                      conv_inC, conv_inH, conv_inW,
                                                                      conv_outC, conv_outH, conv_outW,
                                                                      conv_kernel, conv_kernel,
                                                                      conv_stride, conv_pad);
            cudaDeviceSynchronize();

            // Forward Pass: Max Pooling
            dim3 t_blockDimPool(16, 16);
            dim3 t_gridDimPool((POOL_OUT_SIZE + t_blockDimPool.x - 1) / t_blockDimPool.x,
                               (POOL_OUT_SIZE + t_blockDimPool.y - 1) / t_blockDimPool.y,
                               conv_outC);
            maxpool_forward_kernel<<<t_gridDimPool, t_blockDimPool>>>(d_conv_out, d_pool_out,
                                                                       conv_outC, conv_outH, conv_outW,
                                                                       POOL_OUT_SIZE, POOL_OUT_SIZE,
                                                                       POOL_KERNEL, POOL_STRIDE);
            cudaDeviceSynchronize();

            // Forward Pass: 全結合層
            int t_fc_block = 256;
            int t_fc_grid = (fc_out_size + t_fc_block - 1) / t_fc_block;
            fc_forward_kernel<<<t_fc_grid, t_fc_block>>>(d_pool_out, d_fc_weight, d_fc_bias, d_fc_out,
                                                         fc_in_size, fc_out_size);
            cudaDeviceSynchronize();

            // Forward Pass: Softmax活性化
            softmax_kernel<<<1, 1>>>(d_fc_out, d_sigmoid_out, fc_out_size);
            cudaDeviceSynchronize();

            // 予測結果取得（2クラス：男性, 女性）
            float pred[2];
            cudaMemcpy(pred, d_sigmoid_out, 2*sizeof(float), cudaMemcpyDeviceToHost);

            // ターゲットラベルの設定：ファイル名に "male" が含まれていれば [1, 0]、それ以外は [0, 1]
            float target[2];
            if (filepath.find("male") != string::npos) {
                target[0] = 1.0f; target[1] = 0.0f;
            } else {
                target[0] = 0.0f; target[1] = 1.0f;
            }

            // 誤差計算 (各出力に対して prediction - target)
            float h_error[2];
            for (int o = 0; o < 2; o++){
                h_error[o] = pred[o] - target[o];
            }
            cudaMemcpy(d_error, h_error, 2*sizeof(float), cudaMemcpyHostToDevice);

            // 全結合層の多クラス用バックプロパゲーション更新
            int update_block_multi = 256;
            int update_grid_multi = (fc_in_size + update_block_multi - 1) / update_block_multi;
            fc_backward_update_kernel_multi<<<update_grid_multi, update_block_multi>>>(d_pool_out, d_fc_weight, d_fc_bias,
                                                                                        d_error, learning_rate, fc_in_size, fc_out_size);
            cudaDeviceSynchronize();
        }
        cout << "Epoch " << epoch << " completed." << endl;
    }
    cout << "YOLO Training finished." << endl;
    // ----- YOLO Training終了 -------------

    // ----- テストフェーズ (カメラ画像から取得して判別) -------------
    VideoCapture testCap(0);
    if (!testCap.isOpened()){
         cerr << "No test camera" << endl;
         return -1;
    }
    cout << "Start test phase" << endl;
    while(true){
         Mat frame;
         testCap >> frame;
         if (frame.empty()){
             cerr << "No frame" << endl;
             break;
         }

         // HOGDescriptorを用いて人物検出
         vector<Rect> detections;
         static HOGDescriptor hog;
         hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
         hog.detectMultiScale(frame, detections);

         // 検出された各人物領域について性別分類
         for (const Rect &rect : detections){
             // ROI切り出し
             Mat personROI = frame(rect);
             Mat resizedROI;
             resize(personROI, resizedROI, Size(INPUT_SIZE, INPUT_SIZE));
             resizedROI.convertTo(resizedROI, CV_32F, 1.0/255.0);

             // HWCからCHW形式へ変換
             float* person_data = new float[input_size];
             for (int c = 0; c < CHANNELS; c++){
                 for (int y = 0; y < INPUT_SIZE; y++){
                     for (int x = 0; x < INPUT_SIZE; x++){
                         person_data[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                             resizedROI.at<Vec3f>(y, x)[c];
                     }
                 }
             }
             cudaMemcpy(d_input, person_data, input_size * sizeof(float), cudaMemcpyHostToDevice);
             delete[] person_data;

             // Forward Pass: 畳み込み層
             dim3 t_blockDimConv(16, 16);
             dim3 t_gridDimConv((conv_outW + t_blockDimConv.x - 1) / t_blockDimConv.x,
                                (conv_outH + t_blockDimConv.y - 1) / t_blockDimConv.y,
                                conv_outC);
             conv2d_forward_kernel<<<t_gridDimConv, t_blockDimConv>>>(d_input, d_conv_weight, d_conv_bias, d_conv_out,
                                                                       conv_inC, conv_inH, conv_inW,
                                                                       conv_outC, conv_outH, conv_outW,
                                                                       conv_kernel, conv_kernel,
                                                                       conv_stride, conv_pad);
             cudaDeviceSynchronize();

             // Forward Pass: Max Pooling
             dim3 poolBlock(16, 16);
             dim3 poolGrid((POOL_OUT_SIZE + poolBlock.x - 1)/poolBlock.x, (POOL_OUT_SIZE + poolBlock.y - 1)/poolBlock.y, conv_outC);
             maxpool_forward_kernel<<<poolGrid, poolBlock>>>(d_conv_out, d_pool_out,
                                                              conv_outC, conv_outH, conv_outW,
                                                              POOL_OUT_SIZE, POOL_OUT_SIZE,
                                                              POOL_KERNEL, POOL_STRIDE);
             cudaDeviceSynchronize();

             // Forward Pass: 全結合層
             int fc_block = 256;
             int fc_grid = (fc_out_size + fc_block - 1) / fc_block;
             fc_forward_kernel<<<fc_grid, fc_block>>>(d_pool_out, d_fc_weight, d_fc_bias, d_fc_out,
                                                      fc_in_size, fc_out_size);
             cudaDeviceSynchronize();

             // Forward Pass: Softmax活性化
             softmax_kernel<<<1, 1>>>(d_fc_out, d_sigmoid_out, fc_out_size);
             cudaDeviceSynchronize();

             float gender_probs[2];
             cudaMemcpy(gender_probs, d_sigmoid_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

             // 結果のオーバーレイ表示：検出領域上に性別比率を描画
             string gender_text = "Male: " + to_string(gender_probs[0]*100) + "%, Female: " + to_string(gender_probs[1]*100) + "%";
             putText(frame, gender_text, Point(rect.x, rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,0,0), 2);
             rectangle(frame, rect, Scalar(0,0,255), 2);

             // gender_textの内容をコンソールに出力
             cout << gender_text << endl;
         }

         imshow("Test Camera", frame);
         if (waitKey(30)==27) break; // ESCキーで終了
    }
    testCap.release();

    // クリーンアップ
    cudaFree(d_input); 
    cudaFree(d_conv_out); 
    cudaFree(d_pool_out);
    cudaFree(d_fc_out); 
    cudaFree(d_sigmoid_out); 
    cudaFree(d_loss);
    cudaFree(d_conv_weight); 
    cudaFree(d_conv_bias);
    cudaFree(d_fc_weight); 
    cudaFree(d_fc_bias);
    delete[] h_conv_weight; 
    delete[] h_conv_bias;
    delete[] h_fc_weight; 
    delete[] h_fc_bias;
    
    return 0;
} 