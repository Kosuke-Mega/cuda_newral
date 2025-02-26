#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cfloat>  // FLT_MIN 等を使うため

using namespace cv;
using namespace std;

#define INPUT_SIZE 64            // 入力画像サイズ (64x64)
#define CHANNELS 3               // RGB
#define CONV_OUT_CH 8            // 畳み込み層出力チャネル数
#define FC_OUT_SIZE 2            // 全結合層出力（例：2クラス）
#define POOL_KERNEL 2            // MaxPooling カーネルサイズ
#define POOL_STRIDE 2
#define POOL_OUT_SIZE (INPUT_SIZE / POOL_STRIDE) // 64/2 = 32

// -------------------------------------------------------------------------------
// CUDA カーネル（実装は省略。以前のコードと同様のものを利用してください）
// -------------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(const float* in, const float* weight, const float* bias,
                                      float* out,
                                      int inC, int inH, int inW,
                                      int outC, int outH, int outW,
                                      int kernelH, int kernelW,
                                      int stride, int pad)
{
    // 出力座標 (out_x, out_y) と出力チャネル oc を決定
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z;
    
    if (out_x < outW && out_y < outH && oc < outC) {
         float sum = 0.0f;
         for (int ic = 0; ic < inC; ic++) {
              for (int kh = 0; kh < kernelH; kh++) {
                   for (int kw = 0; kw < kernelW; kw++) {
                        int in_y = out_y * stride - pad + kh;
                        int in_x = out_x * stride - pad + kw;
                        if (in_y >= 0 && in_y < inH && in_x >= 0 && in_x < inW) {
                             int in_index = ic * (inH * inW) + in_y * inW + in_x;
                             // 重みは (outC, inC, kernelH, kernelW) の順列とする
                             int weight_index = oc * (inC * kernelH * kernelW) + ic * (kernelH * kernelW) + kh * kernelW + kw;
                             sum += in[in_index] * weight[weight_index];
                        }
                   }
              }
         }
         sum += bias[oc];
         int out_index = oc * (outH * outW) + out_y * outW + out_x;
         out[out_index] = sum;
    }
}

__global__ void maxpool_forward_kernel(const float* in, float* out,
                                       int C, int inH, int inW,
                                       int outH, int outW,
                                       int kernel, int stride)
{
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;  // out_x
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;  // out_y
    int c = blockIdx.z;  // チャンネルインデックス
    
    if (out_x < outW && out_y < outH && c < C) {
         float max_val = -FLT_MAX;
         for (int ky = 0; ky < kernel; ky++) {
              for (int kx = 0; kx < kernel; kx++) {
                   int in_y = out_y * stride + ky;
                   int in_x = out_x * stride + kx;
                   int index = c * (inH * inW) + in_y * inW + in_x;
                   float val = in[index];
                   if (val > max_val)
                        max_val = val;
              }
         }
         int out_index = c * (outH * outW) + out_y * outW + out_x;
         out[out_index] = max_val;
    }
}

__global__ void fc_forward_kernel(const float* in, const float* weight, const float* bias,
                                  float* out, int in_size, int out_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_size) {
         float sum = 0.0f;
         // 重みは (out_size x in_size) の行列とする
         for (int i = 0; i < in_size; i++) {
              sum += in[i] * weight[idx * in_size + i];
         }
         sum += bias[idx];
         out[idx] = sum;
    }
}

__global__ void softmax_kernel(const float* in, float* out, int size)
{
    // 簡易実装: 1ブロック1スレッドで実行する前提
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float max_val = in[0];
        for (int i = 1; i < size; i++) {
            if (in[i] > max_val)
                max_val = in[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++){
            float e = expf(in[i] - max_val);
            out[i] = e;
            sum += e;
        }
        for (int i = 0; i < size; i++){
            out[i] /= sum;
        }
    }
}

// -------------------------------------------------------------------------------
// ※ このカーネルは単純な全結合層の backward 更新用の例です
// 元の未実装のカーネルを以下で実装します。
// -------------------------------------------------------------------------------
__global__ void fc_backward_update_kernel_multi(const float* pool_out, float* fc_weight, float* fc_bias,
                                                  const float* error, float lr,
                                                  int fc_in_size, int fc_out_size)
{
    // 全結合層の重みは (fc_out_size x fc_in_size)
    int total_weights = fc_out_size * fc_in_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 重みの更新: 各出力ユニット j と各入力ユニット i に対して更新
    //  重み勾配: grad = pool_out[i] * error[j]
    //  更新式: fc_weight[t] -= lr * grad,  t = j * fc_in_size + i
    for (int t = idx; t < total_weights; t += blockDim.x * gridDim.x) {
         int j = t / fc_in_size;    // 出力ユニット (行) のインデックス
         int i = t % fc_in_size;    // 入力ユニット (列) のインデックス
         float grad = pool_out[i] * error[j];
         fc_weight[t] -= lr * grad;
    }
    
    // バイアスの更新: 各出力ユニット j に対して
    for (int j = idx; j < fc_out_size; j += blockDim.x * gridDim.x) {
         fc_bias[j] -= lr * error[j];
    }
}

// -------------------------------------------------------------------------------
// Main 関数：3プロセスに分けたシーケンス
// -------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // ---------------------------------------
    // Phase 1: 画像データの取得 (Data Acquisition)
    // ---------------------------------------
    VideoCapture cap(0);
    if (!cap.isOpened()){
         cerr << "Camera cannot be opened" << endl;
         return -1;
    }
    // ウィンドウ生成（必要に応じて）
    namedWindow("Acquisition", WINDOW_NORMAL);
    cout << "Phase 1: Data Acquisition" << endl;
    vector<Mat> roiImages;  // 取得したROI画像を保持するベクタ
    int requiredDetections = 100; // 例として100回検出するまで取得する
    int detectionCount = 0;

    // Haar Cascadeによる顔検出モデルを読み込み
    CascadeClassifier faceCascade;
    if(!faceCascade.load("C:/opencv/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml")){
        cerr << "Failed to load Haar Cascade model." << endl;
        return -1;
    }

    while(detectionCount < requiredDetections){
         Mat frame;
         cap >> frame;
         if(frame.empty()){
            cout << "Frame is empty" << endl;
            break;
         } 

         // Haar Cascadeを用いた顔検出処理
         Mat gray;
         cvtColor(frame, gray, COLOR_BGR2GRAY);
         equalizeHist(gray, gray);
         // 調整後のパラメータで顔検出（minNeighbors=5, minSize=Size(60,60)）
         vector<Rect> raw_faces;
         faceCascade.detectMultiScale(gray, raw_faces, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(60, 60));

         // 検出された矩形のうち、アスペクト比やサイズでフィルタリングを実施
         vector<Rect> faces;
         for (const Rect& r : raw_faces) {
             float aspectRatio = (float)r.width / r.height;
             if (aspectRatio < 0.8 || aspectRatio > 1.2)
                 continue;
             if (r.width < 60 || r.height < 60)
                 continue;
             faces.push_back(r);
         }

         if(faces.empty()){
              cout << "No valid face detected in this frame." << endl;
              imshow("Acquisition", frame);
              if(waitKey(30)==27) break;
              continue;
         }
         Rect bestRect = *max_element(faces.begin(), faces.end(),
                         [](const Rect &a, const Rect &b){ return a.area() < b.area(); });
         Mat personROI = frame(bestRect).clone();

         // ROI画像を保存
         imwrite("roi_sample_" + to_string(detectionCount+1) + ".png", personROI);
         roiImages.push_back(personROI);
         detectionCount++;
         cout << "Detection count: " << detectionCount << " / " << requiredDetections << endl;

         // 検出領域（bestRect）に矩形を描画して、カメラ画像上で検知結果を表示
         rectangle(frame, bestRect, Scalar(0, 255, 0), 2);
         imshow("Acquisition", frame);
         if(waitKey(30)==27) break;
    }
    cout << "Acquisition completed: " << roiImages.size() << " samples captured." << endl;


    // ---------------------------------------
    // Phase 2: 検出した画像のベクトルを取得 (Feature Extraction)
    // ---------------------------------------
    cout << "Phase 2: Feature Extraction" << endl;
    // ROI画像から直接ベクトルを作成（HOGによる特徴抽出は実施しない）
    vector<vector<float>> featureVectors; // 各ROI画像から変換した入力データ（floatベクトル）
    for (size_t i = 0; i < roiImages.size(); i++){
         Mat roi = roiImages[i];
         Mat resized;
         resize(roi, resized, Size(INPUT_SIZE, INPUT_SIZE));
         resized.convertTo(resized, CV_32F, 1.0/255.0);
 
         // CHW形式への変換（前提：入力はRGB形式）
         vector<float> inputVector(CHANNELS * INPUT_SIZE * INPUT_SIZE, 0);
         for (int c = 0; c < CHANNELS; c++){
             for (int y = 0; y < INPUT_SIZE; y++){
                 for (int x = 0; x < INPUT_SIZE; x++){
                     inputVector[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                         resized.at<Vec3f>(y, x)[c];
                 }
             }
         }
         featureVectors.push_back(inputVector);
    }
    cout << "Feature Extraction completed: " << featureVectors.size() << " feature vectors extracted." << endl;

    // ---------------------------------------
    // Phase 3: ベクトルを学習 (Training)
    // ---------------------------------------
    cout << "Phase 3: Training" << endl;
    // CNNモデルのパラメータ設定
    int conv_inC = CHANNELS;
    int conv_inH = INPUT_SIZE;
    int conv_inW = INPUT_SIZE;
    int conv_outC = CONV_OUT_CH;  // 8
    int conv_kernel = 3;          // 3x3
    int conv_stride = 1;
    int conv_pad = 1;
    int conv_outH = conv_inH;     // パディングにより同じ
    int conv_outW = conv_inW;
    int fc_in_size = conv_outC * POOL_OUT_SIZE * POOL_OUT_SIZE;
    int fc_out_size = FC_OUT_SIZE;  // 2クラス
    float learning_rate = 0.001f;
    
    // GPUメモリの確保
    float *d_input, *d_conv_out, *d_pool_out, *d_fc_out, *d_sigmoid_out, *d_error;
    cudaMalloc(&d_input, CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_conv_out, conv_outC * conv_outH * conv_outW * sizeof(float));
    cudaMalloc(&d_pool_out, conv_outC * POOL_OUT_SIZE * POOL_OUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc_out, fc_out_size * sizeof(float));
    cudaMalloc(&d_sigmoid_out, fc_out_size * sizeof(float));
    cudaMalloc(&d_error, fc_out_size * sizeof(float));
    
    // 畳み込み層の重み・バイアス
    float *d_conv_weight, *d_conv_bias;
    int conv_weight_size = conv_outC * conv_inC * conv_kernel * conv_kernel;
    cudaMalloc(&d_conv_weight, conv_weight_size * sizeof(float));
    cudaMalloc(&d_conv_bias, conv_outC * sizeof(float));
    float *h_conv_weight = new float[conv_weight_size];
    for (int i = 0; i < conv_weight_size; i++){
         h_conv_weight[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }
    float *h_conv_bias = new float[conv_outC];
    memset(h_conv_bias, 0, conv_outC * sizeof(float));
    cudaMemcpy(d_conv_weight, h_conv_weight, conv_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_bias, h_conv_bias, conv_outC * sizeof(float), cudaMemcpyHostToDevice);
    
    // 全結合層の重み・バイアス
    float *d_fc_weight, *d_fc_bias;
    int fc_weight_size = fc_in_size * fc_out_size;
    cudaMalloc(&d_fc_weight, fc_weight_size * sizeof(float));
    cudaMalloc(&d_fc_bias, fc_out_size * sizeof(float));
    float *h_fc_weight = new float[fc_weight_size];
    for (int i = 0; i < fc_weight_size; i++){
         h_fc_weight[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.01f;
    }
    float *h_fc_bias = new float[fc_out_size];
    for (int i = 0; i < fc_out_size; i++){
         h_fc_bias[i] = 0.0f;
    }
    cudaMemcpy(d_fc_weight, h_fc_weight, fc_weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_bias, h_fc_bias, fc_out_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // 学習ループ： featureVectors に対して逐次学習を実施
    int numEpochs = 100;
    for (int epoch = 0; epoch < numEpochs; epoch++){
         cout << "Epoch " << epoch+1 << " / " << numEpochs << endl;
         for (size_t i = 0; i < featureVectors.size(); i++){
              // 入力データ (CHW形式の float vector) を d_input に転送
              cudaMemcpy(d_input, featureVectors[i].data(), CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float),
                         cudaMemcpyHostToDevice);
              // Forward Pass: 畳み込み層
              dim3 blockDimConv(16,16);
              dim3 gridDimConv((conv_outW + blockDimConv.x - 1) / blockDimConv.x,
                               (conv_outH + blockDimConv.y - 1) / blockDimConv.y,
                               conv_outC);
              conv2d_forward_kernel<<<gridDimConv, blockDimConv>>>(d_input, d_conv_weight, d_conv_bias, d_conv_out,
                                                                   CHANNELS, INPUT_SIZE, INPUT_SIZE,
                                                                   conv_outC, conv_outH, conv_outW,
                                                                   conv_kernel, conv_kernel, conv_stride, conv_pad);
              cudaDeviceSynchronize();
              
              // Forward Pass: Max Pooling
              dim3 blockDimPool(16,16);
              dim3 gridDimPool((POOL_OUT_SIZE + blockDimPool.x - 1) / blockDimPool.x,
                               (POOL_OUT_SIZE + blockDimPool.y - 1) / blockDimPool.y,
                               conv_outC);
              maxpool_forward_kernel<<<gridDimPool, blockDimPool>>>(d_conv_out, d_pool_out,
                                                                     conv_outC, conv_outH, conv_outW,
                                                                     POOL_OUT_SIZE, POOL_OUT_SIZE,
                                                                     POOL_KERNEL, POOL_STRIDE);
              cudaDeviceSynchronize();
              
              // Forward Pass: 全結合層
              int fc_block = 256;
              int fc_grid = (fc_out_size + fc_block - 1) / fc_block;
              fc_forward_kernel<<<fc_grid, fc_block>>>(d_pool_out, d_fc_weight, d_fc_bias, d_fc_out,
                                                       conv_outC * POOL_OUT_SIZE * POOL_OUT_SIZE, fc_out_size);
              cudaDeviceSynchronize();
              
              // Softmax活性化
              softmax_kernel<<<1,1>>>(d_fc_out, d_sigmoid_out, fc_out_size);
              cudaDeviceSynchronize();
              
              // 予測結果取得
              float pred[FC_OUT_SIZE];
              cudaMemcpy(pred, d_sigmoid_out, fc_out_size * sizeof(float), cudaMemcpyDeviceToHost);
              float target[FC_OUT_SIZE] = {1.0f, 0.0f};
              
              // クロスエントロピー損失の計算
              float sampleLoss = 0.0f;
              for (int j = 0; j < fc_out_size; j++){
                  sampleLoss += - target[j] * log(pred[j] + 1e-8f);
              }
              cout << "Epoch " << epoch+1 << ", Sample " << i+1 
                   << " Cross Entropy Loss: " << sampleLoss << endl;
              
              // 誤差計算とバックプロパゲーション更新
              float h_error[FC_OUT_SIZE];
              for (int o = 0; o < fc_out_size; o++){
                   h_error[o] = pred[o] - target[o];
              }
              cudaMemcpy(d_error, h_error, fc_out_size * sizeof(float), cudaMemcpyHostToDevice);
              
              // Backward update: 全結合層の更新（conv層の更新は省略）
              int update_block = 256;
              int update_grid = (conv_outC * POOL_OUT_SIZE * POOL_OUT_SIZE + update_block - 1) / update_block;
              fc_backward_update_kernel_multi<<<update_grid, update_block>>>(d_pool_out, d_fc_weight, d_fc_bias,
                                                                              d_error, learning_rate,
                                                                              conv_outC * POOL_OUT_SIZE * POOL_OUT_SIZE, fc_out_size);
              cudaDeviceSynchronize();
         }
    }
    cout << "Training completed." << endl;

    // ---------------------------------------
    // Phase 4: Testing
    // ---------------------------------------
    cout << "Phase 4: Testing" << endl;
    string names[FC_OUT_SIZE] = {"Mega", "Bob"};
    
    // すでに cap, faceCascade などは使用可能なので、
    // 学習フェーズと同じ GPU バッファとカーネル設定を再利用します。
    // ここで forward pass 用の grid/block 設定を定義
    dim3 blockDimConv(16,16);
    dim3 gridDimConv((conv_outW + blockDimConv.x - 1) / blockDimConv.x,
                     (conv_outH + blockDimConv.y - 1) / blockDimConv.y,
                     conv_outC);
    dim3 blockDimPool(16,16);
    dim3 gridDimPool((POOL_OUT_SIZE + blockDimPool.x - 1) / blockDimPool.x,
                     (POOL_OUT_SIZE + blockDimPool.y - 1) / blockDimPool.y,
                     conv_outC);
    int fc_block = 256;
    int fc_grid = (fc_out_size + fc_block - 1) / fc_block;
    
    // テストフェーズ：カメラから連続フレームを取得して判別
    while (true) {
         Mat test_frame;
         cap >> test_frame;
         if (test_frame.empty()) break;
         
         // 顔検出（Haar Cascade を使用）
         Mat test_gray;
         cvtColor(test_frame, test_gray, COLOR_BGR2GRAY);
         equalizeHist(test_gray, test_gray);
         // 調整後のパラメータで顔検出
         vector<Rect> raw_test_faces;
         faceCascade.detectMultiScale(test_gray, raw_test_faces, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(60, 60));

         // アスペクト比やサイズでフィルタリング
         vector<Rect> test_faces;
         for (const Rect& r : raw_test_faces) {
             float aspectRatio = (float)r.width / r.height;
             if (aspectRatio < 0.8 || aspectRatio > 1.2)
                 continue;
             if (r.width < 60 || r.height < 60)
                 continue;
             test_faces.push_back(r);
         }

         if (test_faces.empty()) {
              imshow("Test", test_frame);
              if(waitKey(30)==27) break;
              // リセット：安定性カウンタもリセットする
              static int consistentDetectionCount = 0;
              consistentDetectionCount = 0;
              continue;
         }
         // 複数検出された場合、面積が最大の領域を採用
         Rect best_test = *max_element(test_faces.begin(), test_faces.end(),
                                [](const Rect &a, const Rect &b){ return a.area() < b.area(); });
         Mat test_roi = test_frame(best_test).clone();
         
         // 前処理：リサイズ、正規化、CHW変換（RGBとして扱う）
         Mat test_resized;
         resize(test_roi, test_resized, Size(INPUT_SIZE, INPUT_SIZE));
         test_resized.convertTo(test_resized, CV_32F, 1.0/255.0);
         vector<float> test_vector(CHANNELS * INPUT_SIZE * INPUT_SIZE, 0);
         for (int c = 0; c < CHANNELS; c++){
             for (int y = 0; y < INPUT_SIZE; y++){
                 for (int x = 0; x < INPUT_SIZE; x++){
                      test_vector[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                          test_resized.at<Vec3f>(y, x)[c];
                 }
             }
         }
         // GPUへ転送
         cudaMemcpy(d_input, test_vector.data(), CHANNELS * INPUT_SIZE * INPUT_SIZE * sizeof(float),
                    cudaMemcpyHostToDevice);
         // Forward Pass: 畳み込み層
         conv2d_forward_kernel<<<gridDimConv, blockDimConv>>>(d_input, d_conv_weight, d_conv_bias, d_conv_out,
                                                              CHANNELS, INPUT_SIZE, INPUT_SIZE,
                                                              conv_outC, conv_outH, conv_outW,
                                                              conv_kernel, conv_kernel, conv_stride, conv_pad);
         cudaDeviceSynchronize();
         // Forward Pass: Max Pooling層
         maxpool_forward_kernel<<<gridDimPool, blockDimPool>>>(d_conv_out, d_pool_out,
                                                                conv_outC, conv_outH, conv_outW,
                                                                POOL_OUT_SIZE, POOL_OUT_SIZE,
                                                                POOL_KERNEL, POOL_STRIDE);
         cudaDeviceSynchronize();
         // Forward Pass: 全結合層
         fc_forward_kernel<<<fc_grid, fc_block>>>(d_pool_out, d_fc_weight, d_fc_bias, d_fc_out,
                                                  conv_outC * POOL_OUT_SIZE * POOL_OUT_SIZE, fc_out_size);
         cudaDeviceSynchronize();
         // Softmax
         softmax_kernel<<<1,1>>>(d_fc_out, d_sigmoid_out, fc_out_size);
         cudaDeviceSynchronize();
         
         float pred_test[FC_OUT_SIZE];
         cudaMemcpy(pred_test, d_sigmoid_out, fc_out_size * sizeof(float), cudaMemcpyDeviceToHost);
         // シンプルに確率の高い方を予測クラスとする
         int predicted_class = (pred_test[1] > pred_test[0]) ? 1 : 0;
         string predicted_name = names[predicted_class];
         
         // 時系列フィルタリングによる検出の安定化
         static int consistentDetectionCount = 0;
         static Rect prevDetection;

         const int positionThreshold = 20; // 位置の変化を許容するピクセル数
         if (consistentDetectionCount > 0) {
             if (abs(best_test.x - prevDetection.x) < positionThreshold &&
                 abs(best_test.y - prevDetection.y) < positionThreshold)
             {
                 consistentDetectionCount++;
             } else {
                 consistentDetectionCount = 1;
             }
         } else {
             consistentDetectionCount = 1;
         }
         // 保存：次のフレームと比較するため
         prevDetection = best_test;

         // 連続フレームでの検出が安定していなければ、今回の検出を無視する
         if (consistentDetectionCount < 3) {
             imshow("Test", test_frame);
             if(waitKey(30)==27) break;
             continue;
         }

         // 結果を画面に表示
         rectangle(test_frame, best_test, Scalar(0,255,0), 2);
         putText(test_frame, predicted_name, Point(best_test.x, best_test.y - 10),
                 FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0,255,0), 2);
         imshow("Test", test_frame);
         if(waitKey(30)==27) break;
    }
    
    cap.release();
    return 0;
} 