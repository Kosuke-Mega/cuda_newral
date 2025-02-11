#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <opencv2/opencv.hpp>
#include <fstream>  // 事前学習済みの重み読み込み用

using namespace std;
using namespace cv;

// 定数
#define INPUT_SIZE 28
#define KERNEL_SIZE 5
#define STRIDE 2
#define PADDING 2

// 今回は Pooling後 14x14=196 をFC層に入力、FC出力を10次元とする(例: MNIST風)
#define POOL_SIZE (INPUT_SIZE / 2) // =14
#define FC_IN_SIZE (POOL_SIZE * POOL_SIZE) // 14*14=196
#define FC_OUT_SIZE 1  // バイナリ分類：犬 or 非犬

// カーネル関数(1ch用)
// __global__ void conv2d(float *input, float *output, float *kernel,
//                        int input_size, int kernel_size, int stride, int padding) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < input_size && y < input_size) {
//         float sum = 0.0f;
//         for (int i = 0; i < kernel_size; i++) {
//             for (int j = 0; j < kernel_size; j++) {
//                 int in_y = y + i - padding;
//                 int in_x = x + j - padding;
//                 if (in_x >= 0 && in_y >= 0 && in_x < input_size && in_y < input_size) {
//                     int input_idx = in_y * input_size + in_x;
//                     sum += input[input_idx] * kernel[i * kernel_size + j];
//                 }
//             }
//         }
//         int out_idx = y * input_size + x;
//         output[out_idx] = sum;
//     }
// }

// カーネル関数(幅、高さ、RGB用)
__global__ void conv2d_forward_kernel(const float* in, const float* weight, const float* bias,
                                      float* out,
                                      int inC, int inH, int inW,
                                      int outC, int outH, int outW,
                                      int kernelH, int kernelW,
                                      int stride, int pad) {
    int oc = blockIdx.z;  // 出力チャネル
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if(oc < outC && oy < outH && ox < outW){
        float sum = 0.0f;
        for(int ic = 0; ic < inC; ic++){
            for(int ky = 0; ky < kernelH; ky++){
                for(int kx = 0; kx < kernelW; kx++){
                    int iy = oy * stride - pad + ky;
                    int ix = ox * stride - pad + kx;
                    if(iy >= 0 && ix >= 0 && iy < inH && ix < inW){
                        float v = in[(ic * inH + iy) * inW + ix];
                        float w = weight[ (((oc * inC) + ic) * kernelH + ky) * kernelW + kx ];
                        sum += v * w;
                    }
                }
            }
        }
        if(bias != nullptr) { // バイアスが NULL なら何も加算しない
            sum += bias[oc];
        }
        // ReLU
        if(sum < 0) sum = 0;
        out[(oc * outH + oy)*outW + ox] = sum;
    }
}



// プーリング関数(2x2,stride=2のmax-pool)
// __global__ void max_pooling(float *input, float *output,
//                             int in_size, int kernel_size, int stride) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     if (x < (in_size/stride) && y < (in_size/stride)) {
//         float max_val = -3.402823466e+38F; // -FLT_MAX
//         // 2x2の領域をプーリングする
//         for (int i = 0; i < kernel_size; i++) {
//             for (int j = 0; j < kernel_size; j++) {
//                 int in_x = x * stride + j;
//                 int in_y = y * stride + i;
//                 int input_idx = in_y * in_size + in_x;
//                 if (in_x < in_size && in_y < in_size) {
//                     max_val = fmaxf(max_val, input[input_idx]);
//                 }
//             }
//         }
//         int out_idx = y * (in_size/stride) + x;
//         output[out_idx] = max_val;
//     }
// }

// プーリング関数
__global__ void maxpool_forward_kernel(const float* in, float* out,
                                       int C, int inH, int inW,
                                       int outH, int outW,
                                       int kernel, int stride) {
    int c = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if(c < C && oy < outH && ox < outW){
        float maxv = -1e30f; // -FLT_MAX
        for(int ky=0; ky<kernel; ky++){
            for(int kx=0; kx<kernel; kx++){
                int iy = oy*stride + ky;
                int ix = ox*stride + kx;
                if(iy<inH && ix<inW){
                    float v = in[(c*inH + iy)*inW + ix];
                    if(v>maxv) maxv=v;
                }
            }
        }
        out[(c*outH + oy)*outW + ox] = maxv;
    }
}

// Flatten関数
__global__ void flatten_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;

    if (idx < total) {
        // c, y, x を計算
        int c = idx / (H * W);
        int rem = idx % (H * W);
        int y = rem / W;
        int x = rem % W;

        // 入力 in は (C,H,W) で連続だと想定: index = (c*H + y)*W + x
        // out[idx] にコピーするだけ
        out[idx] = in[(c * H + y) * W + x];
    }
}


// 全結合関数
// __global__ void fully_connected(float *input, float *output, float *weight,
//                                 int input_size, int output_size)
// {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < output_size) {
//         float sum = 0.0f;
//         for (int j = 0; j < input_size; j++) {
//             sum += input[j] * weight[j * output_size + i];
//         }
//         output[i] = sum;
//     }
// }

__global__ void fc_forward_kernel(const float* __restrict__ in,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ out,
                                  int in_size,    // Flatten後のベクトル長
                                  int out_size)   // FCの出力次元
{
    int o = blockIdx.x * blockDim.x + threadIdx.x; // 出力ノード index
    if (o < out_size) {
        float sum = 0.0f;
        // 入力ベクトル(in_size個)との積和
        for (int i = 0; i < in_size; i++) {
            sum += in[i] * weight[i * out_size + o];
        }
        // バイアス加算
        sum += bias[o];

        out[o] = sum;
    }
}

// シグモイド関数を修正（2値分類用）
__global__ void sigmoid_kernel(float* __restrict__ data, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}



// softmax関数
__global__ void softmax(float *input, float *output, int size)
{
    // グローバルスレッドID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        // 1スレッドでまとめてソフトマックスを計算 (サイズが小さいのでOK)
        // もしサイズが大きいならスレッド並列化する実装もあり
        float max_val = -3.402823466e+38F; // -FLT_MAX
        for (int j = 0; j < size; j++) {
            if (input[j] > max_val) max_val = input[j];
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < size; j++) {
            sum_exp += expf(input[j] - max_val);
        }
        for (int j = 0; j < size; j++) {
            output[j] = expf(input[j] - max_val) / sum_exp;
        }
    }
}

// クロスエントロピー損失関数を追加
__global__ void cross_entropy_loss_kernel(float* pred, float* target, float* loss, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        float epsilon = 1e-7f;
        *loss = 0.0f;
        for (int i = 0; i < size; i++) {
            *loss -= (target[i] * logf(pred[i] + epsilon) + 
                     (1.0f - target[i]) * logf(1.0f - pred[i] + epsilon));
        }
    }
}

// --- 新規追加: 全結合層のバックプロパゲーション更新カーネル ---
// このカーネルは、入力層（flatten後の値）とFC層出力の誤差を用いて，
// 全結合層の重みとバイアスを更新します。
__global__ void fc_backward_update_kernel(const float* fc_in, float* fc_weight, float* fc_bias,
                                            float error, float learning_rate, int in_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < in_size) {
        // 重み更新: w = w - learning_rate * (error * fc_in[i])
        fc_weight[i] -= learning_rate * error * fc_in[i];
    }
    if (i == 0) {
        // バイアス更新: b = b - learning_rate * error
        fc_bias[0] -= learning_rate * error;
    }
}

int main()
{
    // CUDA情報表示
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA Device: %s\n", prop.name);

    // 画像読み込みと表示
    std::string image_path = "img/31357138_s.jpg";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()) {
        std::cout << "Error: Could not read the image." << std::endl;
        return -1;
    }

    // 元画像を表示
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Original Image", image);

    // 画像のリサイズ（CNNの入力サイズに合わせる）
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(INPUT_SIZE, INPUT_SIZE));
    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    // === データセットの読み込み ===
    // 実際には、ここでトレーニング用データとテスト用データの画像パスのリストを用意し
    // それぞれから画像を読み込んで処理を行います。
    // 以下はサンプルのプレースホルダー例です。
    std::vector<std::string> trainingImagePaths = { "img/train1.jpg", "img/train2.jpg", "img/train3.jpg", "img/train4.jpg", "img/train5.jpg" };
    std::vector<std::string> testImagePaths = { "img/test1.jpg", "img/test2.jpg", "img/test3.jpg" };
    // ※ 現在のコードでは、resized_image（1枚の画像）を使用していますが、
    //     実際の運用では trainingImagePaths を利用して各バッチごとに異なる学習用画像を読み込む必要があります。

    // 2) GPUメモリ確保
    float *d_input, *d_conv, *d_pool, *d_flatten, *d_fc_in, *d_fc_out, *d_softmax;
    int pool_size = (INPUT_SIZE / 2) * (INPUT_SIZE / 2);
    int numChannels = resized_image.channels(); // 例えば、カラー画像なら3チャネル
    cudaMalloc(&d_input,  INPUT_SIZE*INPUT_SIZE*numChannels*sizeof(float));  // height * width * channels
    cudaMalloc(&d_conv,   INPUT_SIZE*INPUT_SIZE*sizeof(float));  // conv出力 height * width
    cudaMalloc(&d_pool,   pool_size * sizeof(float));  // pool出力 height / 2 * width / 2
    cudaMalloc(&d_flatten, pool_size * sizeof(float));  // flatten出力　height * width　* color
    // 全結合 (196 -> 1)
    cudaMalloc(&d_fc_out, FC_OUT_SIZE*sizeof(float)); // 出力 1

    // --- 畳み込み層用ダミーバイアスの確保 ---
    float *d_conv_bias;
    cudaMalloc(&d_conv_bias, 1 * sizeof(float)); // 畳み込み出力は1チャネル
    cudaMemset(d_conv_bias, 0, 1 * sizeof(float));

    // 3) カーネル用の重み
    std::vector<float> h_conv_weight(KERNEL_SIZE*KERNEL_SIZE);
    std::ifstream convWeightFile("weights/conv_weights.bin", std::ios::binary);
    if(convWeightFile.is_open()) {
        convWeightFile.read(reinterpret_cast<char*>(h_conv_weight.data()), KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
        convWeightFile.close();
        std::cout << "Loaded pretrained convolution weights." << std::endl;
    } else {
        // Fallback: デフォルト値（単純なエッジ検出の例）
        float default_conv_kernel[KERNEL_SIZE*KERNEL_SIZE] = {
            0, 1, 0, 1, 0,
            0, -2, 0, -2, 0,
            0, 1, 0, 1, 0,
            0, -2, 0, -2, 0,
            1, 1, 0, 1, 1
        };
        std::copy(default_conv_kernel, default_conv_kernel + KERNEL_SIZE*KERNEL_SIZE, h_conv_weight.begin());
        std::cout << "Using default convolution kernel values." << std::endl;
    }
    float *d_kernel;
    cudaMalloc(&d_kernel, KERNEL_SIZE*KERNEL_SIZE*sizeof(float));
    cudaMemcpy(d_kernel, h_conv_weight.data(), KERNEL_SIZE*KERNEL_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    // 4) 全結合用の重み (196->1)
    // 196*1個
    int fc_weight_count = FC_IN_SIZE * FC_OUT_SIZE; // 196
    std::vector<float> h_fc_weight(fc_weight_count);
    std::ifstream fcWeightFile("weights/fc_weights.bin", std::ios::binary);
    if(fcWeightFile.is_open()){
        cout << "fcWeightFile is open" << endl;
        fcWeightFile.read(reinterpret_cast<char*>(h_fc_weight.data()), fc_weight_count*sizeof(float));
        fcWeightFile.close();
        std::cout << "Loaded pretrained fully-connected weights." << std::endl;
    } else {
        cout << "fcWeightFile is not open" << endl;
        for (int i = 0; i < fc_weight_count; i++){
            h_fc_weight[i] = ((float)rand()/RAND_MAX)*0.01f; // 小さめの乱数
        }
        std::cout << "Using random fully-connected weights." << std::endl;
    }

    float *d_fc_weight;
    cudaMalloc(&d_fc_weight, fc_weight_count*sizeof(float));
    cudaMemcpy(d_fc_weight, h_fc_weight.data(), fc_weight_count*sizeof(float), cudaMemcpyHostToDevice);

    // 5) ホスト->GPU 転送
    // OpenCVで読み込まれた resized_image は HWC (height, width, channels) 形式となっているため、
    // CNNが想定するCHW形式に変換する
    std::vector<float> hostInput(INPUT_SIZE * INPUT_SIZE * numChannels);
    if(numChannels == 3){
        // カラー画像の場合 (BGR順になっていますが、ここではそのまま数値として扱う)
        for (int y = 0; y < INPUT_SIZE; ++y) {
            for (int x = 0; x < INPUT_SIZE; ++x) {
                // resized_image.at<cv::Vec3f>(y, x) は各画素の3チャネル(B, G, R)を持つ
                cv::Vec3f pixel = resized_image.at<cv::Vec3f>(y, x);
                for (int c = 0; c < numChannels; ++c) {
                    // CHW形式：各チャネルごとに連続した領域にコピーする
                    hostInput[c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] = pixel[c];
                }
            }
        }
    } else {
        // グレースケールの場合はそのままコピー
        for (int y = 0; y < INPUT_SIZE; ++y) {
            for (int x = 0; x < INPUT_SIZE; ++x) {
                hostInput[y * INPUT_SIZE + x] = resized_image.at<float>(y, x);
            }
        }
    }
    // 必要サイズ分をGPUへコピー
    cudaMemcpy(d_input, hostInput.data(), hostInput.size() * sizeof(float), cudaMemcpyHostToDevice);

    /*
     * 今後の改善点:
     * - 複数の畳み込み層やプーリング層を組み合わせた、より深いネットワークアーキテクチャの実装を検討してください。
     * - 事前学習済みの重みを用いることで、実用的な犬検出が期待できます。
     * - 訓練時はデータ拡張やバッチ処理を実装し、モデルの汎化性能向上を図ると良いでしょう。
     * - トレーニング中には、クロスエントロピー損失や精度などの評価指標の計算を追加してください。
     */

    // 全結合層のバイアスをグローバルに確保（トレーニング＆推論両用）
    float *d_fc_bias;
    cudaMalloc(&d_fc_bias, FC_OUT_SIZE * sizeof(float));
    cudaMemset(d_fc_bias, 0, FC_OUT_SIZE*sizeof(float));

    // ========== トレーニングフェーズ（学習用画像によるシミュレーション） ==========
    // ※ 注意: このループはforward-passと損失計算のシミュレーションのみであり、
    //         実際の学習では、ここで各バッチごとに trainingImagePaths から画像と正解ラベルを読み込み、
    //         バックプロパゲーションによる重み更新を実装する必要があります。
    int numEpochs = 25000;
    int batchSize = 1;
    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    float h_loss = 0.0f;

    // ※注意：本トレーニングループはforward-passと損失計算のシミュレーションのみであり、
    //     バックプロパゲーションによる重み更新は実装していません。
    for (int epoch = 0; epoch < numEpochs; epoch++) {
        float epochLoss = 0.0f;
        for (int batch = 0; batch < batchSize; batch++) {
            // Forward Pass: 畳み込み
            {
                dim3 block(16,16);
                dim3 grid((INPUT_SIZE+block.x-1)/block.x, (INPUT_SIZE+block.y-1)/block.y);
                conv2d_forward_kernel<<<grid, block>>>(d_input, d_kernel, d_conv_bias, d_conv,
                    numChannels, INPUT_SIZE, INPUT_SIZE, 1, INPUT_SIZE, INPUT_SIZE,
                    KERNEL_SIZE, KERNEL_SIZE, 1, PADDING);
                cudaDeviceSynchronize();
            }
            // Forward Pass: Max Pooling
            {
                dim3 block(16,16);
                dim3 grid((POOL_SIZE+block.x-1)/block.x, (POOL_SIZE+block.y-1)/block.y);
                maxpool_forward_kernel<<<grid, block>>>(d_conv, d_pool, 3, INPUT_SIZE, INPUT_SIZE,
                    POOL_SIZE, POOL_SIZE, 2, 2);
                cudaDeviceSynchronize();
            }
            // Forward Pass: Fully Connected
            {
                dim3 block(16);
                dim3 grid((FC_OUT_SIZE+block.x-1)/block.x);
                fc_forward_kernel<<<grid, block>>>(d_pool, d_fc_weight, d_fc_bias, d_fc_out,
                    FC_IN_SIZE, FC_OUT_SIZE);
                cudaDeviceSynchronize();
            }
            // --- Forward Pass: Sigmoid ---
            // d_sigmoid_out をループ外で利用できるように宣言・確保
            float *d_sigmoid_out;
            cudaMalloc(&d_sigmoid_out, FC_OUT_SIZE * sizeof(float));

            {
                dim3 block(16);
                dim3 grid((FC_OUT_SIZE + block.x - 1) / block.x);
                sigmoid_kernel<<<grid, block>>>(d_fc_out, d_sigmoid_out, FC_OUT_SIZE);
                cudaDeviceSynchronize();

                // クロスエントロピー損失の計算（ターゲットラベルは1.0と仮定）
                float h_target = 1.0f;
                float *d_target;
                cudaMalloc(&d_target, sizeof(float));
                cudaMemcpy(d_target, &h_target, sizeof(float), cudaMemcpyHostToDevice);

                cross_entropy_loss_kernel<<<1, 1>>>(d_sigmoid_out, d_target, d_loss, FC_OUT_SIZE);
                cudaDeviceSynchronize();
                float batch_loss = 0.0f;
                cudaMemcpy(&batch_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
                epochLoss += batch_loss;

                cudaFree(d_target);
                // ※ d_sigmoid_out はここでは解放せず後続のバックプロパゲーションで利用します。
            }
            // --- FC層に対するバックプロパゲーション ---
            float learning_rate = 0.001f; // 学習率（必要に応じて調整）
            // FC層のシグモイド出力 (d_sigmoid_out) とターゲット値との誤差を計算
            float h_prediction = 0.0f;
            cudaMemcpy(&h_prediction, d_sigmoid_out, sizeof(float), cudaMemcpyDeviceToHost);
            float target = 1.0f; // 例: 犬であればターゲットは1.0
            float error = h_prediction - target; // 誤差 = (p - target)

            // FC層の入力は max pool の出力 (d_pool) であり、サイズは FC_IN_SIZE (例:196) です
            // カーネルを呼び出してd_fc_weight, d_fc_bias を更新
            dim3 block_update(256);
            dim3 grid_update((FC_IN_SIZE + block_update.x - 1) / block_update.x);
            fc_backward_update_kernel<<<grid_update, block_update>>>(d_pool, d_fc_weight, d_fc_bias,
                                                                   error, learning_rate, FC_IN_SIZE);
            cudaDeviceSynchronize();

            // バックプロパゲーションで利用後、d_sigmoid_out を解放
            cudaFree(d_sigmoid_out);
        }
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " average loss: " << (epochLoss / batchSize) << std::endl;
        }
    }
    cudaFree(d_loss);

    // ========== 検出フェーズ（テスト用画像による犬の検出） ==========
    // ※ 本来は testImagePaths の画像を利用して評価する必要があります。
    {
        dim3 block(16,16);
        dim3 grid((INPUT_SIZE+block.x-1)/block.x, (INPUT_SIZE+block.y-1)/block.y);
        conv2d_forward_kernel<<<grid, block>>>(d_input, d_kernel, d_conv_bias, d_conv, numChannels, INPUT_SIZE, INPUT_SIZE, 1, INPUT_SIZE, INPUT_SIZE, KERNEL_SIZE, KERNEL_SIZE, 1, PADDING);
        cudaDeviceSynchronize();
    }

    // conv2d出力
    std::vector<float> h_conv(INPUT_SIZE*INPUT_SIZE);
    cudaMemcpy(h_conv.data(), d_conv, INPUT_SIZE*INPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

    // h_convデータ出力
    cv::Mat conv_image(INPUT_SIZE, INPUT_SIZE, CV_32F);
    for(int y=0; y<INPUT_SIZE; y++){
        for(int x=0; x<INPUT_SIZE; x++){
            conv_image.at<float>(y, x) = h_conv[y*INPUT_SIZE + x];
            printf("%d:%f ", x, conv_image.at<float>(y, x));
        }
        printf("\n");
    }

    // max pooling (2x2, stride=2)
    {
        dim3 block(16,16);
        dim3 grid( (POOL_SIZE+block.x-1)/block.x, (POOL_SIZE+block.y-1)/block.y );
        maxpool_forward_kernel<<<grid, block>>>(d_conv, d_pool, 3, INPUT_SIZE, INPUT_SIZE, POOL_SIZE, POOL_SIZE, 2, 2);
        cudaDeviceSynchronize();
    }

    // fully connected (pool出力=196次元 -> 1次元) using pre-allocated d_fc_bias
    {
        dim3 block(16);
        dim3 grid((FC_OUT_SIZE + block.x -1)/block.x);
        fc_forward_kernel<<<grid, block>>>(d_pool, d_fc_weight, d_fc_bias, d_fc_out, FC_IN_SIZE, FC_OUT_SIZE);
        cudaDeviceSynchronize();
    }

    // シグモイド関数の適用
    float *d_sigmoid_out;
    cudaMalloc(&d_sigmoid_out, FC_OUT_SIZE * sizeof(float));
    {
        dim3 block(16);
        dim3 grid((FC_OUT_SIZE + block.x - 1) / block.x);
        sigmoid_kernel<<<grid, block>>>(d_fc_out, d_sigmoid_out, FC_OUT_SIZE);
        cudaDeviceSynchronize();
    }

    // 結果の取得と表示
    float h_result;
    cudaMemcpy(&h_result, d_sigmoid_out, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Detection Result: " << std::fixed << std::setprecision(4) 
              << h_result << std::endl;
    std::cout << "Prediction: " << (h_result > 0.5f ? "Dog" : "Not a Dog") 
              << std::endl;

    // 検出結果を画像に表示
    cv::Mat output_image = image.clone();
    std::string result_text = "Dog: " + std::to_string(int(h_result * 100)) + "%";
    cv::putText(output_image, result_text, cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    
    cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Detection Result", output_image);
    cv::waitKey(0);

    // GPUメモリ解放
    cudaFree(d_input);
    cudaFree(d_conv);
    cudaFree(d_pool);
    cudaFree(d_kernel);
    cudaFree(d_fc_weight);
    cudaFree(d_fc_out);
    cudaFree(d_sigmoid_out);
    cudaFree(d_conv_bias);

    return 0;
}

/*
 * 注意:
 * このコードは犬検出の実験用サンプルです。実際の利用に際しては以下の追加作業が必要です：
 * 1. 事前学習済みの重みの用意：
 *    - 例: weights/conv_weights.bin, weights/fc_weights.bin などの外部ファイルから重みを読み込む実装を追加。
 * 2. より深いネットワーク構造の実装：
 *    - 現在は単一の畳み込み層・プーリング層・全結合層ですが、必要に応じて複数層の実装を検討。
 * 3. データ拡張やバッチ処理の追加：
 *    - 訓練時にデータ拡張やバッチ処理を組み込むことで、モデルの汎化性能向上が期待されます。
 * 4. モデル評価指標の追加：
 *    - クロスエントロピー損失などの評価指標計算をトレーニング時に実装してください。
 */
