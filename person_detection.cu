#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>  // GPUカーネル用ヘッダ

using namespace cv;
using namespace cv::dnn;
using namespace std;


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
// ※ 注意：OpenCV DNNは推論専用ですが、ここではGPU側でダミーの学習（重み更新）処理を行う例を示します。

// CUDAカーネル：ダミーの重み更新処理（各要素 weight -= learningRate * gradient）
__global__ void weightUpdateKernel(float* weights, const float* gradients, float learningRate, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        weights[idx] -= learningRate * gradients[idx];
}

void updateWeightsAndBiases(Net &net, const Mat &dummyLoss)
{
    const int weightSize = 1024; // 例: 1024要素のダミー重み配列
    float *d_weights, *d_gradients;
    cudaMalloc(&d_weights, weightSize * sizeof(float));
    cudaMalloc(&d_gradients, weightSize * sizeof(float));

    // ホスト側でダミーの重みと勾配を用意（実際はネットワークのパラメータ及び勾配が対象）
    float *h_weights = new float[weightSize];
    float *h_gradients = new float[weightSize];
    for (int i = 0; i < weightSize; i++) {
        h_weights[i] = 1.0f;    // 初期重み（ダミー値）
        h_gradients[i] = 0.1f;  // ダミー勾配（dummyLossに基づく想定値）
    }

    cudaMemcpy(d_weights, h_weights, weightSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradients, h_gradients, weightSize * sizeof(float), cudaMemcpyHostToDevice);

    float learningRate = 0.001f;
    int blockSize = 256;
    int numBlocks = (weightSize + blockSize - 1) / blockSize;
    weightUpdateKernel<<<numBlocks, blockSize>>>(d_weights, d_gradients, learningRate, weightSize);
    cudaDeviceSynchronize();

    // 結果をホストに転送して確認（最初の5要素のみ表示）
    cudaMemcpy(h_weights, d_weights, weightSize * sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Updated Weights (first 5): ";
    for (int i = 0; i < 5; i++) {
        cout << h_weights[i] << " ";
    }
    cout << endl;

    delete[] h_weights;
    delete[] h_gradients;
    cudaFree(d_weights);
    cudaFree(d_gradients);
}

int main()
{
    // ① YOLOモデルの設定（あらかじめ男性:女性=5:5となるような事前学習済みモデル）
    string modelConfiguration = "custom-yolov3.cfg";    // YOLOのネットワーク構造設定ファイル
    string modelWeights = "custom-yolov3.weights";        // 学習済みの重みファイル（男性と女性の比率が5:5）

    // ネットワークを読み込み、CUDAバックエンドを使用する設定にする
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);

    // ② カメラ映像のキャプチャ
    VideoCapture cap(0);  // デフォルトカメラ
    if (!cap.isOpened())
    {
        cerr << "カメラがオープンできません" << endl;
        return -1;
    }

    // メインループ：各フレームに対して処理を実施
    while (true)
    {
        Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // ③ フレームをYOLO用のblobに変換
        Mat blob;
        blobFromImage(frame, blob, 1 / 255.0, Size(416, 416), Scalar(), true, false);
        net.setInput(blob);

        // ④ 出力層の名前を取得して推論実施
        vector<String> outNames = net.getUnconnectedOutLayersNames();
        vector<Mat> outs;
        net.forward(outs, outNames);

        // ⑤ 検出結果の処理
        float confThreshold = 0.5;
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        // 各出力レイヤの各検出についてループ
        for (size_t i = 0; i < outs.size(); i++)
        {
            // 出力はN×(5+クラス数)の形状（最初の4はバウンディングボックス、次は信頼度と各クラスのスコア）
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; j++, data += outs[i].cols)
            {
                // 各クラスのスコアを取得（例ではindex 5以降）
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                // クラスID 0 が "person" と仮定。信頼度が閾値以上であれば検出
                if (confidence > confThreshold && classIdPoint.x == 0)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width   = (int)(data[2] * frame.cols);
                    int height  = (int)(data[3] * frame.rows);
                    int left    = centerX - width / 2;
                    int top     = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        // ⑥ Non-Maximum Suppression（NMS）で重複する検出を除去
        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, 0.4, indices);

        // ⑦ アノテーション：検出領域に矩形とラベルを描画
        for (size_t i = 0; i < indices.size(); i++)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            rectangle(frame, box, Scalar(0, 0, 255), 2); // 赤枠
            string label = "Person: " + to_string(confidences[idx]);
            putText(frame, label, Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }

        // ⑧ カメラ映像と検出結果を表示
        imshow("Person Detection", frame);

        // ⑨ GPU側の疑似学習更新を周期的に実行（例: 100フレーム毎）
        static int frameCount = 0;
        frameCount++;
        if (frameCount % 100 == 0) {
            Mat dummyLoss = Mat::ones(1, 1, CV_32F);  // ダミー損失
            updateWeightsAndBiases(net, dummyLoss);
        }

        // 'ESC'キーで終了
        if (waitKey(1) == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
} 