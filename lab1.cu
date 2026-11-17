#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <chrono>   // Для замера времени CPU
#include <vector>   // Для удобного управления буферами CPU

using namespace cv;
using namespace std;

// Ядро Гаусса 3x3. Используем unsigned char* для полной совместимости с OpenCV
__global__ void gaussianBlurKernel(const unsigned char* in, unsigned char* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    int sumB = 0, sumG = 0, sumR = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            
            int idx = (ny * width + nx) * 3;
            int k = kernel[dy + 1][dx + 1];
            
            sumB += in[idx] * k;
            sumG += in[idx + 1] * k;
            sumR += in[idx + 2] * k;
        }
    }

    int out_idx = (y * width + x) * 3;
    out[out_idx]     = sumB / 16;
    out[out_idx + 1] = sumG / 16;
    out[out_idx + 2] = sumR / 16;
}

int main(int argc, char** argv) {
    const string filename = "test.jpg";
    
    // 1. Загружаем изображение
    Mat image = imread(filename, IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Не удалось загрузить " << filename << ". Положи файл рядом с exe!" << endl;
        return -1;
    }
    cout << "Изображение загружено: " << image.cols << "x" << image.rows << endl;

    Mat original = image.clone(); // Сохраняем для отображения

    // 2. Получаем количество итераций
    int iterations = 0;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    } else {
        cout << "Введите количество применений размытия: ";
        cin >> iterations;
    }
    
    if (iterations <= 0) {
        cout << "Количество итераций должно быть > 0. Установлено 1." << endl;
        iterations = 1;
    }
    cout << "Будет применено размытий: " << iterations << endl;

    // 3. Делаем матрицу непрерывной
    if (!image.isContinuous()) {
        image = image.clone();
    }

    int width = image.cols;
    int height = image.rows;
    size_t bytes = width * height * 3 * sizeof(unsigned char);

    // ==========================================
    // 4. ЗАМЕР ВРЕМЕНИ НА CPU
    // ==========================================
    cout << "Выполнение на CPU..." << endl;
    vector<unsigned char> cpu_in(bytes);
    vector<unsigned char> cpu_out(bytes);
    memcpy(cpu_in.data(), image.data, bytes);

    int kernel[3][3] = { {1, 2, 1}, {2, 4, 2}, {1, 2, 1} };

    auto cpu_start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; iter++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sumB = 0, sumG = 0, sumR = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = min(max(x + dx, 0), width - 1);
                        int ny = min(max(y + dy, 0), height - 1);
                        
                        int idx = (ny * width + nx) * 3;
                        int k = kernel[dy + 1][dx + 1];
                        
                        sumB += cpu_in[idx] * k;
                        sumG += cpu_in[idx + 1] * k;
                        sumR += cpu_in[idx + 2] * k;
                    }
                }
                int out_idx = (y * width + x) * 3;
                cpu_out[out_idx]     = sumB / 16;
                cpu_out[out_idx + 1] = sumG / 16;
                cpu_out[out_idx + 2] = sumR / 16;
            }
        }
        // Пинг-понг буферов, как на GPU
        cpu_in.swap(cpu_out);
    }

    auto cpu_end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_duration = cpu_end - cpu_start;
    // ==========================================


    // ==========================================
    // 5. ЗАМЕР ВРЕМЕНИ НА GPU
    // ==========================================
    cout << "Выполнение на GPU..." << endl;
    unsigned char* d_a = nullptr;
    unsigned char* d_b = nullptr;
    
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);

    // Копируем исходные данные в d_a
    cudaMemcpy(d_a, image.data, bytes, cudaMemcpyHostToDevice);

    int threads = 256; // 1D или 2D, оставим 2D как было для наглядности, но пересчитаем
    dim3 threads2D(16, 16);
    dim3 blocks2D((width + threads2D.x - 1) / threads2D.x,
                  (height + threads2D.y - 1) / threads2D.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize(); 
    cudaEventRecord(start);

    unsigned char* d_in = d_a;
    unsigned char* d_out = d_b;
    
    for (int i = 0; i < iterations; i++) {
        gaussianBlurKernel<<<blocks2D, threads2D>>>(d_in, d_out, width, height);
        unsigned char* temp = d_in;
        d_in = d_out;
        d_out = temp;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);
    // ==========================================

    // 6. Копируем результат GPU обратно на хост
    cudaMemcpy(image.data, d_in, bytes, cudaMemcpyDeviceToHost);

    // 7. Чистим память GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    // ==========================================
    // 8. ПРОВЕРКА КОРРЕКТНОСТИ (Сверка CPU и GPU)
    // ==========================================
    bool is_match = true;
    unsigned char* gpu_result_ptr = image.data;
    // После цикла swap результат на CPU всегда в cpu_in
    for (size_t i = 0; i < bytes; ++i) {
        if (gpu_result_ptr[i] != cpu_in[i]) {
            is_match = false;
            cerr << "Расхождение на байте " << i << ": CPU=" << (int)cpu_in[i] 
                 << ", GPU=" << (int)gpu_result_ptr[i] << endl;
            break;
        }
    }
    cout << "\nРезультаты CPU и GPU совпадают: " << (is_match ? "ДА (Идеально!)" : "НЕТ (Ошибка!)") << endl;
    // ==========================================


    // 9. Выводим результат замеров
    cout << "-----------------------------------" << endl;
    cout << "Время выполнения на CPU:  " << cpu_duration.count() << " мс" << endl;
    cout << "Время выполнения на GPU:  " << gpu_time_ms << " мс" << endl;
    
    if (gpu_time_ms > 0) {
        cout << "Ускорение (Speedup):    " << cpu_duration.count() / gpu_time_ms << " x" << endl;
    }
    cout << "-----------------------------------" << endl;

    // 10. Показываем результат
    imshow("Original", original);
    imshow("Gaussian Blur (GPU)", image);
    
    imwrite("result.jpg", image);
    cout << "Результат сохранен в файл result.jpg" << endl;
    
    waitKey(0);

    return 0;
}