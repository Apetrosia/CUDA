#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <iomanip>

using namespace std;

// ==========================================
// ВАРИАНТ 1: Максимально простой и короткий
// 1 поток обрабатывает 1 строку целиком.
// ==========================================
__global__ void dotProductSimple(const float* A, const float* B, float* C, int cols) {
    int row = blockIdx.x; // Один блок = одна строка
    if (row >= gridDim.x) return;

    float sum = 0.0f;
    // Последовательный цикл по всем элементам строки
    for (int j = 0; j < cols; j++) {
        sum += A[row * cols + j] * B[row * cols + j];
    }
    C[row] = sum;
}

// ==========================================
// ВАРИАНТ 2: Быстрый, с разделяемой памятью
// Блок потоков обрабатывает одну строку, используя параллельную редукцию.
// ==========================================
__global__ void dotProductShared(const float* A, const float* B, float* C, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Выделяем разделяемую память динамически (размер задается при запуске ядра)
    extern __shared__ float sdata[];

    float sum = 0.0f;
    // Стайд (stride) позволяет обработать строки любой длины, даже если cols > blockDim.x
    for (int j = tid; j < cols; j += blockDim.x) {
        sum += A[row * cols + j] * B[row * cols + j];
    }
    
    sdata[tid] = sum;
    __syncthreads(); // Ждем, пока все потоки запишут свои частичные суммы

    // Параллельная редукция (дерево сложения) в разделяемой памяти
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Результат записывает только первый поток блока
    if (tid == 0) {
        C[row] = sdata[0];
    }
}

// ==========================================
// Эталонная функция на CPU для проверки
// ==========================================
void dotProductCPU(const float* A, const float* B, float* C, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += A[i * cols + j] * B[i * cols + j];
        }
        C[i] = sum;
    }
}

int main() {
    system("chcp 65001 >nul"); // Фикс кракозябр
    
    const int ROWS = 8192;   // Количество строк
    const int COLS = 1024;   // Количество столбцов (элементов в строке)
    const size_t bytes = ROWS * COLS * sizeof(float);
    const size_t out_bytes = ROWS * sizeof(float);

    cout << "Генерация случайных данных..." << endl;
    vector<float> h_A(ROWS * COLS), h_B(ROWS * COLS);
    for (size_t i = 0; i < ROWS * COLS; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // CPU результат для проверки
    vector<float> h_C_cpu(ROWS, 0.0f);
    dotProductCPU(h_A.data(), h_B.data(), h_C_cpu.data(), ROWS, COLS);

    // Выделение памяти на GPU
    float *d_A, *d_B, *d_C_simple, *d_C_shared;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C_simple, out_bytes);
    cudaMalloc(&d_C_shared, out_bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    // Параметры запуска
    int threads = 256;
    int blocks = ROWS;
    size_t shared_mem_size = threads * sizeof(float);

    // --- ЗАМЕР ВАРИАНТА 1 (Простой) ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    dotProductSimple<<<blocks, 1>>>(d_A, d_B, d_C_simple, COLS); // 1 поток на блок!
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_simple = 0;
    cudaEventElapsedTime(&time_simple, start, stop);

    // --- ЗАМЕР ВАРИАНТА 2 (С разделяемой памятью) ---
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    dotProductShared<<<blocks, threads, shared_mem_size>>>(d_A, d_B, d_C_shared, COLS);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_shared = 0;
    cudaEventElapsedTime(&time_shared, start, stop);

    // Копирование результатов обратно
    vector<float> h_C_simple(ROWS), h_C_shared(ROWS);
    cudaMemcpy(h_C_simple.data(), d_C_simple, out_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_shared.data(), d_C_shared, out_bytes, cudaMemcpyDeviceToHost);

    // --- ПРОВЕРКА КОРРЕКТНОСТИ ---
    bool ok_simple = true, ok_shared = true;
    float epsilon = 1e-4f;
    for (int i = 0; i < ROWS; i++) {
        if (abs(h_C_simple[i] - h_C_cpu[i]) > epsilon) ok_simple = false;
        if (abs(h_C_shared[i] - h_C_cpu[i]) > epsilon) ok_shared = false;
    }

    cout << "\n=== Результаты проверки ===" << endl;
    cout << "Простой вариант совпадает с CPU: " << (ok_simple ? "ДА" : "НЕТ") << endl;
    cout << "Быстрый вариант совпадает с CPU: " << (ok_shared ? "ДА" : "НЕТ") << endl;

    cout << "\n=== Замеры времени ===" << endl;
    cout << "Простой вариант:  " << fixed << setprecision(3) << time_simple << " мс" << endl;
    cout << "Быстрый вариант:  " << time_shared << " мс" << endl;
    cout << "Ускорение:        " << time_simple / time_shared << " x" << endl;

    // Очистка
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_simple); cudaFree(d_C_shared);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    cout << "\nПрограмма завершена. Нажмите Enter..." << endl;
    cin.get();
    return 0;
}