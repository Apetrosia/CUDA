#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> // <-- 1. Добавили это

// Простейшее ядро: каждый поток складывает по одному элементу
__global__ void addKernel(int *c, const int *a, const int *b) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    // 2. Добавили это: переключаем консоль на UTF-8 и скрываем сообщение об этом (>nul)
    system("chcp 65001 >nul"); 

    const int size = 5;
    const int a[size] = {1, 2, 3, 4, 5};
    const int b[size] = {10, 20, 30, 40, 50};
    int c[size] = {0};

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    // 1. Выделяем память на GPU
    cudaMalloc((void**)&dev_c, size * sizeof(int));
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));

    // 2. Копируем данные с CPU на GPU
    cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Запускаем ядро (1 блок, 5 потоков)
    addKernel<<<1, 5>>>(dev_c, dev_a, dev_b);

    // Проверяем, не упало ли ядро
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Ошибка запуска ядра: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 4. Копируем результат обратно на CPU
    cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 5. Выводим результат
    std::cout << "Результат сложения на GPU: {";
    for (int i = 0; i < size; i++) {
        std::cout << c[i] << (i != size - 1 ? ", " : "");
    }
    std::cout << "}" << std::endl;

    if (c[0] == 11 && c[1] == 22 && c[2] == 33 && c[3] == 44 && c[4] == 55) {
        std::cout << "Ура! CUDA работает как швейцарские часы!" << std::endl;
    } else {
        std::cout << "Что-то пошло не так. Данные не совпадают." << std::endl;
    }

    // 6. Чистим память
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    std::cout << "Нажмите Enter для выхода..." << std::endl;
    std::cin.get();

    return 0;
}