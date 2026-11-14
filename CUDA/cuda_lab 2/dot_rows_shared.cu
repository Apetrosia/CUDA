#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)            \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// 32 threads (one warp) cooperate on one row. For a fixed column iteration,
// neighboring lanes access neighboring floats in the same row: coalesced loads.
// Shared memory stores 32 partial sums per row and is used for the final reduction.
constexpr int WARP = 32;
constexpr int ROWS_PER_BLOCK = 8;
constexpr int BLOCK_X = WARP;
constexpr int BLOCK_Y = ROWS_PER_BLOCK;

__global__ void dot_rows_shared(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int rows, int cols) {
    __shared__ float partial[ROWS_PER_BLOCK][WARP];

    const int lane = threadIdx.x;
    const int local_row = threadIdx.y;
    const int row = blockIdx.x * ROWS_PER_BLOCK + local_row;

    float sum = 0.0f;
    if (row < rows) {
        const int base = row * cols;
        for (int j = lane; j < cols; j += WARP) {
            sum += A[base + j] * B[base + j];
        }
    }

    partial[local_row][lane] = sum;
    __syncthreads();

    if (lane == 0 && row < rows) {
        float total = 0.0f;
        #pragma unroll
        for (int k = 0; k < WARP; ++k)
            total += partial[local_row][k];
        C[row] = total;
    }
}

static std::vector<float> make_random(std::size_t n, std::uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(gen);
    return v;
}

static std::vector<float> cpu_reference(const std::vector<float>& A,
                                        const std::vector<float>& B,
                                        int rows, int cols) {
    std::vector<float> C(rows);
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        const std::size_t base = static_cast<std::size_t>(r) * cols;
        for (int j = 0; j < cols; ++j)
            sum += static_cast<double>(A[base + j]) * B[base + j];
        C[r] = static_cast<float>(sum);
    }
    return C;
}

static double max_abs_error(const std::vector<float>& x,
                            const std::vector<float>& ref) {
    double e = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        e = std::max(e, std::abs(static_cast<double>(x[i]) - ref[i]));
    return e;
}

int main(int argc, char** argv) {
    int rows = 4096;
    int cols = 4096;
    int iters = 100;
    std::uint32_t seed = 12345;

    if (argc > 1) rows = std::stoi(argv[1]);
    if (argc > 2) cols = std::stoi(argv[2]);
    if (argc > 3) iters = std::stoi(argv[3]);
    if (argc > 4) seed = static_cast<std::uint32_t>(std::stoul(argv[4]));

    if (rows <= 0 || cols <= 0 || iters <= 0) {
        std::cerr << "Usage: " << argv[0]
                  << " [rows] [cols] [iters] [seed]\n";
        return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(rows) * cols;
    std::vector<float> hA = make_random(n, seed);
    std::vector<float> hB = make_random(n, seed + 1);
    std::vector<float> hC(rows, 0.0f);
    const std::vector<float> hRef = cpu_reference(hA, hB, rows, cols);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const int grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

    dot_rows_shared<<<grid, block>>>(dA, dB, dC, rows, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i)
        dot_rows_shared<<<grid, block>>>(dA, dB, dC, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_total = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, start, stop));
    const double ms = ms_total / iters;

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, rows * sizeof(float), cudaMemcpyDeviceToHost));

    const double abs_err = max_abs_error(hC, hRef);
    const double tol = 2e-3 * std::sqrt(static_cast<double>(cols)) + 1e-4;
    const bool ok = abs_err <= tol;

    const double flops = 2.0 * static_cast<double>(rows) * cols;
    const double ideal_bytes =
        (2.0 * static_cast<double>(n) + rows) * sizeof(float);
    const double gflops = flops / (ms * 1e6);
    const double ideal_gbs = ideal_bytes / (ms * 1e6);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Matrix: " << rows << " x " << cols << " (row-major)\n";
    std::cout << "Grid/Block: " << grid << " x (" << BLOCK_X << ", "
              << BLOCK_Y << ", 1) = " << BLOCK_X * BLOCK_Y << " threads/block\n";
    std::cout << "Dynamic shared memory: 0 bytes\n";
    std::cout << "Static shared memory: "
              << sizeof(float) * ROWS_PER_BLOCK * WARP << " bytes/block\n";
    std::cout << "Kernel average: " << ms << " ms\n";
    std::cout << "GFLOP/s (algorithmic): " << gflops << '\n';
    std::cout << "Ideal memory traffic: " << ideal_bytes / 1e9 << " GB\n";
    std::cout << "Ideal GB/s: " << ideal_gbs << '\n';
    std::cout << "Max abs error: " << abs_err << " (tol " << tol << ")\n";
    std::cout << "Validation: " << (ok ? "PASS" : "FAIL") << '\n';

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
