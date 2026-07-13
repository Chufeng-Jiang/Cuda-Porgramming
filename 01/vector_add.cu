// vector_add.cu

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

// CUDA 错误检查
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << ": " << cudaGetErrorString(error) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

//////////////////////////////////////////////////////////////
// CUDA 加法 kernel
template <typename T>
__global__ void add_kernel(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    std::size_t offset
) {
    std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x
        + offset;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}


// 使用固定 grid/block 配置完成整个向量加法
template <typename T>
void vector_add(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    const dim3& grid,
    const dim3& block
) {
    // 每次 kernel launch 最多处理的元素数量
    std::size_t step =
        static_cast<std::size_t>(grid.x) * block.x;

    // 外层循环保证全部 n 个元素都被处理
    for (std::size_t offset = 0; offset < n; offset += step) {
        // 内部启动 kernel 处理当前 step 个元素
        add_kernel<T><<<grid, block>>>(
            c,
            a,
            b,
            n,
            offset
        );

        CUDA_CHECK(cudaGetLastError());
    }
}
/////////////////////////////////////////////////////////////////

template <typename T>
__global__ void add_kernel_inner_loop(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    std::size_t step
) {
    std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    for (std::size_t i = idx; i < n; i += step) {
        c[i] = a[i] + b[i];
    }
}

template <typename T>
void vector_add_inner_loop(
    T* c,
    const T* a,
    const T* b,
    std::size_t n,
    const dim3& grid,
    const dim3& block
) {
    std::size_t step =
        static_cast<std::size_t>(grid.x) * block.x;

    add_kernel_inner_loop<T><<<grid, block>>>(
        c,
        a,
        b,
        n,
        step
    );

    CUDA_CHECK(cudaGetLastError());
}

////////////////////////////////////////////////////////////////

int main() {

    // 计算量
    constexpr std::size_t SIZE = 1 << 24;
    constexpr std::size_t BYTES = SIZE * sizeof(float);

    std::vector<float> h_a(SIZE);
    std::vector<float> h_b(SIZE);
    std::vector<float> h_c(SIZE);

    // 初始化输入数据
    for (std::size_t i = 0; i < SIZE; ++i) {
        // h_a[i] = static_cast<float>(i);
        // h_b[i] = static_cast<float>(2 * i);
        h_a[i] = static_cast<float>(1);
        h_b[i] = static_cast<float>(1);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, BYTES));
    CUDA_CHECK(cudaMalloc(&d_b, BYTES));
    CUDA_CHECK(cudaMalloc(&d_c, BYTES));

    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        BYTES,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        BYTES,
        cudaMemcpyHostToDevice
    ));

    /*
     * 可以测试不同配置：
     *
     * 1. 单线程：
     *    dim3 grid_dim(1);
     *    dim3 block_dim(1);
     *
     * 2. 256 个 block，每个 block 256 个线程：
     *    dim3 grid_dim(256);
     *    dim3 block_dim(256);
     *
     * 3. 根据数据大小自动计算 grid：
     *    dim3 block_dim(256);
     *    dim3 grid_dim(
     *        (SIZE + block_dim.x - 1) / block_dim.x
     *    );
     */

    dim3 block_dim(1);
    dim3 grid_dim(1);

    // 创建 CUDA Event 进行计时
    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // vector_add<float>(
    //     d_c,
    //     d_a,
    //     d_b,
    //     SIZE,
    //     grid_dim,
    //     block_dim
    // );

    vector_add_inner_loop<float>(
        d_c,
        d_a,
        d_b,
        SIZE,
        grid_dim,
        block_dim
    );


    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms,
        start,
        stop
    ));

    // CUDA_CHECK(cudaMemcpy(
    //     h_c.data(),
    //     d_c,
    //     BYTES,
    //     cudaMemcpyDeviceToHost
    // ));

    // 验证计算结果
    // bool correct = true;

    // for (std::size_t i = 0; i < SIZE; ++i) {
    //     float expected = h_a[i] + h_b[i];

    //     if (h_c[i] != expected) {
    //         std::cerr
    //             << "结果错误，位置 i = " << i
    //             << ", expected = " << expected
    //             << ", actual = " << h_c[i]
    //             << std::endl;

    //         correct = false;
    //         break;
    //     }
    // }

    std::cout << "Grid size:  " << grid_dim.x << std::endl;
    std::cout << "Block size: " << block_dim.x << std::endl;
    std::cout << "运行时间: " << elapsed_ms << " ms" << std::endl;
    // std::cout << "验证结果: "
    //           << (correct ? "正确" : "错误")
    //           << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}