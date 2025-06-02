#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>

// 核函数1：计算每个元素的指数并求总和
__global__ void softmaxKernel(float *input, float *output, float *sum, int size) {
    extern __shared__ float shared_mem[]; // 动态共享内存用于块内求和

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Step 1: 计算指数并写入全局内存
    if (idx < size) {
        float val = expf(input[idx]);
        output[idx] = val;
        shared_mem[tid] = val; // 将值存入共享内存用于块内求和
    } else {
        shared_mem[tid] = 0.0f;
    }
    __syncthreads();

    // Step 2: 块内归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    // Step 3: 将块总和存入全局内存
    if (tid == 0) {
        sum[blockIdx.x] = shared_mem[0];
        printf("Block %d 局部总和: %.4f\n", blockIdx.x, shared_mem[0]);
    }
}

// 核函数2：归一化
__global__ void normalizeKernel(float *values, float sum, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        values[idx] /= sum;
        // 打印前5个元素的归一化过程
        if (idx < 5) {
            printf("Thread %d (Block %d): value[%d] = %.4f\n", 
                   threadIdx.x, blockIdx.x, idx, values[idx]);
        }
    }
}

int main() {
    const int N = 8;  // 示例数据量较小以便观察
    float *h_input, *h_output;
    float *d_input, *d_values, *d_block_sums;

    // === 主机内存分配和初始化 ===
    printf("初始化主机数据...\n");
    h_input = new float[N];
    h_output = new float[N];
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;  // 示例输入 [0.0, 1.0, ..., 7.0]
        printf("输入数据[%d] = %.2f\n", i, h_input[i]);
    }

    // === 设备内存分配 ===
    printf("\n分配设备内存...\n");
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_values, N * sizeof(float));
    int blockSize = 4;  // 示例使用较小的块大小以便观察
    int numBlocks = (N + blockSize - 1) / blockSize;
    cudaMalloc(&d_block_sums, numBlocks * sizeof(float));

    // === 数据拷贝到设备 ===
    printf("\n拷贝输入数据到设备...\n");
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // === 第一步：计算指数和局部总和 ===
    printf("\n启动Softmax第一步核函数...\n");
    softmaxKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(d_input, d_values, d_block_sums, N);
    cudaDeviceSynchronize();

    // === 主机端计算总和 ===
    float *h_block_sums = new float[numBlocks];
    cudaMemcpy(h_block_sums, d_block_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    float total_sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        printf("读取块 %d 总和: %.4f\n", i, h_block_sums[i]);
        total_sum += h_block_sums[i];
    }
    printf("\n全局总和: %.4f\n", total_sum);

    // === 第二步：归一化 ===
    printf("\n启动归一化核函数...\n");
    normalizeKernel<<<numBlocks, blockSize>>>(d_values, total_sum, N);
    cudaDeviceSynchronize();

    // === 拷贝结果回主机 ===
    printf("\n拷贝归一化结果到主机...\n");
    cudaMemcpy(h_output, d_values, N * sizeof(float), cudaMemcpyDeviceToHost);

    // === 打印结果 ===
    printf("\n最终Softmax结果：\n");
    float check_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        printf("输出[%d] = %.4f\n", i, h_output[i]);
        check_sum += h_output[i];
    }
    printf("结果总和 = %.4f (应为1.0)\n", check_sum);

    // === 清理 ===
    cudaFree(d_input);
    cudaFree(d_values);
    cudaFree(d_block_sums);
    delete[] h_input;
    delete[] h_output;
    delete[] h_block_sums;

    return 0;
}
