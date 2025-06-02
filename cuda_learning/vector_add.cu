#include <stdio.h>
#include <cuda_runtime.h>

// CUDA 核函数：打印线程信息和计算结果
__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
        // 打印前 5 个元素的计算过程（避免过多输出）
        if (i < 5) {
            printf("GPU Thread %d: a[%d]=%d + b[%d]=%d → c[%d]=%d\n",
                   i, i, a[i], i, b[i], i, c[i]);
        }
    }
}

void vectorAddCPU(int *a, int *b, int *c, int n) {
    printf("\nCPU 计算开始...\n");
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
        if (i < 5) {
            printf("CPU 计算 %d: a[%d]=%d + b[%d]=%d → c[%d]=%d\n",
                   i, i, a[i], i, b[i], i, c[i]);
        }
    }
}

int main() {
    const int N = 1024;
    int *a, *b, *c_cpu, *c_gpu;
    int *d_a, *d_b, *d_c;

    // === 初始化阶段 ===
    printf("初始化主机内存...\n");
    a = new int[N];
    b = new int[N];
    c_cpu = new int[N];
    c_gpu = new int[N];

    printf("填充初始数据(a[i]=i, b[i]=i*2)...\n");
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
        if (i < 5) {
            printf("Init: a[%d]=%d, b[%d]=%d\n", i, a[i], i, b[i]);
        }
    }

    // === GPU 内存操作 ===
    printf("\n分配设备内存...\n");
    cudaError_t err;
    err = cudaMalloc(&d_a, N * sizeof(int));
    printf("cudaMalloc(d_a): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_b, N * sizeof(int));
    printf("cudaMalloc(d_b): %s\n", cudaGetErrorString(err));
    err = cudaMalloc(&d_c, N * sizeof(int));
    printf("cudaMalloc(d_c): %s\n", cudaGetErrorString(err));

    printf("\n拷贝数据到设备...\n");
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    printf("cudaMemcpy(d_a): %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    printf("cudaMemcpy(d_b): %s\n", cudaGetErrorString(cudaGetLastError()));

    // === 核函数执行 ===
    printf("\n启动核函数...\n");
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    printf("配置参数: blockSize=%d, numBlocks=%d, 总线程数=%d\n",
           blockSize, numBlocks, blockSize * numBlocks);
    
    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);
    printf("核函数已启动, 等待 GPU 完成...\n");
    cudaDeviceSynchronize();
    printf("GPU 计算完成: %s\n", cudaGetErrorString(cudaGetLastError()));

    // === 取回结果 ===
    printf("\n拷贝结果回主机...\n");
    cudaMemcpy(c_gpu, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("cudaMemcpy(c_gpu): %s\n", cudaGetErrorString(cudaGetLastError()));

    // === CPU 验证 ===
    vectorAddCPU(a, b, c_cpu, N);

    // === 结果检查 ===
    printf("\n验证结果...\n");
    bool error = false;
    for (int i = 0; i < N; i++) {
        if (c_cpu[i] != c_gpu[i]) {
            printf("错误! 索引 %d: CPU=%d, GPU=%d\n", i, c_cpu[i], c_gpu[i]);
            error = true;
            break;
        } else if (i < 5) {  // 打印前 5 个结果
            printf("验证通过: c[%d] = %d\n", i, c_gpu[i]);
        }
    }
    if (!error) printf("\n所有元素验证通过！\n");

    // === 清理 ===
    printf("\n释放资源...\n");
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a; delete[] b; delete[] c_cpu; delete[] c_gpu;
    printf("释放完成！\n");

    return 0;
}