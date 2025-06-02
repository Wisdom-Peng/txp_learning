// kernel.cu
__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    c[tid] = a[tid] + b[tid];
}
int main() {
    // 分配内存、启动核函数等
}

