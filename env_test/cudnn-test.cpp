#include <cudnn.h>
#include <iostream>

int main() {
    cudnnHandle_t handle;
    cudnnStatus_t status = cudnnCreate(&handle);
    if (status == CUDNN_STATUS_SUCCESS) {
        std::cout << "[Success] cuDNN initialized." << std::endl;
        cudnnDestroy(handle);
    } else {
        std::cerr << "[Error] cuDNN failed: Code " << status << std::endl;
    }
    return 0;
}
