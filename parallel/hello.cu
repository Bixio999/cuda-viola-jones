#include <stdio.h>

__global__ void helloFromGPU (void) {
    printf("Hello World from GPU!\n");
}

int main(void) {
    // hello from GPU
    printf("Hello World from CPU!\n");
    cudaSetDevice(1);
    helloFromGPU <<<1,10>>>();
    cudaDeviceReset();
    return 0;
}




