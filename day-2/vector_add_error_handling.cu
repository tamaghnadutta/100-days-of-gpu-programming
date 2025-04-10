#include <stdio.h>
#include <stdlib.h>

__global__ void addVectors(int *a, int *b, int *c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}

// Function to clean up resources
void cleanupResources(int *h_a, int *h_b, int *h_c, int *d_a, int *d_b, int *d_c) {
    // Free host memory
    if (h_a) free(h_a);
    if (h_b) free(h_b);
    if (h_c) free(h_c);
    
    // Free device memory
    if (d_a) cudaFree(d_a);
    if (d_b) cudaFree(d_b);
    if (d_c) cudaFree(d_c);
    
    // Reset device
    cudaDeviceReset();
}

int main() {
    // Define return code
    int ret = EXIT_SUCCESS;
    
    // Size of our vectors
    int size = 1000000;
    size_t bytes = size * sizeof(int);
    
    // Define thread and block dimensions early
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Host vectors
    int *h_a = NULL, *h_b = NULL, *h_c = NULL;
    
    // Device vectors
    int *d_a = NULL, *d_b = NULL, *d_c = NULL;
    
    // Allocate memory for host vectors
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    
    // Check if host allocation was successful
    if (h_a == NULL || h_b == NULL || h_c == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    // Initialize host vectors
    for (int i = 0; i < size; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Allocate memory for device vectors
    if (cudaMalloc(&d_a, bytes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_a\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    if (cudaMalloc(&d_b, bytes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_b\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    if (cudaMalloc(&d_c, bytes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for d_c\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    // Copy input data from host to device
    if (cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device for d_a\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    if (cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device for d_b\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    // Launch kernel on the GPU
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
    
    // Check for errors in kernel launch
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kernelError));
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    // Wait for GPU to finish
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    // Copy result back to host
    if (cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from device to host\n");
        ret = EXIT_FAILURE;
        cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
        return ret;
    }
    
    // Verify results
    for (int i = 0; i < 10; i++) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Cleanup and return
    cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
    return ret;
}