#include <stdio.h>
#include <stdlib.h>

// Matrix dimensions
#define M 256 // Rows in A, Rows in C
#define K 256 // Rows in A, Rows in B
#define N 256 // Columns in B, Rows in C

// A -> input matrix (m x k)
// B -> input matrix (k x n)
// C -> output matrix (m x n)
// m -> no. of rows in A and C
// n -> no. of columns in B and C
// k -> no. of columns in A and no. of rows in B

// Cuda Kernel for matrix multiplication
__global__ void matrixMultiply(float *A, float *B, float *C, int m, int n, int k) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within the matrix bounds
    if (row < m && col < n) {
        // Each thread computes one element of C
        float sum = 0.0f;
        for (int i=0; i<k; i++) {
            sum += A[row * k + i] * B[i * n + col]; // row major indexing
        }
        C[row * n + col] = sum;
    }
}

// Function to initialize a matrix with random values
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Function to clean up resources
void cleanupResources(float *h_A, float *h_B, float *h_C, 
    float *d_A, float *d_B, float *d_C) {
// Free host memory
if (h_A) free(h_A);
if (h_B) free(h_B);
if (h_C) free(h_C);

// Free device memory
if (d_A) cudaFree(d_A);
if (d_B) cudaFree(d_B);
if (d_C) cudaFree(d_C);
}

int main() {
    // Set return code
    int ret = EXIT_SUCCESS;
    
    // Sizes of the matrices in bytes
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    
    // Check if host allocation was successful
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        cleanupResources(h_A, h_B, h_C, NULL, NULL, NULL);
        return EXIT_FAILURE;
    }
    
    // Initialize host matrices
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    
    if (cudaMalloc(&d_A, bytes_A) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }
    
    if (cudaMalloc(&d_B, bytes_B) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }
    
    if (cudaMalloc(&d_C, bytes_C) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }
    
    // Transfer data from host to device
    if (cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device for A\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }
    
    if (cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy data from host to device for B\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }

    // Define grid and block dimensions
    // We'll use a 2D grid of 2D blocks
    dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    // Launch the kernel
    printf("Launching kernel with grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);

    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Check for kernel launch errors
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(kernelError));
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }

    // Wait for kernel to finish
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "CUDA synchronize failed\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }

    // Record end time and calculate elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // Copy result back to host
    if (cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Failed to copy result from device to host\n");
        cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
        return EXIT_FAILURE;
    }
    
    // Calculate performance metrics
    float numOperations = 2.0f * M * N * K;  // Multiply-adds
    float gigaFlops = (numOperations / milliseconds) / 1e6;
    printf("Performance: %.2f GFLOPS\n", gigaFlops);
    
    // Clean up and exit
    cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ret;
}