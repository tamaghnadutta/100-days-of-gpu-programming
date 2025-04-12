#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Matrix dimensions - using smaller matrices to ensure it runs in LeetGPU's time limits
#define M 1024   // Rows in A, Rows in C
#define K 1024   // Columns in A, Rows in B
#define N 1024   // Columns in B, Columns in C
#define TILE_SIZE 16  // Size of the tile (16x16 = 256 threads per block)

// Forward declaration of cleanup function
void cleanupResources(float *h_A, float *h_B, float *h_C, 
                      float *d_A, float *d_B, float *d_C);

// Error handling macro - much cleaner way to check CUDA calls
#define CUDA_CHECK(call, message) \
    do { \
        cudaError_t status = call; \
        if (status != cudaSuccess) { \
            fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(status)); \
            cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C); \
            return EXIT_FAILURE; \
        } \
    } while(0)

// Host error checking macro
#define HOST_CHECK(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "%s\n", message); \
            cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C); \
            return EXIT_FAILURE; \
        } \
    } while(0)

// Initialize a matrix with random values
void initializeMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

// Verify the result against a CPU implementation for selected elements
void verifyResult(float *A, float *B, float *C, int m, int n, int k) {
    // Set seed to get reproducible verification points
    srand(42);
    
    // We'll check a few random elements to avoid lengthy CPU computation
    for (int check = 0; check < 5; check++) {
        // Choose random indices
        int i = rand() % m;
        int j = rand() % n;
        
        // Calculate expected value for C[i,j] on CPU
        float expected = 0.0f;
        for (int x = 0; x < k; x++) {
            expected += A[i * k + x] * B[x * n + j];
        }
        
        // Compare with actual value
        float actual = C[i * n + j];
        float diff = fabs(expected - actual);
        float relError = diff / (fabs(expected) > 1e-6 ? fabs(expected) : 1e-6);
        
        printf("C[%d,%d]: Expected = %f, Actual = %f, Relative Error = %e\n", 
               i, j, expected, actual, relError);
        
        // Check if the error is acceptable (allowing for floating-point imprecision)
        if (relError > 1e-5) {
            printf("  WARNING: Error exceeds threshold!\n");
        }
    }
}

// Cleanup resources - centralized function for memory management
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

// Naive matrix multiplication kernel
__global__ void matrixMultiplyNaive(float *A, float *B, float *C, int m, int n, int k) {
    // Calculate global row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within bounds
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled matrix multiplication kernel
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int m, int n, int k) {
    // Shared memory for tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate global row and column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Accumulate result
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        
        // Load element from matrix A
        if (row < m && t * TILE_SIZE + threadIdx.x < k) {
            As[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load element from matrix B
        if (col < n && t * TILE_SIZE + threadIdx.y < k) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Ensure all threads have loaded their data
        __syncthreads();
        
        // Compute partial dot product for this tile
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        // Ensure all computations are done before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    // Host matrices
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    
    // Device matrices
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    
    // Matrix sizes in bytes
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    printf("Matrix sizes: A[%d x %d], B[%d x %d], C[%d x %d]\n", M, K, K, N, M, N);
    
    // Allocate host memory
    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B);
    h_C = (float*)malloc(bytes_C);
    
    // Check if host allocation was successful using our macro
    HOST_CHECK(h_A != NULL && h_B != NULL && h_C != NULL, 
               "Failed to allocate host memory");
    
    // Initialize host matrices
    printf("Initializing matrices...\n");
    initializeMatrix(h_A, M, K);
    initializeMatrix(h_B, K, N);
    
    // Allocate device memory - using our macro for cleaner error handling
    printf("Allocating device memory...\n");
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A), "Failed to allocate device memory for A");
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B), "Failed to allocate device memory for B");
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C), "Failed to allocate device memory for C");
    
    // Transfer data from host to device - much cleaner with macros!
    printf("Copying data from host to device...\n");
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice),
              "Failed to copy data from host to device for A");
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice),
              "Failed to copy data from host to device for B");
    
    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);
    
    printf("Launching kernel with grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
    printf("Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start), "Failed to create start event");
    CUDA_CHECK(cudaEventCreate(&stop), "Failed to create stop event");
    
    // First, run the naive kernel for comparison
    printf("\nRunning naive kernel...\n");
    CUDA_CHECK(cudaEventRecord(start), "Failed to record start event");
    
    matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "Naive kernel launch failed");
    CUDA_CHECK(cudaDeviceSynchronize(), "Naive kernel execution failed");
    
    CUDA_CHECK(cudaEventRecord(stop), "Failed to record stop event");
    CUDA_CHECK(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    // Calculate and print execution time
    float naive_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, start, stop), 
              "Failed to get elapsed time");
    printf("Naive kernel execution time: %.3f ms\n", naive_ms);
    
    // Copy naive result back to host for verification
    float *h_C_naive = (float*)malloc(bytes_C);
    HOST_CHECK(h_C_naive != NULL, "Failed to allocate host memory for naive result");
    
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes_C, cudaMemcpyDeviceToHost),
              "Failed to copy naive result from device to host");
    
    // Verify naive result
    printf("\nVerifying naive kernel result:\n");
    verifyResult(h_A, h_B, h_C_naive, M, N, K);
    free(h_C_naive); // Free naive result
    
    // Now run the tiled kernel
    printf("\nRunning tiled kernel...\n");
    CUDA_CHECK(cudaEventRecord(start), "Failed to record start event");
    
    matrixMultiplyTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError(), "Tiled kernel launch failed");
    CUDA_CHECK(cudaDeviceSynchronize(), "Tiled kernel execution failed");
    
    CUDA_CHECK(cudaEventRecord(stop), "Failed to record stop event");
    CUDA_CHECK(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    // Calculate and print execution time
    float tiled_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, start, stop), 
              "Failed to get elapsed time");
    printf("Tiled kernel execution time: %.3f ms\n", tiled_ms);
    
    // Calculate performance metrics
    float numOperations = 2.0f * M * N * K;  // Multiply-adds
    float naive_gflops = (numOperations / naive_ms) / 1e6;
    float tiled_gflops = (numOperations / tiled_ms) / 1e6;
    
    
    printf("\nPerformance comparison:\n");
    printf("Naive implementation: %.2f GFLOPS\n", naive_gflops);
    printf("Tiled implementation: %.2f GFLOPS\n", tiled_gflops);
    printf("Speedup from tiling: %.2fx\n", naive_ms / tiled_ms);
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost),
              "Failed to copy result from device to host");
    
    // Verify result
    printf("\nVerifying tiled kernel result:\n");
    verifyResult(h_A, h_B, h_C, M, N, K);
    
    // Clean up
    cleanupResources(h_A, h_B, h_C, d_A, d_B, d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("\nMatrix multiplication completed successfully!\n");
    
    return EXIT_SUCCESS;
}