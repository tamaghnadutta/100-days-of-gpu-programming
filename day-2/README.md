# CUDA Vector Addition with Error Handling

A CUDA program that demonstrates parallel vector addition with error handling. It builds upon the basic vector addition example by adding comprehensive error checks at each stage of the CUDA workflow.

## Introduction

CUDA (Compute Unified Device Architecture) allows us to leverage NVIDIA GPUs for general-purpose computing tasks. While the massive parallelism of GPUs offers tremendous performance benefits, it also introduces complexity in memory management and execution flow that requires careful error handling.

This example shows how to implement proper error detection and resource management in CUDA applications – a critical skill for reliable GPU programming that is often overlooked in introductory examples.

## Why Error Handling Matters in CUDA

GPU programming introduces several potential failure points not typically encountered in CPU-only code:

- Device detection and initialization failures
- Memory allocation failures (GPU memory is limited and separate from system RAM)
- Memory transfer errors between host and device
- Kernel launch failures
- Runtime execution errors within kernels
- Synchronization issues
- Resource cleanup problems

Without proper error detection and handling, CUDA applications may:
- Crash without clear explanation
- Silently produce incorrect results
- Leak GPU resources
- Leave the device in an inconsistent state

## Code Structure and Components

Our vector addition program follows the standard CUDA workflow but with added error handling at each step:

1. **Memory allocation** on both host (CPU) and device (GPU)
2. **Data initialization** on the host
3. **Data transfer** from host to device
4. **Kernel execution** to perform the parallel computation
5. **Results retrieval** from device back to host
6. **Verification** of the results
7. **Resource cleanup** to free memory and reset the device

### Core Components Explained

#### Resource Management Function

```c
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
```

This function centralizes all cleanup operations, making the code more maintainable and ensuring resources are properly released regardless of where an error occurs.

#### Error Checking Pattern

```c
if (cudaMalloc(&d_a, bytes) != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device memory for d_a\n");
    ret = EXIT_FAILURE;
    cleanupResources(h_a, h_b, h_c, d_a, d_b, d_c);
    return ret;
}
```

After each CUDA operation, we check its return value against `cudaSuccess`. If an error occurs, we:
1. Print a descriptive error message
2. Set the return code to indicate failure
3. Clean up all allocated resources
4. Exit the current function

#### Kernel Function

```c
__global__ void addVectors(int *a, int *b, int *c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
```

The kernel is the function that runs on the GPU. Each thread calculates its global index and performs addition for its assigned element. The boundary check (`if (i < size)`) prevents out-of-bounds access when the number of threads exceeds the array size.

#### Kernel Launch and Error Detection

```c
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
```

Kernel launches don't return error codes directly, so we must call `cudaGetLastError()` to check if the launch succeeded. We also call `cudaDeviceSynchronize()` afterward to ensure the kernel has completed execution before checking for runtime errors.

## Error Handling Principles Demonstrated

1. **Early initialization and NULL checks**
   All pointers are initialized to NULL and checked before use/freeing

2. **Immediate error detection and handling**
   Each CUDA operation is checked immediately after execution

3. **Descriptive error messages**
   Each error case provides specific information about what went wrong

4. **Consistent resource cleanup**
   All resources are freed regardless of where an error occurs

5. **Return code pattern**
   The program uses a return code to indicate success or failure

6. **No uninitialized variables**
   All variables are properly initialized before use

## Compiling and Running

To compile the program:

```bash
nvcc vector_add_error_handling.cu -o vector_add_error_handling
```

To run the program:

```bash
./vector_add_error_handling
```

Expected output (on successful execution):
```
0 + 0 = 0
1 + 2 = 3
2 + 4 = 6
3 + 6 = 9
4 + 8 = 12
5 + 10 = 15
6 + 12 = 18
7 + 14 = 21
8 + 16 = 24
9 + 18 = 27
```
