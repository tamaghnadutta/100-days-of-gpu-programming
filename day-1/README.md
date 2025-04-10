# CUDA Vector Addition

A simple CUDA program that demonstrates parallel vector addition on a GPU.

## Overview

This program demonstrates how to:
1. Allocate memory on both the CPU (host) and GPU (device)
2. Transfer data between host and device
3. Execute a parallel computation on the GPU
4. Retrieve results back to the CPU

The example adds two vectors (arrays) element by element. While this is a simple operation, it illustrates the fundamental workflow of most CUDA applications.

## Understanding CUDA Basics

### CPU vs GPU Architecture

CPUs (Central Processing Units) are designed for sequential processing with a few powerful cores, while GPUs (Graphics Processing Units) have thousands of simpler cores designed for parallel operations. CUDA allows us to harness this massive parallelism for general-purpose computing.

### CUDA Programming Model

CUDA uses a Single Instruction, Multiple Thread (SIMT) architecture where many threads execute the same code but operate on different data. This is perfect for operations like vector addition where the same operation is performed on each element of the array.

## Code Explanation

### Host and Device Code

The program is divided into two parts:
- **Host code**: Runs on the CPU, handles memory management and launches GPU kernels
- **Device code**: Runs on the GPU, performs the actual parallel computation

### Memory Management

Four key operations are performed:
1. Allocation of memory on the host (CPU) using `malloc()`
2. Allocation of memory on the device (GPU) using `cudaMalloc()`
3. Copying data from host to device using `cudaMemcpy()` with `cudaMemcpyHostToDevice`
4. Copying results from device back to host using `cudaMemcpy()` with `cudaMemcpyDeviceToHost`

```c
// Host allocation
h_a = (int*)malloc(bytes);

// Device allocation
cudaMalloc(&d_a, bytes);

// Host to device transfer
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

// Device to host transfer
cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
```

### The Kernel Function

The kernel is the function that runs on the GPU:

```c
__global__ void addVectors(int *a, int *b, int *c, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
```

The `__global__` keyword indicates this function runs on the device (GPU) but can be called from the host (CPU).

### Thread Organization

CUDA organizes threads in a hierarchical structure:
- **Threads**: Individual execution units
- **Blocks**: Groups of threads that can cooperate
- **Grid**: Collection of blocks

Each thread needs to know which element it should process. This is calculated using:

```c
int i = blockDim.x * blockIdx.x + threadIdx.x;
```

Where:
- `threadIdx.x`: Index of the thread within its block
- `blockIdx.x`: Index of the block within the grid
- `blockDim.x`: Number of threads per block

### Kernel Launch

The kernel is launched with this syntax:

```c
addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
```

The `<<<blocksPerGrid, threadsPerBlock>>>` configuration specifies:
- How many blocks to create
- How many threads per block

We determine the number of blocks with:

```c
int threadsPerBlock = 256;
int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
```

This formula ensures we have enough threads to cover all elements, rounding up the division to handle cases where the vector size isn't a multiple of the block size.

## Compiling and Running

To compile the program:

```bash
nvcc vector_add.cu -o vector_add
```

To run the program:

```bash
./vector_add
```

## Expected Output

The program will print the first 10 addition results:

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

## Requirements

- An NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- A C/C++ compiler compatible with your CUDA version

or

You can run this on LeetGPU