## The CUDA Thread Hierarchy
```
┌─────────────────────────── ENTIRE GRID ───────────────────────────┐
│                                                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ Block   │ │ Block   │ │ Block   │ │ Block   │ │ Block   │      │
│  │ (0,0)   │ │ (1,0)   │ │ (2,0)   │ │ (3,0)   │ │ (4,0)   │ ...  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
│                                                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ Block   │ │ Block   │ │ Block   │ │ Block   │ │ Block   │      │
│  │ (0,1)   │ │ (1,1)   │ │ (2,1)   │ │ (3,1)   │ │ (4,1)   │ ...  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
│                                                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ Block   │ │ Block   │ │ Block   │ │ Block   │ │ Block   │      │
│  │ (0,2)   │ │ (1,2)   │ │ (2,2)   │ │ (3,2)   │ │ (4,2)   │ ...  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘      │
│                                                                   │
│       .          .          .          .          .               │
│       .          .          .          .          .               │
│       .          .          .          .          .               │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

## Inside a Single Block (shown here block with 10x10 threads)
```
┌───────────────── BLOCK (blockIdx.x, blockIdx.y) ───────────────────┐
│                                                                    │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ... ┌───┐         │
│  │0,0│ │1,0│ │2,0│ │3,0│ │4,0│ │5,0│ │6,0│ │7,0│     │9,0│         │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘         │
│                                                                    │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐         │
│  │0,1│ │1,1│ │2,1│ │3,1│ │4,1│ │5,1│ │6,1│ │7,1│     │9,1│         │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘         │
│                                                                    │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐         │
│  │0,2│ │1,2│ │2,2│ │3,2│ │4,2│ │5,2│ │6,2│ │7,2│     │9,2│         │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘         │
│                                                                    │
│   .     .     .     .     .     .     .     .         .            │
│   .     .     .     .     .     .     .     .         .            │
│   .     .     .     .     .     .     .     .         .            │
│                                                                    │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     ┌───┐         │
│  │0,9│ │1,9│ │2,9│ │3,9│ │4,9│ │5,9│ │6,9│ │7,9│     │9,9│         │
│  └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     └───┘         │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
      ^ Each small box represents a thread with (threadIdx.x, threadIdx.y)
```

## Thread Index Calculation
```
For a grid with 16×16 threads per block:

Thread (3,5) in Block (2,1) computes element C[?,?] where:

Row index = blockIdx.y * blockDim.y + threadIdx.y
          = 1 * 16 + 5
          = 21

Column index = blockIdx.x * blockDim.x + threadIdx.x
             = 2 * 16 + 3
             = 35

So this thread computes element C[21,35]
```

## How One Thread Computes One Matrix Element
```
Matrix A                   Matrix B                Matrix C
┌───┬───┬───┬───┬───┐     ┌───┬───┬───┬───┬───┐     ┌───┬───┬───┬───┬───┐
│   │   │   │   │   │     │   │   │   │↑  │   │     │   │   │   │   │   │
├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤
│   │   │   │   │   │     │   │   │   │↑  │   │     │   │   │   │   │   │
├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤
│→→→│→→→│→→→│→→→│→→→│     │   │   │   │↑  │   │     │   │   │   │ ● │   │
├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤
│   │   │   │   │   │     │   │   │   │↑  │   │     │   │   │   │   │   │
├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤     ├───┼───┼───┼───┼───┤
│   │   │   │   │   │     │   │   │   │↑  │   │     │   │   │   │   │   │
└───┴───┴───┴───┴───┘     └───┴───┴───┴───┴───┘     └───┴───┴───┴───┴───┘

Thread calculates C[2,3] = A[2,0]*B[0,3] + A[2,1]*B[1,3] + A[2,2]*B[2,3] + A[2,3]*B[3,3] + A[2,4]*B[4,3]

Each thread reads one entire row of A and one entire column of B to compute a single element of C.
```

## Thread Block Dimensions and Grid Dimensions

For our matrix multiplication where:

Matrix A is M × K (M rows, K columns)
Matrix B is K × N (K rows, N columns)
Matrix C is M × N (M rows, N columns)

```
dim3 blockDim(16, 16);  // 16×16 = 256 threads per block

dim3 gridDim((N + blockDim.x - 1) / blockDim.x, 
             (M + blockDim.y - 1) / blockDim.y);
```

If M = N = 1024

- Each block handles 16×16 = 256 elements of the output matrix
- We need (1024 + 16 - 1) / 16 = 64 blocks in each dimension
- Total grid size is 64×64 = 4,096 blocks
- Total threads = 4,096 blocks × 256 threads/block = 1,048,576 threads


## Results on LeetGPU

With 256 x 256 matrix sizes

```
Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
Compiling...
Executing...
Launching kernel with grid dimensions: (16, 16)
Block dimensions: (16, 16)
Kernel execution time: 196.789459 ms
Performance: 0.17 GFLOPS
Exit status: 0
```

With 512x512 matrix sizes

```
Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
Compiling...
Executing...
Launching kernel with grid dimensions: (32, 32)
Block dimensions: (16, 16)
Kernel execution time: 1916.048828 ms
Performance: 0.14 GFLOPS
Exit status: 0
```

With 1024x1024 matrix sizes

```
Running NVIDIA GTX TITAN X in FUNCTIONAL mode...
Compiling...
Executing...
Launching kernel with grid dimensions: (64, 64)
Block dimensions: (16, 16)
Execution timed out after 10 seconds
```