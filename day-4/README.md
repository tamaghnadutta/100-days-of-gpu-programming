# Tiled Matrix Multiplication

## Toy Setup for understanding concept

Matrix A: 8×8
Matrix B: 8×8
Matrix C: 8×8
Tile size: 4×4

Each block will have 4×4=16 threads, and we'll need a 2×2 grid of blocks to cover the entire 8×8 output matrix.


## Algorithm Breakdown

1. **Thread Block Responsibilities**

Each thread block is responsible for computing a 4×4 tile of the output matrix C. The 2×2 grid of blocks would look like:

```
Grid of Thread Blocks             Each Block Computes
┌─────────┬─────────┐             ┌─────────┬─────────┐
│ Block   │ Block   │             │ C Tile  │ C Tile  │
│ (0,0)   │ (1,0)   │             │ (0,0)   │ (0,1)   │
├─────────┼─────────┤             ├─────────┼─────────┤
│ Block   │ Block   │             │ C Tile  │ C Tile  │
│ (0,1)   │ (1,1)   │             │ (1,0)   │ (1,1)   │
└─────────┴─────────┘             └─────────┴─────────┘
```

2. **Breaking Matrix Multiplication into Tiles**

For each output tile C, we need to compute:
```
C tile = (A tile 0 × B tile 0) + (A tile 1 × B tile 1) + ... + (A tile k-1 × B tile k-1)
```

In the 8×8 example with 4×4 tiles, there are 2 tiles in the K dimension:
```
K dimension split into 2 tiles
           /                 \
Matrix A  ┌─────────┬─────────┐
          │ Tile A00│ Tile A01│
          ├─────────┼─────────┤
          │ Tile A10│ Tile A11│
          └─────────┴─────────┘

Matrix B  ┌─────────┬─────────┐
          │ Tile B00│ Tile B10│
          ├─────────┼─────────┤
          │ Tile B01│ Tile B11│
          └─────────┴─────────┘
```

3. ** Computing One Output Tile**

Let's follow block (0,0) which computes the top-left 4×4 tile of matrix C:

**Phase 1: First Tile Pair**

- Load A tile (0,0) and B tile (0,0) into shared memory
- Compute partial products using these tiles
- Each thread accumulates results for its assigned element

**Phase 2: Second Tile Pair**

- Load A tile (0,1) and B tile (1,0) into shared memory
- Compute partial products using these tiles
- Each thread adds these results to its running sum

**Final Step:**

Write the final accumulated sums to the corresponding elements in matrix C.

Detailed Execution Trace

Let's trace the execution for block (0,0) computing the top-left 4×4 tile of C:

Initial Setup
Thread (0,0) in block (0,0) is responsible for computing C[0,0]
Thread (0,1) in block (0,0) is responsible for computing C[0,1]
...and so on.

**Phase 1: First Tile Computation**

1. Load tiles into shared memory:

```
A tile (0,0):         B tile (0,0):
┌───┬───┬───┬───┐     ┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │     │ 1 │ 5 │ 9 │13 │
├───┼───┼───┼───┤     ├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │     │ 2 │ 6 │10 │14 │
├───┼───┼───┼───┤     ├───┼───┼───┼───┤
│ 9 │10 │11 │12 │     │ 3 │ 7 │11 │15 │
├───┼───┼───┼───┤     ├───┼───┼───┼───┤
│13 │14 │15 │16 │     │ 4 │ 8 │12 │16 │
└───┴───┴───┴───┘     └───┴───┴───┴───┘
```

2. Each thread calculates partial dot products:

- Thread (0,0) computes: 1×1 + 2×2 + 3×3 + 4×4 = 30
- Thread (0,1) computes: 1×5 + 2×6 + 3×7 + 4×8 = 70
- ...and so on for all 16 threads in the block


3. Threads now have partial sums:

```
Partial Results C tile (0,0) after first phase:
┌────┬────┬────┬────┐
│ 30 │ 70 │110 │150 │
├────┼────┼────┼────┤
│ 70 │174 │278 │382 │
├────┼────┼────┼────┤
│110 │278 │446 │614 │
├────┼────┼────┼────┤
│150 │382 │614 │846 │
└────┴────┴────┴────┘
```

**Phase 2: Second Tile Computation**

1. Load the next tiles:

```
A tile (0,1):         B tile (1,0):
┌───┬───┬───┬───┐     ┌───┬───┬───┬───┐
│17 │18 │19 │20 │     │17 │21 │25 │29 │
├───┼───┼───┼───┤     ├───┼───┼───┼───┤
│21 │22 │23 │24 │     │18 │22 │26 │30 │
├───┼───┼───┼───┤     ├───┼───┼───┼───┤
│25 │26 │27 │28 │     │19 │23 │27 │31 │
├───┼───┼───┼───┤     ├───┼───┼───┼───┤
│29 │30 │31 │32 │     │20 │24 │28 │32 │
└───┴───┴───┴───┘     └───┴───┴───┴───┘
```

2. Each thread calculates the next partial products and adds to existing sum:

- Thread (0,0) adds: 17×17 + 18×18 + 19×19 + 20×20 = 1370
- Thread (0,1) adds: 17×21 + 18×22 + 19×23 + 20×24 = 1690
- ...and so on


3. Threads now have final sums:

```
Final Results C tile (0,0):
┌────┬────┬─────┬─────┐
│1400│1760│ 2120│ 2480│
├────┼────┼─────┼─────┤
│1760│2220│ 2680│ 3140│
├────┼────┼─────┼─────┤
│2120│2680│ 3240│ 3800│
├────┼────┼─────┼─────┤
│2480│3140│ 3800│ 4460│
└────┴────┴─────┴─────┘
```

Write the final results to global memory (matrix C)


## Key Implementation Details

1. Shared Memory Declaration

```c
__shared__ float As[TILE_SIZE][TILE_SIZE];
__shared__ float Bs[TILE_SIZE][TILE_SIZE];
```

2. Thread Indexing

```c
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

3. Accumulation Setup and Tile Loop

```c
float sum = 0.0f;
for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
```

4. Collaborative Loading of Tiles

```c
if (row < m && t * TILE_SIZE + threadIdx.x < k)
    As[threadIdx.y][threadIdx.x] = A[row * k + t * TILE_SIZE + threadIdx.x];
else
    As[threadIdx.y][threadIdx.x] = 0.0f;
```

5. Thread Synchronization

```c
__syncthreads();
```

6. Tile Computation

```c
for (int i = 0; i < TILE_SIZE; i++) {
    sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
}
```

7. Another Synchronization

```c
__syncthreads();
```

8. Final Result

```c
if (row < m && col < n) {
    C[row * n + col] = sum;
}
```



## Results

This code was run on a **NVIDIA GeForce RTX 4090** machine on Vast.ai

```
(main) root@C.19417821:/workspace$ nvidia-smi
Sat Apr 12 14:18:20 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.142                Driver Version: 550.142        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090        On  |   00000000:49:00.0 Off |                  Off |
|  0%   31C    P8             12W /  450W |      12MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

```
(main) root@C.19417821:/workspace$ nvcc -o matrix_multiplication_with_tiling matrix_multiplication_with_tiling.cu
(main) root@C.19417821:/workspace$ ./matrix_multiplication_with_tiling
Matrix sizes: A[1024 x 1024], B[1024 x 1024], C[1024 x 1024]
Initializing matrices...
Allocating device memory...
Copying data from host to device...
Launching kernel with grid dimensions: (64, 64)
Block dimensions: (16, 16)

Running naive kernel...
Naive kernel execution time: 0.758 ms

Verifying naive kernel result:
C[582,100]: Expected = 246.989014, Actual = 246.989014, Relative Error = 0.000000e+00
C[49,809]: Expected = 258.738403, Actual = 258.738373, Relative Error = 1.179476e-07
C[356,390]: Expected = 270.946930, Actual = 270.946930, Relative Error = 0.000000e+00
C[1005,412]: Expected = 266.758698, Actual = 266.758667, Relative Error = 1.144014e-07
C[727,287]: Expected = 272.579590, Actual = 272.579590, Relative Error = 0.000000e+00

Running tiled kernel...
Tiled kernel execution time: 0.393 ms

Performance comparison:
Naive implementation: 2833.15 GFLOPS
Tiled implementation: 5462.67 GFLOPS
Speedup from tiling: 1.93x

Verifying tiled kernel result:
C[582,100]: Expected = 246.989014, Actual = 246.989014, Relative Error = 0.000000e+00
C[49,809]: Expected = 258.738403, Actual = 258.738373, Relative Error = 1.179476e-07
C[356,390]: Expected = 270.946930, Actual = 270.946930, Relative Error = 0.000000e+00
C[1005,412]: Expected = 266.758698, Actual = 266.758667, Relative Error = 1.144014e-07
C[727,287]: Expected = 272.579590, Actual = 272.579590, Relative Error = 0.000000e+00

Matrix multiplication completed successfully!
```