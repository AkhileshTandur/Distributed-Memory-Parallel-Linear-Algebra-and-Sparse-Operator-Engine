# Distributed-Memory Parallel Linear Algebra and Sparse Operator Engine

Runnable C++17 MPI + OpenMP CSR SpMV engine demonstrating row decomposition, nonblocking neighbor exchange, interior/communication overlap and distributed correctness reduction. No model training is involved.

## Scope
The matrix is a generated tridiagonal operator, with 2 on the diagonal and -1 on adjacent diagonals. Each rank owns contiguous rows and matching vector elements. Only immediate neighbor vector entries are needed, so halo exchange is well defined. Uneven partitions and one-row ranks are supported. OpenMP computes interior rows while MPI communications progress; boundary rows follow MPI_Waitall. MPI calls occur only on the main thread (FUNNELED).

This first implementation does not include arbitrary sparsity communication schedules, dense GEMV, hand-written AVX2, or a custom ring AllReduce. MPI_Allreduce is used for correctness; its internal algorithm is chosen by MPI. Single-machine MPI runs do not establish multi-node performance.

## Build (Linux or WSL2)
```bash
sudo apt-get install build-essential cmake libopenmpi-dev openmpi-bin
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
OMP_NUM_THREADS=2 ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=2 mpiexec -n 2 ./build/spmv 10003 100
```
Direct compiler alternative:
```bash
mkdir -p build
mpicxx -O3 -std=c++17 -fopenmp src/main.cpp -o build/spmv
```
## Correctness and scaling
```bash
OMP_NUM_THREADS=1 mpiexec -n 4 ./build/spmv 7 3
python3 scripts/scaling.py --n 1000000 --threads 2 --ranks 1 2 4 > scaling.jsonl
```
Output records include maximum rank time and globally reduced absolute error against an analytical reference. Speedup baseline is one rank with the SAME thread count; efficiency = speedup / rank count. Pin ranks/threads appropriately for your machine and do not oversubscribe when reporting performance. For multi-node use your cluster's launcher/allocation.

## Data and reliability
Synthetic deterministic input x[i]=sin(0.01*i). No downloaded datasets. Runtime rejects N smaller than number of ranks. Extremely large sizes can exhaust memory; indices use 32-bit int. This is an educational implementation, not a fault-tolerant distributed service. See VALIDATION.md for checks run here. No 6.1x speedup or 76% efficiency is claimed.
