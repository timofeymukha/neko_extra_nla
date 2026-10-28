/* HIP build of the rs_line / rs_colagg kernels (AMD GPUs): hipcc compiles
   the CUDA source through the macro layer at its top, e.g.
     hipcc -O3 -fPIC --offload-arch=gfx942 -c rs_line_hip.cpp
   and the driver links -lamdhip64 -lrocblas -lrocsolver.
   neko_extra_nla: cudaMalloc and cudaFree are added here; the macro layer of
   rs_line_cuda.cu lacks them since its rocBLAS/rocSOLVER path was added (the
   same fix as for the MI300A runs of box_solvers, 2026-10-14). */
#include <hip/hip_runtime.h>
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#include "rs_line_cuda.cu"
