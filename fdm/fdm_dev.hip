// Device side of the fdm preconditioner (see box_fdm.f90): the gather and
// scatter between Neko's element layout and the free unique-node grid, the
// packing of the slab <-> pencil transpose, the spectral scaling, and the
// transforms as hipBLAS/cuBLAS (batched) GEMMs. The MPI exchanges between
// the steps are done by the Fortran side on the buffers exposed here.
// hipBLAS is loaded with dlopen because makeneko's link line is fixed.
// Vendor-neutral: compiles as .cu with the macros below (no C++ runtime:
// makeneko links with the Fortran compiler).
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define devStream_t hipStream_t
#define devMalloc hipMalloc
#define devFree hipFree
#define devMemcpy hipMemcpy
#define devMemcpyAsync hipMemcpyAsync
#define devMemcpyHostToDevice hipMemcpyHostToDevice
#define devMemcpyDeviceToHost hipMemcpyDeviceToHost
#define devMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define devStreamSynchronize hipStreamSynchronize
#define devGetErrorString hipGetErrorString
#define devError_t hipError_t
#define devSuccess hipSuccess
#define devGetLastError hipGetLastError
#define devEvent_t hipEvent_t
#define devEventCreateWithFlags hipEventCreateWithFlags
#define devEventDisableTiming hipEventDisableTiming
#define devEventRecord hipEventRecord
#define devEventSynchronize hipEventSynchronize
#define devEventDestroy hipEventDestroy
#define BLAS_LIB "libhipblas.so"
#define BLAS_LIB_ALT {"libhipblas.so", "libhipblas.so.3", "libhipblas.so.2", "libhipblas.so.1"}
#define BLAS_OP_N HIPBLAS_OP_N
#define BLAS_OP_T HIPBLAS_OP_T
typedef hipblasHandle_t blasHandle_t;
typedef hipblasStatus_t blasStatus_t;
typedef hipblasOperation_t blasOperation_t;
#define BLAS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define BLAS_PREFIX "hipblas"
#define BLAS_V2 ""
#define KLAUNCH(kernel, nblocks, stream, ...) \
  hipLaunchKernelGGL(kernel, dim3(nblocks), dim3(256), 0, stream, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define devStream_t cudaStream_t
#define devMalloc cudaMalloc
#define devFree cudaFree
#define devMemcpy cudaMemcpy
#define devMemcpyAsync cudaMemcpyAsync
#define devMemcpyHostToDevice cudaMemcpyHostToDevice
#define devMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define devMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define devStreamSynchronize cudaStreamSynchronize
#define devGetErrorString cudaGetErrorString
#define devError_t cudaError_t
#define devSuccess cudaSuccess
#define devGetLastError cudaGetLastError
#define devEvent_t cudaEvent_t
#define devEventCreateWithFlags cudaEventCreateWithFlags
#define devEventDisableTiming cudaEventDisableTiming
#define devEventRecord cudaEventRecord
#define devEventSynchronize cudaEventSynchronize
#define devEventDestroy cudaEventDestroy
#define BLAS_LIB "libcublas.so"
#define BLAS_LIB_ALT {"libcublas.so", "libcublas.so.13", "libcublas.so.12", "libcublas.so.11"}
#define BLAS_OP_N CUBLAS_OP_N
#define BLAS_OP_T CUBLAS_OP_T
typedef cublasHandle_t blasHandle_t;
typedef cublasStatus_t blasStatus_t;
typedef cublasOperation_t blasOperation_t;
#define BLAS_SUCCESS CUBLAS_STATUS_SUCCESS
#define BLAS_PREFIX "cublas"
#define BLAS_V2 "_v2"
#define KLAUNCH(kernel, nblocks, stream, ...) kernel<<<nblocks, 256, 0, stream>>>(__VA_ARGS__)
#endif

#define DEV_CHECK(call)                                                        \
  do {                                                                         \
    devError_t e_ = (call);                                                    \
    if (e_ != devSuccess) {                                                    \
      fprintf(stderr, "fdm_dev: %s at %s:%d (last device error: %s)\n", devGetErrorString(e_), __FILE__, \
              __LINE__, devGetErrorString(devGetLastError()));                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define BLAS_CHECK(call)                                                       \
  do {                                                                         \
    blasStatus_t s_ = (call);                                                  \
    if (s_ != BLAS_SUCCESS) {                                                  \
      fprintf(stderr, "fdm_dev: " BLAS_PREFIX " error %d at %s:%d (last device error: %s)\n", (int)s_, \
              __FILE__, __LINE__, devGetErrorString(devGetLastError()));        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef blasStatus_t (*fn_create_t)(blasHandle_t*);
typedef blasStatus_t (*fn_destroy_t)(blasHandle_t);
typedef blasStatus_t (*fn_setstream_t)(blasHandle_t, devStream_t);
typedef blasStatus_t (*fn_dgemm_t)(blasHandle_t, blasOperation_t, blasOperation_t, int, int, int,
                                   const double*, const double*, int, const double*, int,
                                   const double*, double*, int);
typedef blasStatus_t (*fn_dgemmsb_t)(blasHandle_t, blasOperation_t, blasOperation_t, int, int, int,
                                     const double*, const double*, int, long long, const double*,
                                     int, long long, const double*, double*, int, long long, int);

// Buffer ids for fdm_dev_ptr / fdm_dev_copy (the Fortran side uses the same);
// 5 and 6: the packed send and receive buffers of the 2D scheme
enum { BUF_SBUF = 1, BUF_RBUF = 2, BUF_T = 3, BUF_PB = 4, BUF_S2 = 5, BUF_S3 = 6 };

// The work buffers are one pool shared by every preconditioner instance
// (the solves never overlap; the three instances of a case differ by a few
// per cent in size): slot sizes grow to the largest request. On one rank
// rbuf aliases sbuf and pb aliases t (no exchange, no copy).
enum { POOL_SBUF = 0, POOL_RBUF, POOL_A, POOL_T, POOL_PB, POOL_PC, POOL_S0, POOL_S1, POOL_S2, POOL_S3, POOL_W, POOL_N };
static double* g_pool[POOL_N];
static size_t g_pool_n[POOL_N];
// the exchange buffers in single precision ("exchange_precision": "single"):
// sbuf, rbuf, t, pb as floats (the transforms stay in double)
enum { FPOOL_SBUF = 0, FPOOL_RBUF, FPOOL_T, FPOOL_PB, FPOOL_S2, FPOOL_S3, FPOOL_N };
static float* g_fpool[FPOOL_N];
static size_t g_fpool_n[FPOOL_N];

struct fdm_ctx {
  devStream_t stream;
  blasHandle_t handle;
  void* lib;
  fn_destroy_t destroy;
  fn_dgemm_t dgemm;
  fn_dgemmsb_t dgemmsb;
  int nfx, nfy, nfz, nzl, ncl, n, one_rank, single;
  size_t ncol, nheld, nrecv, nslab, npencil;
  double *Sx, *Sy, *Sz, *lx, *ly, *lz;
  int *gsend, *rpos, *sidx, *cdest, *cstride;
  int cs_me;  // first global column of this rank's pencils
  // the 2D scheme: layouts A (nfx x nyA x nzA), B (nxB x nfy x nzA), C (nxB x nyC x nfz),
  // the 1-based permutations of the two transposes (see box_fdm.f90)
  int twod, nyA, nzA, nxB, nyC, x0B, y0C;
  size_t nA, nB, nC;
  int *posA, *tabB, *posB, *tabC;
  // overlap of the slab scheme's transposes: one event per chunk of planes
  int nch;
  devEvent_t ev[16];
};

namespace {

template <class T>
T* upload(const T* h, size_t count) {
  T* d = nullptr;
  if (count == 0) return d;
  DEV_CHECK(devMalloc((void**)&d, count * sizeof(T)));
  DEV_CHECK(devMemcpy(d, h, count * sizeof(T), devMemcpyHostToDevice));
  return d;
}

void pool_reserve(int slot, size_t count) {
  if (count <= g_pool_n[slot] && g_pool[slot]) return;
  if (g_pool[slot]) (void)devFree(g_pool[slot]);
  DEV_CHECK(devMalloc((void**)&g_pool[slot], (count > 0 ? count : 1) * sizeof(double)));
  g_pool_n[slot] = count > 0 ? count : 1;
}

void fpool_reserve(int slot, size_t count) {
  if (count <= g_fpool_n[slot] && g_fpool[slot]) return;
  if (g_fpool[slot]) (void)devFree(g_fpool[slot]);
  DEV_CHECK(devMalloc((void**)&g_fpool[slot], (count > 0 ? count : 1) * sizeof(float)));
  g_fpool_n[slot] = count > 0 ? count : 1;
}

// the buffers of a solve; on one rank rbuf is sbuf and pb is t
double* buf(const fdm_ctx* c, int slot) {
  if (c->one_rank && slot == POOL_RBUF) slot = POOL_SBUF;
  if (c->one_rank && slot == POOL_PB) slot = POOL_T;
  return g_pool[slot];
}

float* fbuf(const fdm_ctx* c, int slot) {
  if (c->one_rank && slot == FPOOL_RBUF) slot = FPOOL_SBUF;
  if (c->one_rank && slot == FPOOL_PB) slot = FPOOL_T;
  return g_fpool[slot];
}

// counts the entries of h that differ from ref (relative 1e-12): a check
// that the Helmholtz coefficient is uniform
__global__ void nonuniform_k(const double* __restrict__ h, size_t n, double ref, int* count) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    if (fabs(h[i] - ref) > 1e-12 * (fabs(ref) + 1e-300)) atomicAdd(count, 1);
}

// The exchange buffers are double or float (T); the grid arrays are double.
// w[i] = r[idx[i] - 1]
template <class T>
__global__ void gather_k(T* __restrict__ w, const double* __restrict__ r, const int* __restrict__ idx, size_t m) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x)
    w[i] = (T)r[idx[i] - 1];
}

// w[idx[i] - 1] = r[i]  (duplicates write the same value)
template <class T>
__global__ void scatter_k(double* __restrict__ w, const T* __restrict__ r, const int* __restrict__ idx, size_t m) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x)
    w[idx[i] - 1] = (double)r[i];
}

// z[p] = w[sidx[p] - 1], or 0 where sidx[p] == 0 (Dirichlet nodes)
template <class T>
__global__ void scatter_masked_k(double* __restrict__ z, const T* __restrict__ w, const int* __restrict__ sidx,
                                 size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const int u = sidx[i];
    z[i] = u > 0 ? (double)w[u - 1] : 0.0;
  }
}

// slab a(c, izl) (c fastest, ncol x nzl) <-> packed t: t[cdest[c] + izl cstride[c]]
template <class T>
__global__ void pack_k(T* __restrict__ t, const double* __restrict__ a, const int* __restrict__ cdest,
                       const int* __restrict__ cstride, size_t ncol, size_t nslab) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nslab; i += (size_t)gridDim.x * blockDim.x) {
    const size_t c = i % ncol, izl = i / ncol;
    t[(size_t)cdest[c] + izl * (size_t)cstride[c]] = (T)a[i];
  }
}

template <class T>
__global__ void unpack_k(double* __restrict__ a, const T* __restrict__ t, const int* __restrict__ cdest,
                         const int* __restrict__ cstride, size_t ncol, size_t nslab) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nslab; i += (size_t)gridDim.x * blockDim.x) {
    const size_t c = i % ncol, izl = i / ncol;
    a[i] = (double)t[(size_t)cdest[c] + izl * (size_t)cstride[c]];
  }
}

// w[idx[i] - 1] = r[i] into a typed buffer (the pack of the 2D scheme)
template <class T>
__global__ void scatter_to_k(T* __restrict__ w, const double* __restrict__ r, const int* __restrict__ idx,
                             size_t m) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x)
    w[idx[i] - 1] = (T)r[i];
}

// w[i] = r[idx[i] - 1] from a typed buffer (the unpack back of the 2D scheme)
template <class T>
__global__ void gather_from_k(double* __restrict__ w, const T* __restrict__ r, const int* __restrict__ idx,
                              size_t m) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x)
    w[i] = (double)r[idx[i] - 1];
}

// pack / unpack of the slab indices [i0, i1) only (the chunks of the overlap)
template <class T>
__global__ void pack_range_k(T* __restrict__ t, const double* __restrict__ a, const int* __restrict__ cdest,
                             const int* __restrict__ cstride, size_t ncol, size_t i0, size_t i1) {
  for (size_t i = i0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < i1; i += (size_t)gridDim.x * blockDim.x) {
    const size_t c = i % ncol, izl = i / ncol;
    t[(size_t)cdest[c] + izl * (size_t)cstride[c]] = (T)a[i];
  }
}

template <class T>
__global__ void unpack_range_k(double* __restrict__ a, const T* __restrict__ t, const int* __restrict__ cdest,
                               const int* __restrict__ cstride, size_t ncol, size_t i0, size_t i1) {
  for (size_t i = i0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < i1; i += (size_t)gridDim.x * blockDim.x) {
    const size_t c = i % ncol, izl = i / ncol;
    a[i] = (double)t[(size_t)cdest[c] + izl * (size_t)cstride[c]];
  }
}

__global__ void f2d_k(double* __restrict__ d, const float* __restrict__ f, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    d[i] = (double)f[i];
}

__global__ void d2f_k(float* __restrict__ f, const double* __restrict__ d, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    f[i] = (float)d[i];
}

// layout C of the 2D scheme (xl fastest, nxB x nyC x nfz), global x0B + xl, y0C + yl
__global__ void scale2_k(double* __restrict__ w, const double* __restrict__ lx, const double* __restrict__ ly,
                         const double* __restrict__ lz, int nxB, int nyC, int x0B, int y0C, size_t total,
                         double h1, double h2, long long null_idx) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int xl = (int)(i % nxB);
    const size_t t = i / nxB;
    const int yl = (int)(t % nyC);
    const int iz = (int)(t / nyC);
    const double den = h1 * (lx[x0B + xl] + ly[y0C + yl] + lz[iz]) + h2;
    w[i] = ((long long)i == null_idx) ? 0.0 : w[i] / den;
  }
}

// pencil b(cl, iz) (cl fastest, ncl x nfz), global column cs_me + cl:
// b /= h1 (lx + ly + lz) + h2; the null mode (index null_idx or -1) is set
// to zero: the pseudo-inverse of a singular operator for a compatible
// right-hand side
__global__ void scale_k(double* __restrict__ w, const double* __restrict__ lx, const double* __restrict__ ly,
                        const double* __restrict__ lz, int nfx, int ncl, int cs_me, size_t total, double h1,
                        double h2, long long null_idx) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int cl = (int)(i % ncl);
    const int iz = (int)(i / ncl);
    const int c = cs_me + cl;
    const int ix = c % nfx, iy = c / nfx;
    const double den = h1 * (lx[ix] + ly[iy] + lz[iz]) + h2;
    w[i] = ((long long)i == null_idx) ? 0.0 : w[i] / den;
  }
}

unsigned blocks_for(size_t n) {
  size_t b = (n + 255) / 256;
  return (unsigned)(b > 65535 ? 65535 : (b == 0 ? 1 : b));
}

void* must_sym(void* lib, const char* name) {
  void* p = dlsym(lib, name);
  if (!p) {
    fprintf(stderr, "fdm_dev: %s not found in " BLAS_LIB "\n", name);
    exit(1);
  }
  return p;
}

}  // namespace

// the 2D steps for the exchange type T (double, or float with single-precision exchanges)
template <class T>
void dev2_step(fdm_ctx* c, int which, double h1, double h2, long long null_idx, T* s2, T* s3, T* rbuf) {
  const double one = 1.0, zero = 0.0;
  double *s0 = buf(c, POOL_S0), *s1 = buf(c, POOL_S1);
  const int nfx = c->nfx, nfy = c->nfy, nfz = c->nfz, nyA = c->nyA, nzA = c->nzA, nxB = c->nxB, nyC = c->nyC;
  switch (which) {
    case 1:
      if (c->nA == 0) return;
      if (c->nrecv > 0) KLAUNCH(scatter_k<T>, blocks_for(c->nrecv), c->stream, s0, rbuf, c->rpos, c->nrecv);
      BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_T, BLAS_OP_N, nfx, nyA * nzA, nfx, &one, c->Sx, nfx, s0, nfx, &zero,
                          s1, nfx));
      KLAUNCH(scatter_to_k<T>, blocks_for(c->nA), c->stream, s2, s1, c->posA, c->nA);
      break;
    case 2:
      if (c->nB == 0) return;
      KLAUNCH(scatter_k<T>, blocks_for(c->nB), c->stream, s0, s3, c->tabB, c->nB);
      BLAS_CHECK(c->dgemmsb(c->handle, BLAS_OP_N, BLAS_OP_N, nxB, nfy, nfy, &one, s0, nxB, (long long)nxB * nfy,
                            c->Sy, nfy, 0, &zero, s1, nxB, (long long)nxB * nfy, nzA));
      KLAUNCH(scatter_to_k<T>, blocks_for(c->nB), c->stream, s2, s1, c->posB, c->nB);
      break;
    case 3:
      if (c->nC == 0) return;
      KLAUNCH(scatter_k<T>, blocks_for(c->nC), c->stream, s0, s3, c->tabC, c->nC);
      BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_N, nxB * nyC, nfz, nfz, &one, s0, nxB * nyC, c->Sz, nfz,
                          &zero, s1, nxB * nyC));
      KLAUNCH(scale2_k, blocks_for(c->nC), c->stream, s1, c->lx, c->ly, c->lz, nxB, nyC, c->x0B, c->y0C, c->nC, h1,
              h2, null_idx);
      BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_T, nxB * nyC, nfz, nfz, &one, s1, nxB * nyC, c->Sz, nfz,
                          &zero, s0, nxB * nyC));
      KLAUNCH(gather_k<T>, blocks_for(c->nC), c->stream, s2, s0, c->tabC, c->nC);
      break;
    case 4:
      if (c->nB == 0) return;
      KLAUNCH(gather_from_k<T>, blocks_for(c->nB), c->stream, s0, s3, c->posB, c->nB);
      BLAS_CHECK(c->dgemmsb(c->handle, BLAS_OP_N, BLAS_OP_T, nxB, nfy, nfy, &one, s0, nxB, (long long)nxB * nfy,
                            c->Sy, nfy, 0, &zero, s1, nxB, (long long)nxB * nfy, nzA));
      KLAUNCH(gather_k<T>, blocks_for(c->nB), c->stream, s2, s1, c->tabB, c->nB);
      break;
    default:
      if (c->nA == 0) return;
      KLAUNCH(gather_from_k<T>, blocks_for(c->nA), c->stream, s0, s3, c->posA, c->nA);
      BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_N, nfx, nyA * nzA, nfx, &one, c->Sx, nfx, s0, nfx, &zero,
                          s1, nfx));
      if (c->nrecv > 0) KLAUNCH(gather_k<T>, blocks_for(c->nrecv), c->stream, rbuf, s1, c->rpos, c->nrecv);
      break;
  }
  DEV_CHECK(devGetLastError());
}

// The steps of the 2D scheme between the exchanges (S2 is sent, S3 received):
// 1 A: the slab from rbuf, x transform, pack for A -> B;
// 2 B: unpack, y transform, pack for B -> C;
// 3 C: unpack, z transform, scaling, z back, pack for C -> B;
// 4 B: unpack, y back, pack for B -> A;
// 5 A: unpack, x back, the received values into rbuf.

extern "C" {

// Sizes: the free grid nfx x nfy x nfz, this rank's slab of nzl planes and
// pencil of ncl columns (first global column cs_me), n element-layout
// points, nheld held nodes (sent to the slab owners), nrecv received slab
// values. Arrays (1-based indices): gsend[nheld] element-layout index of a
// held node, rpos[nrecv] slab position of a received value, sidx[n] held
// node of a point or 0, cdest[ncol] / cstride[ncol] packed position of a
// slab column (pos = cdest + izl cstride, 0-based), S* and lam* the
// eigenbases (by columns) and eigenvalues.
void* fdm_dev_setup(void* stream, int nfx, int nfy, int nfz, int nzl, int ncl, int cs_me, int n,
                    long long nheld, long long nrecv, int one_rank, int single, const double* Sx, const double* Sy,
                    const double* Sz, const double* lamx, const double* lamy, const double* lamz,
                    const int* gsend, const int* rpos, const int* sidx, const int* cdest,
                    const int* cstride) {
  // a device error left by earlier work (Neko's own kernels or MPI) would be
  // blamed on the first call below that checks errors: report it as such
  {
    devError_t e = devGetLastError();
    if (e != devSuccess) {
      fprintf(stderr, "fdm_dev: a device error is pending before the fdm setup, not caused by it: %s\n",
              devGetErrorString(e));
      exit(1);
    }
  }
  fdm_ctx* c = (fdm_ctx*)calloc(1, sizeof(fdm_ctx));
  c->stream = (devStream_t)stream;
  c->one_rank = one_rank;
  c->single = single;
  c->nfx = nfx;
  c->nfy = nfy;
  c->nfz = nfz;
  c->nzl = nzl;
  c->ncl = ncl;
  c->cs_me = cs_me;
  c->n = n;
  c->ncol = (size_t)nfx * nfy;
  c->nheld = (size_t)nheld;
  c->nrecv = (size_t)nrecv;
  c->nslab = c->ncol * (size_t)nzl;
  c->npencil = (size_t)ncl * (size_t)nfz;
  // the BLAS library by its unversioned name, else by the common sonames
  // (a cluster's runtime often ships only libcublas.so.12 or so); it is
  // searched on LD_LIBRARY_PATH as usual
  const char* names[] = BLAS_LIB_ALT;
  for (size_t k = 0; k < sizeof(names) / sizeof(names[0]) && !c->lib; ++k)
    c->lib = dlopen(names[k], RTLD_NOW | RTLD_GLOBAL);
  if (!c->lib) {
    fprintf(stderr, "fdm_dev: cannot dlopen " BLAS_LIB " (nor its sonames): %s\n", dlerror());
    exit(1);
  }
  fn_create_t create = (fn_create_t)must_sym(c->lib, BLAS_PREFIX "Create" BLAS_V2);
  fn_setstream_t setstream = (fn_setstream_t)must_sym(c->lib, BLAS_PREFIX "SetStream" BLAS_V2);
  c->destroy = (fn_destroy_t)must_sym(c->lib, BLAS_PREFIX "Destroy" BLAS_V2);
  c->dgemm = (fn_dgemm_t)must_sym(c->lib, BLAS_PREFIX "Dgemm" BLAS_V2);
  c->dgemmsb = (fn_dgemmsb_t)must_sym(c->lib, BLAS_PREFIX "DgemmStridedBatched");
  BLAS_CHECK(create(&c->handle));
  BLAS_CHECK(setstream(c->handle, c->stream));
  c->Sx = upload(Sx, (size_t)nfx * nfx);
  c->Sy = upload(Sy, (size_t)nfy * nfy);
  c->Sz = upload(Sz, (size_t)nfz * nfz);
  c->lx = upload(lamx, (size_t)nfx);
  c->ly = upload(lamy, (size_t)nfy);
  c->lz = upload(lamz, (size_t)nfz);
  c->gsend = upload(gsend, c->nheld);
  c->rpos = upload(rpos, c->nrecv);
  c->sidx = upload(sidx, (size_t)n);
  // the slab scheme's column tables (the 2D scheme passes nzl = 0)
  c->cdest = upload(cdest, nzl > 0 ? c->ncol : 1);
  c->cstride = upload(cstride, nzl > 0 ? c->ncol : 1);
  pool_reserve(POOL_A, c->nslab);
  pool_reserve(POOL_T, one_rank ? (c->nslab > c->npencil ? c->nslab : c->npencil) : c->nslab);
  pool_reserve(POOL_PC, c->npencil);
  if (single) {
    // the exchanges in floats; the double pencil buffer is the GEMM workspace
    pool_reserve(POOL_PB, c->npencil);
    fpool_reserve(FPOOL_SBUF, one_rank ? (c->nheld > c->nrecv ? c->nheld : c->nrecv) : c->nheld);
    if (!one_rank) fpool_reserve(FPOOL_RBUF, c->nrecv);
    fpool_reserve(FPOOL_T, one_rank ? (c->nslab > c->npencil ? c->nslab : c->npencil) : c->nslab);
    if (!one_rank) fpool_reserve(FPOOL_PB, c->npencil);
  } else {
    pool_reserve(POOL_SBUF, one_rank ? (c->nheld > c->nrecv ? c->nheld : c->nrecv) : c->nheld);
    if (!one_rank) pool_reserve(POOL_RBUF, c->nrecv);
    if (!one_rank) pool_reserve(POOL_PB, c->npencil);
  }
  return c;
}

void fdm_dev_free(void* ctx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (!c) return;
  double* ds[] = {c->Sx, c->Sy, c->Sz, c->lx, c->ly, c->lz};
  for (double* p : ds) (void)devFree(p);
  int* is[] = {c->gsend, c->rpos, c->sidx, c->cdest, c->cstride, c->posA, c->tabB, c->posB, c->tabC};
  for (int* p : is) (void)devFree(p);
  for (int k = 0; k < c->nch; ++k) (void)devEventDestroy(c->ev[k]);
  if (c->handle) c->destroy(c->handle);
  if (c->lib) dlclose(c->lib);
  free(c);
}

// Overlap: nch chunks of planes with one event each (at most 16).
void fdm_dev_chunks(void* ctx, int nch) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  c->nch = nch > 16 ? 16 : nch;
  for (int k = 0; k < c->nch; ++k) DEV_CHECK(devEventCreateWithFlags(&c->ev[k], devEventDisableTiming));
  pool_reserve(POOL_W, c->nslab);   // the GEMM scratch of the chunked path (t holds packed data)
}

void fdm_dev_event_record(void* ctx, int k) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  DEV_CHECK(devEventRecord(c->ev[k], c->stream));
}

void fdm_dev_event_sync(void* ctx, int k) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  DEV_CHECK(devEventSynchronize(c->ev[k]));
}

// The slab from rbuf (all planes), before the chunked transforms.
void fdm_dev_slab_scatter(void* ctx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  double* a = buf(c, POOL_A);
  if (c->nslab == 0 || c->nrecv == 0) return;
  if (c->single)
    KLAUNCH(scatter_k<float>, blocks_for(c->nrecv), c->stream, a, fbuf(c, FPOOL_RBUF), c->rpos, c->nrecv);
  else
    KLAUNCH(scatter_k<double>, blocks_for(c->nrecv), c->stream, a, buf(c, POOL_RBUF), c->rpos, c->nrecv);
  DEV_CHECK(devGetLastError());
}

// The x and y transforms and the pack of the planes [z0, z1) (the forward
// transpose of these planes can start when their event has fired).
void fdm_dev_slab_in_planes(void* ctx, int z0, int z1) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  const int nfx = c->nfx, nfy = c->nfy, nz = z1 - z0;
  const double one = 1.0, zero = 0.0;
  const size_t off = (size_t)nfx * nfy * z0, cnt = (size_t)nfx * nfy * nz;
  double *a = buf(c, POOL_A) + off, *w = buf(c, POOL_W) + off;
  if (nz <= 0) return;
  BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_T, BLAS_OP_N, nfx, nfy * nz, nfx, &one, c->Sx, nfx, a, nfx, &zero, w,
                      nfx));
  BLAS_CHECK(c->dgemmsb(c->handle, BLAS_OP_N, BLAS_OP_N, nfx, nfy, nfy, &one, w, nfx, (long long)nfx * nfy, c->Sy,
                        nfy, 0, &zero, a, nfx, (long long)nfx * nfy, nz));
  // the pack of these planes only: pack_k indexes the slab from its start,
  // so it runs on the full slab arrays with the plane range as the bounds
  if (c->single)
    KLAUNCH(pack_range_k<float>, blocks_for(cnt), c->stream, fbuf(c, FPOOL_T), buf(c, POOL_A), c->cdest,
            c->cstride, c->ncol, off, off + cnt);
  else
    KLAUNCH(pack_range_k<double>, blocks_for(cnt), c->stream, buf(c, POOL_T), buf(c, POOL_A), c->cdest, c->cstride,
            c->ncol, off, off + cnt);
  DEV_CHECK(devGetLastError());
}

// The unpack and the y and x transforms back of the planes [z0, z1)
// (after their transpose back has arrived).
void fdm_dev_slab_out_planes(void* ctx, int z0, int z1) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  const int nfx = c->nfx, nfy = c->nfy, nz = z1 - z0;
  const double one = 1.0, zero = 0.0;
  const size_t off = (size_t)nfx * nfy * z0, cnt = (size_t)nfx * nfy * nz;
  double *a = buf(c, POOL_A) + off, *w = buf(c, POOL_W) + off;
  if (nz <= 0) return;
  if (c->single)
    KLAUNCH(unpack_range_k<float>, blocks_for(cnt), c->stream, buf(c, POOL_A), fbuf(c, FPOOL_T), c->cdest,
            c->cstride, c->ncol, off, off + cnt);
  else
    KLAUNCH(unpack_range_k<double>, blocks_for(cnt), c->stream, buf(c, POOL_A), buf(c, POOL_T), c->cdest,
            c->cstride, c->ncol, off, off + cnt);
  BLAS_CHECK(c->dgemmsb(c->handle, BLAS_OP_N, BLAS_OP_T, nfx, nfy, nfy, &one, a, nfx, (long long)nfx * nfy, c->Sy,
                        nfy, 0, &zero, w, nfx, (long long)nfx * nfy, nz));
  BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_N, nfx, nfy * nz, nfx, &one, c->Sx, nfx, w, nfx, &zero, a,
                      nfx));
  DEV_CHECK(devGetLastError());
}

// The received values of the slab into rbuf (after all planes are back).
void fdm_dev_slab_gather(void* ctx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  double* a = buf(c, POOL_A);
  if (c->nslab == 0 || c->nrecv == 0) return;
  if (c->single)
    KLAUNCH(gather_k<float>, blocks_for(c->nrecv), c->stream, fbuf(c, FPOOL_RBUF), a, c->rpos, c->nrecv);
  else
    KLAUNCH(gather_k<double>, blocks_for(c->nrecv), c->stream, buf(c, POOL_RBUF), a, c->rpos, c->nrecv);
  DEV_CHECK(devGetLastError());
}

// The buffers the Fortran side exchanges: 1 sbuf (nheld), 2 rbuf (nrecv),
// 3 t (packed slab, nslab), 4 pb (pencil, npencil).
void* fdm_dev_ptr(void* ctx, int which) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (c->single) {
    switch (which) {
      case BUF_SBUF: return fbuf(c, FPOOL_SBUF);
      case BUF_RBUF: return fbuf(c, FPOOL_RBUF);
      case BUF_T: return fbuf(c, FPOOL_T);
      case BUF_S2: return fbuf(c, FPOOL_S2);
      case BUF_S3: return fbuf(c, FPOOL_S3);
      default: return fbuf(c, FPOOL_PB);
    }
  }
  switch (which) {
    case BUF_SBUF: return buf(c, POOL_SBUF);
    case BUF_RBUF: return buf(c, POOL_RBUF);
    case BUF_T: return buf(c, POOL_T);
    case BUF_S2: return buf(c, POOL_S2);
    case BUF_S3: return buf(c, POOL_S3);
    default: return buf(c, POOL_PB);
  }
}

// One rank: the exchange is a copy (on the stream, no synchronization).
void fdm_dev_copy(void* ctx, int from, int to, long long count) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  const size_t es = c->single ? sizeof(float) : sizeof(double);
  if (count > 0)
    DEV_CHECK(devMemcpyAsync(fdm_dev_ptr(c, to), fdm_dev_ptr(c, from), (size_t)count * es,
                             devMemcpyDeviceToDevice, c->stream));
}

void fdm_dev_sync(void* ctx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  DEV_CHECK(devStreamSynchronize(c->stream));
}

// Host staging of an exchange buffer (no device MPI): after the stream's work.
void fdm_dev_d2h(void* ctx, int which, double* host, long long count) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (count > 0)
    DEV_CHECK(devMemcpyAsync(host, fdm_dev_ptr(c, which), (size_t)count * sizeof(double),
                             devMemcpyDeviceToHost, c->stream));
  DEV_CHECK(devStreamSynchronize(c->stream));
}

void fdm_dev_h2d(void* ctx, int which, const double* host, long long count) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (count > 0)
    DEV_CHECK(devMemcpyAsync(fdm_dev_ptr(c, which), host, (size_t)count * sizeof(double),
                             devMemcpyHostToDevice, c->stream));
  DEV_CHECK(devStreamSynchronize(c->stream));  // the host buffer may be reused
}

// Step 1: sbuf = the held nodes of r (element layout).
void fdm_dev_gather(void* ctx, const double* r) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (c->nheld == 0) return;
  if (c->single)
    KLAUNCH(gather_k<float>, blocks_for(c->nheld), c->stream, fbuf(c, FPOOL_SBUF), r, c->gsend, c->nheld);
  else
    KLAUNCH(gather_k<double>, blocks_for(c->nheld), c->stream, buf(c, POOL_SBUF), r, c->gsend, c->nheld);
}

// Step 2 (after the node exchange sbuf -> rbuf): the slab from rbuf, the x
// and y transforms, the pack for the transpose into t.
void fdm_dev_slab_in(void* ctx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  const int nfx = c->nfx, nfy = c->nfy, nzl = c->nzl;
  const double one = 1.0, zero = 0.0;
  double *a = buf(c, POOL_A), *t = buf(c, POOL_T);
  if (c->nslab == 0) return;
  if (c->nrecv > 0) {
    if (c->single)
      KLAUNCH(scatter_k<float>, blocks_for(c->nrecv), c->stream, a, fbuf(c, FPOOL_RBUF), c->rpos, c->nrecv);
    else
      KLAUNCH(scatter_k<double>, blocks_for(c->nrecv), c->stream, a, buf(c, POOL_RBUF), c->rpos, c->nrecv);
  }
  BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_T, BLAS_OP_N, nfx, nfy * nzl, nfx, &one, c->Sx, nfx, a, nfx, &zero, t,
                      nfx));
  BLAS_CHECK(c->dgemmsb(c->handle, BLAS_OP_N, BLAS_OP_N, nfx, nfy, nfy, &one, t, nfx, (long long)nfx * nfy, c->Sy,
                        nfy, 0, &zero, a, nfx, (long long)nfx * nfy, nzl));
  if (c->single)
    KLAUNCH(pack_k<float>, blocks_for(c->nslab), c->stream, fbuf(c, FPOOL_T), a, c->cdest, c->cstride, c->ncol,
            c->nslab);
  else
    KLAUNCH(pack_k<double>, blocks_for(c->nslab), c->stream, t, a, c->cdest, c->cstride, c->ncol, c->nslab);
  DEV_CHECK(devGetLastError());
}

// Step 3 (after the transpose t -> pb): the z transform, the scaling, back.
// null_idx: the local pencil index of the null mode, or -1.
void fdm_dev_pencil(void* ctx, double h1, double h2, long long null_idx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  const int ncl = c->ncl, nfz = c->nfz;
  const double one = 1.0, zero = 0.0;
  double *pb = buf(c, POOL_PB), *pc = buf(c, POOL_PC);
  if (c->npencil == 0) return;
  if (c->single) KLAUNCH(f2d_k, blocks_for(c->npencil), c->stream, pb, fbuf(c, FPOOL_PB), c->npencil);
  BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_N, ncl, nfz, nfz, &one, pb, ncl, c->Sz, nfz, &zero, pc, ncl));
  KLAUNCH(scale_k, blocks_for(c->npencil), c->stream, pc, c->lx, c->ly, c->lz, c->nfx, ncl, c->cs_me, c->npencil, h1,
          h2, null_idx);
  BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_T, ncl, nfz, nfz, &one, pc, ncl, c->Sz, nfz, &zero, pb, ncl));
  if (c->single) KLAUNCH(d2f_k, blocks_for(c->npencil), c->stream, fbuf(c, FPOOL_PB), pb, c->npencil);
  DEV_CHECK(devGetLastError());
}

// Step 4 (after the transpose back pb -> t): unpack, the y and x transforms
// back, the received values gathered into rbuf for the node exchange back.
void fdm_dev_slab_out(void* ctx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  const int nfx = c->nfx, nfy = c->nfy, nzl = c->nzl;
  const double one = 1.0, zero = 0.0;
  double *a = buf(c, POOL_A), *t = buf(c, POOL_T);
  if (c->nslab == 0) return;
  if (c->single)
    KLAUNCH(unpack_k<float>, blocks_for(c->nslab), c->stream, a, fbuf(c, FPOOL_T), c->cdest, c->cstride, c->ncol,
            c->nslab);
  else
    KLAUNCH(unpack_k<double>, blocks_for(c->nslab), c->stream, a, t, c->cdest, c->cstride, c->ncol, c->nslab);
  BLAS_CHECK(c->dgemmsb(c->handle, BLAS_OP_N, BLAS_OP_T, nfx, nfy, nfy, &one, a, nfx, (long long)nfx * nfy, c->Sy,
                        nfy, 0, &zero, t, nfx, (long long)nfx * nfy, nzl));
  BLAS_CHECK(c->dgemm(c->handle, BLAS_OP_N, BLAS_OP_N, nfx, nfy * nzl, nfx, &one, c->Sx, nfx, t, nfx, &zero, a,
                      nfx));
  if (c->nrecv > 0) {
    if (c->single)
      KLAUNCH(gather_k<float>, blocks_for(c->nrecv), c->stream, fbuf(c, FPOOL_RBUF), a, c->rpos, c->nrecv);
    else
      KLAUNCH(gather_k<double>, blocks_for(c->nrecv), c->stream, buf(c, POOL_RBUF), a, c->rpos, c->nrecv);
  }
  DEV_CHECK(devGetLastError());
}

// Step 5 (after the node exchange back rbuf -> sbuf): z in the element
// layout, zero at Dirichlet nodes.
void fdm_dev_scatter(void* ctx, double* z) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (c->single)
    KLAUNCH(scatter_masked_k<float>, blocks_for((size_t)c->n), c->stream, z, fbuf(c, FPOOL_SBUF), c->sidx,
            (size_t)c->n);
  else
    KLAUNCH(scatter_masked_k<double>, blocks_for((size_t)c->n), c->stream, z, buf(c, POOL_SBUF), c->sidx,
            (size_t)c->n);
  DEV_CHECK(devGetLastError());
}

// The number of entries of the device array h (n doubles) that differ from
// ref by more than 1e-12 relative (the uniformity check of h1, h2).
long long fdm_dev_nonuniform(void* stream, const void* h, long long n, double ref) {
  int* d = nullptr;
  int count = 0;
  DEV_CHECK(devMalloc((void**)&d, sizeof(int)));
  DEV_CHECK(devMemcpyAsync(d, &count, sizeof(int), devMemcpyHostToDevice, (devStream_t)stream));
  KLAUNCH(nonuniform_k, blocks_for((size_t)n), (devStream_t)stream, (const double*)h, (size_t)n, ref, d);
  DEV_CHECK(devMemcpyAsync(&count, d, sizeof(int), devMemcpyDeviceToHost, (devStream_t)stream));
  DEV_CHECK(devStreamSynchronize((devStream_t)stream));
  (void)devFree(d);
  return count;
}

// The 2D scheme's tables and work slots.
void fdm_dev2_setup(void* ctx, long long nA, long long nB, long long nC, int nyA, int nzA, int nxB, int nyC,
                    int x0B, int y0C, const int* posA, const int* tabB, const int* posB, const int* tabC) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  c->twod = 1;
  c->nA = (size_t)nA;
  c->nB = (size_t)nB;
  c->nC = (size_t)nC;
  c->nyA = nyA;
  c->nzA = nzA;
  c->nxB = nxB;
  c->nyC = nyC;
  c->x0B = x0B;
  c->y0C = y0C;
  c->posA = upload(posA, c->nA);
  c->tabB = upload(tabB, c->nB);
  c->posB = upload(posB, c->nB);
  c->tabC = upload(tabC, c->nC);
  size_t m = c->nA > c->nB ? c->nA : c->nB;
  if (c->nC > m) m = c->nC;
  for (int k = POOL_S0; k <= (c->single ? POOL_S1 : POOL_S3); ++k) pool_reserve(k, m);
  if (c->single) {
    fpool_reserve(FPOOL_S2, m);
    fpool_reserve(FPOOL_S3, m);
  }
}

void fdm_dev2_step(void* ctx, int which, double h1, double h2, long long null_idx) {
  fdm_ctx* c = (fdm_ctx*)ctx;
  if (c->single)
    dev2_step<float>(c, which, h1, h2, null_idx, fbuf(c, FPOOL_S2), fbuf(c, FPOOL_S3), fbuf(c, FPOOL_RBUF));
  else
    dev2_step<double>(c, which, h1, h2, null_idx, buf(c, POOL_S2), buf(c, POOL_S3), buf(c, POOL_RBUF));
}

// A few values from device memory (the Helmholtz coefficients), after the
// stream's work.
void fdm_dev_download(double* dst, const void* src, size_t bytes, void* stream) {
  DEV_CHECK(devStreamSynchronize((devStream_t)stream));
  DEV_CHECK(devMemcpy(dst, src, bytes, devMemcpyDeviceToHost));
}

}  // extern "C"
