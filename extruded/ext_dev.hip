// Device path of the "extruded" solver (HIP): the kernels around the layout
// exchanges (the exchanges themselves are MPI calls in Fortran on these
// buffers: makeneko compiles this file without MPI headers and links no
// extra libraries, so rocBLAS is loaded with dlopen), the z transform (rocBLAS
// dgemm), and the batched supernodal
// sweep with the G blocks in fp32 (one launch per elimination-tree level,
// tiles of 256 x 64, two factor slots per block; see
// simulations/2026-10-14-supernodal-gpu in the project for the measurements).
// One source for both backends: hipcc compiles it as HIP (ext_dev.hip), nvcc as
// CUDA (copy it to ext_dev.cu for makeneko on a CUDA Neko).
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#if defined(__HIP__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define devError_t hipError_t
#define devSuccess hipSuccess
#define devGetErrorString hipGetErrorString
#define devMalloc hipMalloc
#define devMemcpy hipMemcpy
#define devMemcpyHostToDevice hipMemcpyHostToDevice
#define devMemsetAsync hipMemsetAsync
#define devDeviceSynchronize hipDeviceSynchronize
#define ATOMIC_ADD unsafeAtomicAdd
#define SHFL_DOWN8(v, o) __shfl_down((v), (o), 8)
// rocBLAS, loaded with dlopen (makeneko's link line is fixed)
typedef void *blas_handle;
typedef int (*create_t)(blas_handle *);
typedef int (*dgemm_t)(blas_handle, int, int, int, int, int, const double *, const double *, int,
                       const double *, int, const double *, double *, int);
#define BLAS_OP_N 111   /* BLAS_OP_N */
#define BLAS_OP_T 112   /* BLAS_OP_T */
static const char *BLAS_NAMES[] = {"librocblas.so", "librocblas.so.5", "librocblas.so.4", "librocblas.so.0"};
#define BLAS_CREATE "rocblas_create_handle"
#define BLAS_DGEMM "rocblas_dgemm"
#else
#include <cuda_runtime.h>
#define devError_t cudaError_t
#define devSuccess cudaSuccess
#define devGetErrorString cudaGetErrorString
#define devMalloc cudaMalloc
#define devMemcpy cudaMemcpy
#define devMemcpyHostToDevice cudaMemcpyHostToDevice
#define devMemsetAsync cudaMemsetAsync
#define devDeviceSynchronize cudaDeviceSynchronize
#define ATOMIC_ADD atomicAdd
// the 8-lane groups of a warp may run different loop lengths: name only the group's lanes
#define SHFL_DOWN8(v, o) __shfl_down_sync(0xffu << (threadIdx.x & 24), (v), (o), 8)
// cuBLAS, loaded with dlopen
typedef void *blas_handle;
typedef int (*create_t)(blas_handle *);
typedef int (*dgemm_t)(blas_handle, int, int, int, int, int, const double *, const double *, int,
                       const double *, int, const double *, double *, int);
#define BLAS_OP_N 0     /* CUBLAS_OP_N */
#define BLAS_OP_T 1     /* CUBLAS_OP_T */
static const char *BLAS_NAMES[] = {"libcublas.so", "libcublas.so.13", "libcublas.so.12", "libcublas.so.11"};
#define BLAS_CREATE "cublasCreate_v2"
#define BLAS_DGEMM "cublasDgemm_v2"
#endif

#define HC(x) do { devError_t e_ = (x); if (e_ != devSuccess) { \
  fprintf(stderr, "extruded: device error %s at %s:%d\n", devGetErrorString(e_), __FILE__, __LINE__); \
  abort(); } } while (0)

constexpr int TB = 256, TRW = 256, TCW = 64, BTR = 8;

namespace {

template <int R, int SB>
__global__ __launch_bounds__(TB) void k_fwd(const float *__restrict__ G, size_t gstride,
    const long *__restrict__ px, const int *__restrict__ pi, const int *__restrict__ super,
    const int *__restrict__ srow, const int4 *__restrict__ tiles, double *b, double *y, size_t n) {
  constexpr int Q = R * SB;
  __shared__ double bs[Q][TCW];
  __shared__ double red[Q][TB];
  const int4 t = tiles[blockIdx.x];
  const int s = t.x, i0 = t.y, j0 = t.z;
  const int k1 = super[s], nc = super[s + 1] - k1, nsrow = pi[s + 1] - pi[s];
  const int nj = min(TCW, nc - j0), ni = min(TRW, nsrow - i0);
  double *bv = b + (size_t)blockIdx.y * Q * n, *yv = y + (size_t)blockIdx.y * Q * n;
  for (int k = threadIdx.x; k < nj; k += TB)
    for (int q = 0; q < Q; q++) bs[q][k] = bv[q * n + k1 + j0 + k];
  __syncthreads();
  const float *g = G + (size_t)blockIdx.y * SB * gstride + px[s] + (size_t)j0 * nsrow + i0;
  const int TR = ni < TB ? ni : TB, CG = TB / TR, tr = threadIdx.x % TR, cg = threadIdx.x / TR;
  const bool multi = nc > TCW;
  for (int ib = 0; ib < ni; ib += TR) {
    const int i = ib + tr;
    double acc[Q];
    for (int q = 0; q < Q; q++) acc[q] = 0.0;
    if (cg < CG && i < ni)
      for (int j = cg; j < nj; j += CG) {
        double gj[SB];
        for (int sl = 0; sl < SB; sl++) gj[sl] = g[sl * gstride + (size_t)j * nsrow + i];
        for (int sl = 0; sl < SB; sl++)
          for (int q = 0; q < R; q++) acc[sl * R + q] += gj[sl] * bs[sl * R + q][j];
      }
    if (CG > 1) {
      for (int q = 0; q < Q; q++) red[q][threadIdx.x] = acc[q];
      __syncthreads();
      if (cg == 0) for (int c = 1; c < CG; c++) for (int q = 0; q < Q; q++) acc[q] += red[q][c * TR + tr];
    }
    if (cg == 0 && i < ni) {
      const int r = i0 + i;
      if (r < nc) {
        for (int q = 0; q < Q; q++)
          if (multi) ATOMIC_ADD(&yv[q * n + k1 + r], acc[q]); else yv[q * n + k1 + r] = acc[q];
      } else {
        const int row = srow[pi[s] + r];
        for (int q = 0; q < Q; q++) ATOMIC_ADD(&bv[q * n + row], -acc[q]);
      }
    }
  }
}

template <int R, int SB>
__global__ __launch_bounds__(TB) void k_bwd(const float *__restrict__ G, size_t gstride,
    const long *__restrict__ px, const int *__restrict__ pi, const int *__restrict__ super,
    const int *__restrict__ srow, const int4 *__restrict__ tiles, const double *y, double *x, size_t n) {
  constexpr int Q = R * SB;
  __shared__ double vs[Q][TRW];
  const int4 t = tiles[blockIdx.x];
  const int s = t.x, i0 = t.y, j0 = t.z;
  const int k1 = super[s], nc = super[s + 1] - k1, nsrow = pi[s + 1] - pi[s];
  const int nj = min(TCW, nc - j0), ni = min(TRW, nsrow - i0);
  const double *yv = y + (size_t)blockIdx.y * Q * n;
  double *xv = x + (size_t)blockIdx.y * Q * n;
  for (int k = threadIdx.x; k < ni; k += TB) {
    const int r = i0 + k;
    const int row = r < nc ? 0 : srow[pi[s] + r];
    for (int q = 0; q < Q; q++) vs[q][k] = r < nc ? yv[q * n + k1 + r] : -xv[q * n + row];
  }
  __syncthreads();
  const float *g = G + (size_t)blockIdx.y * SB * gstride + px[s] + (size_t)j0 * nsrow + i0;
  const int TR = BTR, NG = TB / TR;   // BTR = 8 (SHFL_DOWN8)
  const int lane = threadIdx.x % TR, grp = threadIdx.x / TR;
  const bool multi = nsrow > TRW;
  for (int j = grp; j < nj; j += NG) {
    const float *gc = g + (size_t)j * nsrow;
    double acc[Q];
    for (int q = 0; q < Q; q++) acc[q] = 0.0;
    for (int i = lane; i < ni; i += TR) {
      double gi[SB];
      for (int sl = 0; sl < SB; sl++) gi[sl] = gc[sl * gstride + i];
      for (int sl = 0; sl < SB; sl++)
        for (int q = 0; q < R; q++) acc[sl * R + q] += gi[sl] * vs[sl * R + q][i];
    }
    for (int q = 0; q < Q; q++)
      for (int o = TR / 2; o > 0; o /= 2) acc[q] += SHFL_DOWN8(acc[q], o);
    if (lane == 0)
      for (int q = 0; q < Q; q++)
        if (multi) ATOMIC_ADD(&xv[q * n + k1 + j0 + j], acc[q]); else xv[q * n + k1 + j0 + j] = acc[q];
  }
}

// element layout -> send buffer: sb[i] = r[gsend[i]]
template <class T>
__global__ void k_gather(const T *__restrict__ r, const int *__restrict__ gsend, double *sb, int m) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) sb[i] = (double)r[gsend[i]];
}
// received keys -> pencil (and back)
__global__ void k_place(const double *__restrict__ rb, const int *__restrict__ rpos, double *X, int m) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) X[rpos[i]] = rb[i];
}
__global__ void k_take(const double *__restrict__ X, const int *__restrict__ rpos, double *rb, int m) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += gridDim.x * blockDim.x) rb[i] = X[rpos[i]];
}
// send buffer -> element layout (every copy), Dirichlet points 0
template <class T>
__global__ void k_scatter(const double *__restrict__ sb, const int *__restrict__ sidx, T *z, int n) {
  for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += gridDim.x * blockDim.x) {
    const int k = sidx[p];
    z[p] = k >= 0 ? (T)sb[k] : (T)0;
  }
}
// mode planes -> solve vectors (permuted, scaled): b[v][i] = d[slot][node] * mp[base[node] + m * ns[node]]
__global__ void k_in(const double *__restrict__ mp, double *b, const int *__restrict__ perm,
                     const double *__restrict__ d, const int *__restrict__ vmode, const long *__restrict__ base,
                     const int *__restrict__ ns, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const int v = blockIdx.y;
  if (i >= n) return;
  const int m = vmode[v];
  double val = 0.0;
  if (m >= 0) {
    const int node = perm[i];
    val = d[(size_t)(v / 2) * n + node] * mp[base[node] + (long)m * ns[node]];
  }
  b[(size_t)v * n + i] = val;
}
__global__ void k_out(const double *__restrict__ x, double *mp, const int *__restrict__ perm,
                      const double *__restrict__ d, const int *__restrict__ vmode, const long *__restrict__ base,
                      const int *__restrict__ ns, size_t n) {
  const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  const int v = blockIdx.y;
  if (i >= n) return;
  const int m = vmode[v];
  if (m < 0) return;
  const int node = perm[i];
  mp[base[node] + (long)m * ns[node]] = d[(size_t)(v / 2) * n + node] * x[(size_t)v * n + i];
}

// host arrays with malloc (makeneko links no libstdc++)
template <class T> T *hal(size_t cnt) {
  T *p = (T *)malloc(std::max<size_t>(cnt, 1) * sizeof(T));
  if (!p) { fprintf(stderr, "extruded: host allocation failed\n"); abort(); }
  return p;
}

template <class T> T *dup(const T *h, size_t cnt) {
  T *d = nullptr;
  HC(devMalloc(&d, std::max<size_t>(cnt, 1) * sizeof(T)));
  if (cnt) HC(devMemcpy(d, h, cnt * sizeof(T), devMemcpyHostToDevice));
  return d;
}

struct Ctx {
  int P = 1, me = 0, n = 0, nheld = 0, nrecv = 0, nr = 0, nz = 0, nfree = 0, nslot = 0, nm = 0, H = 0;
  int *toff = nullptr;
  int *gsend = nullptr, *sidx = nullptr, *rpos = nullptr, *perm = nullptr, *vmode = nullptr, *ns = nullptr;
  long *base = nullptr;
  double *Sp = nullptr, *d = nullptr, *sb = nullptr, *rb = nullptr, *X = nullptr, *Y = nullptr, *mp = nullptr;
  double *b = nullptr, *y = nullptr, *x = nullptr;
  float *G = nullptr;
  size_t xsize = 0;
  long *px = nullptr;
  int *pi = nullptr, *super = nullptr, *srow = nullptr;
  int4 *tiles = nullptr;
  blas_handle rb_h = nullptr;
  dgemm_t dgemm = nullptr;
};

}  // namespace

extern "C" {

// All index arrays 1-based as in Fortran (converted here); f0 has P + 1 entries.
void *ext_dev_setup(int me, int P, int n, int nheld, int nrecv, int nr, int nz, int nfree,
                    int nslot, int nm, const int *gsend, const int *sidx, const int *rpos,
                    const int *hs_cnt, const int *hs_dsp, const int *hr_cnt, const int *hr_dsp,
                    const int *ms_cnt, const int *ms_dsp, const int *mr_cnt, const int *mr_dsp,
                    const int *f0, const double *Sp, const int *vmode, const double *d, const int *perm,
                    int nsuper, const int *sfc, const int *srp, const int *srow, long nsrow, const long *slp) {
  Ctx *c = (Ctx *)calloc(1, sizeof(Ctx));
  c->P = P; c->me = me; c->n = n; c->nheld = nheld; c->nrecv = nrecv; c->nr = nr; c->nz = nz;
  c->nfree = nfree; c->nslot = nslot; c->nm = nm;
  int *t = hal<int>(std::max(n, std::max(nheld, std::max(nrecv, nfree))));
  for (int i = 0; i < nheld; i++) t[i] = gsend[i] - 1;
  c->gsend = dup(t, nheld);
  for (int i = 0; i < n; i++) t[i] = sidx[i] - 1;   // -1: Dirichlet
  c->sidx = dup(t, n);
  for (int i = 0; i < nrecv; i++) t[i] = rpos[i] - 1;
  c->rpos = dup(t, nrecv);
  for (int i = 0; i < nfree; i++) t[i] = perm[i] - 1;
  c->perm = dup(t, nfree);
  free(t);
  c->vmode = dup(vmode, 2 * nslot);
  // mode-plane position of every free node: base = mr_dsp(src) + (node - f0(src)), ns = n_src
  long *hb = hal<long>(nfree);
  int *hn = hal<int>(nfree);
  for (int s = 0; s < P; s++)
    for (int f = f0[s] - 1; f < f0[s + 1] - 1; f++) { hb[f] = mr_dsp[s] + (f - (f0[s] - 1)); hn[f] = f0[s + 1] - f0[s]; }
  c->base = dup(hb, nfree);
  c->ns = dup(hn, nfree);
  free(hb); free(hn);
  c->Sp = dup(Sp, (size_t)nz * nz);
  c->d = dup(d, (size_t)nslot * nfree);
  HC(devMalloc(&c->sb, std::max(nheld, 1) * 8));
  HC(devMalloc(&c->rb, std::max(nrecv, 1) * 8));
  HC(devMalloc(&c->X, std::max<size_t>((size_t)nr * nz, 1) * 8));
  HC(devMalloc(&c->Y, std::max<size_t>((size_t)nr * nz, 1) * 8));
  size_t nmp = 0;
  for (int s = 0; s < P; s++) nmp += mr_cnt[s];
  HC(devMalloc(&c->mp, std::max<size_t>(nmp, 1) * 8));
  const size_t nv = std::max<size_t>((size_t)2 * nslot * nfree, 1);
  HC(devMalloc(&c->b, nv * 8)); HC(devMalloc(&c->y, nv * 8)); HC(devMalloc(&c->x, nv * 8));
  // supernodal structure (0-based), levels and tiles
  int *sup = hal<int>(nsuper + 1), *pi = hal<int>(nsuper + 1), *sr = hal<int>(nsrow);
  long *px = hal<long>(nsuper + 1);
  for (int s = 0; s <= nsuper; s++) { sup[s] = sfc[s] - 1; pi[s] = srp[s] - 1; px[s] = slp[s]; }
  for (long i = 0; i < nsrow; i++) sr[i] = srow[i] - 1;
  c->xsize = px[nsuper];
  int *col2s = hal<int>(nfree), *lev = hal<int>(nsuper);
  for (int s = 0; s < nsuper; s++) { lev[s] = 0; for (int k = sup[s]; k < sup[s + 1]; k++) col2s[k] = s; }
  c->H = 0;
  for (int s = 0; s < nsuper; s++) {
    const int nc = sup[s + 1] - sup[s], nrow = pi[s + 1] - pi[s];
    if (nrow > nc) {
      const int p = col2s[sr[pi[s] + nc]];
      if (p <= s) { fprintf(stderr, "extruded: supernodes not in topological order\n"); abort(); }
      lev[p] = std::max(lev[p], lev[s] + 1);
    }
    c->H = std::max(c->H, lev[s] + 1);
  }
  // tiles grouped by level (counting sort)
  c->toff = hal<int>(c->H + 1);
  long ntile = 0;
  for (int s = 0; s < nsuper; s++) {
    const int nc = sup[s + 1] - sup[s], nrow = pi[s + 1] - pi[s];
    for (int i0 = 0; i0 < nrow; i0 += TRW)
      for (int j0 = 0; j0 < nc; j0 += TCW)
        if (std::min(nrow, i0 + TRW) > j0) ntile++;
  }
  int4 *ti = hal<int4>(ntile);
  int *fill = hal<int>(c->H + 1);
  for (int l = 0; l <= c->H; l++) { c->toff[l] = 0; fill[l] = 0; }
  for (int s = 0; s < nsuper; s++) {
    const int nc = sup[s + 1] - sup[s], nrow = pi[s + 1] - pi[s];
    for (int i0 = 0; i0 < nrow; i0 += TRW)
      for (int j0 = 0; j0 < nc; j0 += TCW)
        if (std::min(nrow, i0 + TRW) > j0) c->toff[lev[s] + 1]++;
  }
  for (int l = 0; l < c->H; l++) c->toff[l + 1] += c->toff[l];
  for (int s = 0; s < nsuper; s++) {
    const int nc = sup[s + 1] - sup[s], nrow = pi[s + 1] - pi[s];
    for (int i0 = 0; i0 < nrow; i0 += TRW)
      for (int j0 = 0; j0 < nc; j0 += TCW)
        if (std::min(nrow, i0 + TRW) > j0) ti[c->toff[lev[s]] + fill[lev[s]]++] = make_int4(s, i0, j0, 0);
  }
  free(fill); free(col2s); free(lev);
  c->px = dup(px, nsuper + 1);
  c->pi = dup(pi, nsuper + 1);
  c->super = dup(sup, nsuper + 1);
  c->srow = dup(sr, nsrow);
  c->tiles = dup(ti, ntile);
  free(px); free(pi); free(sup); free(sr); free(ti);
  HC(devMalloc(&c->G, std::max<size_t>(c->xsize * nslot, 1) * sizeof(float)));
  void *lib = nullptr;
  for (const char *nm : BLAS_NAMES) if ((lib = dlopen(nm, RTLD_NOW | RTLD_GLOBAL))) break;
  if (!lib) { fprintf(stderr, "extruded: cannot dlopen %s: %s\n", BLAS_NAMES[0], dlerror()); abort(); }
  create_t create = (create_t)dlsym(lib, BLAS_CREATE);
  c->dgemm = (dgemm_t)dlsym(lib, BLAS_DGEMM);
  if (!create || !c->dgemm) { fprintf(stderr, "extruded: BLAS symbols missing\n"); abort(); }
  create(&c->rb_h);
  return c;
}

// slot k (0-based): the G blocks in double -> fp32 on the device
void ext_dev_factor(void *ctx, int k, const double *g) {
  Ctx *c = (Ctx *)ctx;
  float *h = hal<float>(c->xsize);
  for (size_t i = 0; i < c->xsize; i++) h[i] = (float)g[i];
  HC(devMemcpy(c->G + (size_t)k * c->xsize, h, c->xsize * 4, devMemcpyHostToDevice));
  free(h);
}

// device buffers for the exchanges: 1 sb, 2 rb, 3 Y, 4 mp
void *ext_dev_ptr(void *ctx, int which) {
  Ctx *c = (Ctx *)ctx;
  switch (which) {
    case 1: return c->sb;
    case 2: return c->rb;
    case 3: return c->Y;
    default: return c->mp;
  }
}

// phase 1: element layout -> sb (rp_bytes 4 or 8: Neko's real)
void ext_dev_gather(void *ctx, void *r, int rp_bytes) {
  Ctx *c = (Ctx *)ctx;
  if (c->nheld) {
    if (rp_bytes == 8) k_gather<double><<<1024, 256>>>((const double *)r, c->gsend, c->sb, c->nheld);
    else k_gather<float><<<1024, 256>>>((const float *)r, c->gsend, c->sb, c->nheld);
  }
  HC(devDeviceSynchronize());
}

// phase 2: rb -> pencil X, Y = X Sp
void ext_dev_forward(void *ctx) {
  Ctx *c = (Ctx *)ctx;
  const size_t npen = (size_t)c->nr * c->nz;
  HC(devMemsetAsync(c->X, 0, std::max<size_t>(npen, 1) * 8));
  if (c->nrecv) k_place<<<1024, 256>>>(c->rb, c->rpos, c->X, c->nrecv);
  const double one = 1.0, zero = 0.0;
  if (c->nr)
    c->dgemm(c->rb_h, BLAS_OP_N, BLAS_OP_N, c->nr, c->nz, c->nz, &one, c->X, c->nr,
             c->Sp, c->nz, &zero, c->Y, c->nr);
  HC(devDeviceSynchronize());
}

// phase 3: the mode solves on mp (in place)
void ext_dev_solve(void *ctx) {
  Ctx *c = (Ctx *)ctx;
  const size_t n = c->nfree;
  if (c->nslot) {
    dim3 g((n + 255) / 256, 2 * c->nslot);
    k_in<<<g, 256>>>(c->mp, c->b, c->perm, c->d, c->vmode, c->base, c->ns, n);
    const size_t nv = (size_t)2 * c->nslot * n;
    HC(devMemsetAsync(c->y, 0, nv * 8));
    HC(devMemsetAsync(c->x, 0, nv * 8));
    const int s2 = c->nslot / 2 * 2;
    for (int l = 0; l < c->H; l++) {
      const int nt = c->toff[l + 1] - c->toff[l];
      if (!nt) continue;
      if (s2) k_fwd<2, 2><<<dim3(nt, s2 / 2), TB>>>(c->G, c->xsize, c->px, c->pi, c->super, c->srow,
                                                   c->tiles + c->toff[l], c->b, c->y, n);
      if (c->nslot > s2) k_fwd<2, 1><<<dim3(nt, 1), TB>>>(c->G + s2 * c->xsize, c->xsize, c->px, c->pi, c->super,
                                                         c->srow, c->tiles + c->toff[l], c->b + 2 * s2 * n,
                                                         c->y + 2 * s2 * n, n);
    }
    for (int l = c->H - 1; l >= 0; l--) {
      const int nt = c->toff[l + 1] - c->toff[l];
      if (!nt) continue;
      if (s2) k_bwd<2, 2><<<dim3(nt, s2 / 2), TB>>>(c->G, c->xsize, c->px, c->pi, c->super, c->srow,
                                                   c->tiles + c->toff[l], c->y, c->x, n);
      if (c->nslot > s2) k_bwd<2, 1><<<dim3(nt, 1), TB>>>(c->G + s2 * c->xsize, c->xsize, c->px, c->pi, c->super,
                                                         c->srow, c->tiles + c->toff[l], c->y + 2 * s2 * n,
                                                         c->x + 2 * s2 * n, n);
    }
    k_out<<<g, 256>>>(c->x, c->mp, c->perm, c->d, c->vmode, c->base, c->ns, n);
  }
  HC(devDeviceSynchronize());
}

// phase 4: X = Y Sp^T, pencil -> rb
void ext_dev_backward(void *ctx) {
  Ctx *c = (Ctx *)ctx;
  const double one = 1.0, zero = 0.0;
  if (c->nr)
    c->dgemm(c->rb_h, BLAS_OP_N, BLAS_OP_T, c->nr, c->nz, c->nz, &one, c->Y, c->nr,
             c->Sp, c->nz, &zero, c->X, c->nr);
  if (c->nrecv) k_take<<<1024, 256>>>(c->X, c->rpos, c->rb, c->nrecv);
  HC(devDeviceSynchronize());
}

// phase 5: sb -> element layout (all copies; Dirichlet points 0)
void ext_dev_scatter(void *ctx, void *z, int rp_bytes) {
  Ctx *c = (Ctx *)ctx;
  if (c->n) {
    if (rp_bytes == 8) k_scatter<double><<<1024, 256>>>(c->sb, c->sidx, (double *)z, c->n);
    else k_scatter<float><<<1024, 256>>>(c->sb, c->sidx, (float *)z, c->n);
  }
  HC(devDeviceSynchronize());
}

}  // extern "C"
