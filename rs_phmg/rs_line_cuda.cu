/* CUDA kernels of the column-line smoother (rs_line): one thread per
   column line gathers the weighted residual along the line, solves the
   banded Cholesky system (lower band storage, bandwidth lx-1) and scatters
   the result to the nodes it owns. Double precision only.
   Layouts (0-based, c = column, ncol columns, lx = band width + 1, maxn =
   longest column):
     cab[(j*lx + k)*ncol + c]   = L(j+k, j) of column c (k = 0 diagonal)
     y[q*ncol + c]              = work entry q of column c
     cnode[coff[c] + q]         = node index (0-based) of entry q
     cwrite[coff[c] + q]        = 1 if this column writes that node        */
/* One source for both GPU backends: nvcc compiles it as CUDA, hipcc (through
   rs_line_hip.cpp, which includes this file) as HIP with the macro layer
   below. rs_line_cuda_backend() tells the Fortran side which one it got
   (1 CUDA, 2 HIP; the stubs return 0). */
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetLastError hipGetLastError
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetErrorString hipGetErrorString
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define RS_LINE_BACKEND 2
#define RS_SHFL_XOR(v, m) __shfl_xor((v), (m), 8)
#define RS_SHFL(v, l) __shfl((v), (l), 8)
#define RS_DECL_MASK (void) 0
#define RS_SYNCGROUP() (void) 0
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#define RS_HAVE_HIPBLAS 1
#else
#include <cuda_runtime.h>
#define RS_LINE_BACKEND 1
/* CUDA: the four 8-lane groups of a warp run different loop lengths, so
   the shuffles name only the lanes of their own group, and the groups
   synchronise where one lane's store is read by the others */
#define RS_DECL_MASK const unsigned rs_mask = 0xFFu << (threadIdx.x & 24)
#define RS_SHFL_XOR(v, m) __shfl_xor_sync(rs_mask, (v), (m), 8)
#define RS_SHFL(v, l) __shfl_sync(rs_mask, (v), (l), 8)
#define RS_SYNCGROUP() __syncwarp(rs_mask)
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif
#include <stdio.h>
#include <stdlib.h>

__global__ void rs_line_solve_kernel(double * __restrict__ z,
                                     const double * __restrict__ r,
                                     const double * __restrict__ cab,
                                     const int * __restrict__ cnode,
                                     const int * __restrict__ cwrite,
                                     const int * __restrict__ coff,
                                     const int * __restrict__ clen,
                                     const double * __restrict__ wsq,
                                     const double * __restrict__ mask,
                                     double * __restrict__ y,
                                     const int ncol, const int lx)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  const int len = clen[c];
  const int off = coff[c];
  const int kd = lx - 1;

  for (int q = 0; q < len; q++) {
    const int idx = cnode[off + q];
    y[(size_t) q * ncol + c] = wsq[idx] * mask[idx] * r[idx];
  }
  /* forward: L v = b */
  for (int j = 0; j < len; j++) {
    double acc = y[(size_t) j * ncol + c];
    int i0 = j - kd; if (i0 < 0) i0 = 0;
    for (int i = i0; i < j; i++)
      acc -= cab[((size_t) i * lx + (j - i)) * ncol + c] * y[(size_t) i * ncol + c];
    y[(size_t) j * ncol + c] = acc / cab[((size_t) j * lx) * ncol + c];
  }
  /* backward: L^T x = v */
  for (int j = len - 1; j >= 0; j--) {
    double acc = y[(size_t) j * ncol + c];
    int i1 = j + kd; if (i1 > len - 1) i1 = len - 1;
    for (int i = j + 1; i <= i1; i++)
      acc -= cab[((size_t) j * lx + (i - j)) * ncol + c] * y[(size_t) i * ncol + c];
    y[(size_t) j * ncol + c] = acc / cab[((size_t) j * lx) * ncol + c];
  }
  for (int q = 0; q < len; q++) {
    if (cwrite[off + q]) {
      const int idx = cnode[off + q];
      z[idx] = wsq[idx] * mask[idx] * y[(size_t) q * ncol + c];
    }
  }
}

/* distributed columns: pack the weighted residual at the nodes of the
   pieces this rank sends (one entry per sbuf slot) and unpack the
   solution pieces at the slots flagged for writing */
__global__ void rs_line_gpack_kernel(const double * __restrict__ r,
                                     const double * __restrict__ wsq,
                                     const double * __restrict__ mask,
                                     const int * __restrict__ idx,
                                     double * __restrict__ sb, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { const int j = idx[i]; sb[i] = wsq[j] * mask[j] * r[j]; }
}

__global__ void rs_line_gunpack_kernel(double * __restrict__ z,
                                       const double * __restrict__ wsq,
                                       const double * __restrict__ mask,
                                       const int * __restrict__ idx,
                                       const int * __restrict__ wr,
                                       const double * __restrict__ sb, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && wr[i]) { const int j = idx[i]; z[j] = wsq[j] * mask[j] * sb[i]; }
}

/* owner side of the distributed columns: received pieces -> packed gy
   (y[q*ncol + c]), in-place band solve of each owned column (factors in
   the cab layout, length glen[c]), and back into the receive buffer */
__global__ void rs_line_gfill_kernel(double * __restrict__ gy, const double * __restrict__ rb,
                                     const int * __restrict__ map, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) gy[map[i]] = rb[i];
}

__global__ void rs_line_gunfill_kernel(double * __restrict__ rb, const double * __restrict__ gy,
                                       const int * __restrict__ map, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) rb[i] = gy[map[i]];
}

__global__ void rs_line_gsolve_kernel(double * __restrict__ y, const double * __restrict__ cab,
                                      const int * __restrict__ glen, const int ncol, const int lx)
{
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ncol) return;
  const int len = glen[c];
  const int kd = lx - 1;
  for (int j = 0; j < len; j++) {
    double acc = y[(size_t) j * ncol + c];
    int i0 = j - kd; if (i0 < 0) i0 = 0;
    for (int i = i0; i < j; i++)
      acc -= cab[((size_t) i * lx + (j - i)) * ncol + c] * y[(size_t) i * ncol + c];
    y[(size_t) j * ncol + c] = acc / cab[((size_t) j * lx) * ncol + c];
  }
  for (int j = len - 1; j >= 0; j--) {
    double acc = y[(size_t) j * ncol + c];
    int i1 = j + kd; if (i1 > len - 1) i1 = len - 1;
    for (int i = j + 1; i <= i1; i++)
      acc -= cab[((size_t) j * lx + (i - j)) * ncol + c] * y[(size_t) i * ncol + c];
    y[(size_t) j * ncol + c] = acc / cab[((size_t) j * lx) * ncol + c];
  }
}

/* Group kernel: eight lanes per column (lx <= 8, so the band width kd <= 7).
   The factors are packed per column without padding, two layouts of
   lx entries per row/column j (cofn[c] + j*lx + t):
     cabf: forward,  t < kd: L(j, j-kd+t) (0 outside), t = kd: L(j,j)
     cabb: backward, t < kd: L(j+1+t, j)  (0 beyond the end)
   Lane t keeps a register window: forward y_{j-kd+t}, backward x_{j+1+t};
   each step is one load per lane, a 3-step shuffle reduction and a
   one-lane shift. The eight columns of a wavefront form a block whose
   rows are interleaved (columns sorted by length, block rows padded to
   the block's longest column): row j of column (cb, ci) is at
   ((cboff[cb] + j) * 8 + ci) * lx + t in cabf / cabb and the work entry
   at (cboff[cb] + j) * 8 + ci, so that a wavefront's step loads 512
   contiguous bytes. */
/* factors F = double or float (line_factor_precision single: half the
   factor traffic, arithmetic still double); the diagonal lane holds the
   inverse of the diagonal (multiplication instead of a division on the
   sequential path) */
template <typename F>
__global__ void rs_line_solve_group_kernel(double * __restrict__ z,
                                           const double * __restrict__ r,
                                           const F * __restrict__ cabf,
                                           const F * __restrict__ cabb,
                                           const int * __restrict__ cnode,
                                           const int * __restrict__ cwrite,
                                           const int * __restrict__ coff,
                                           const int * __restrict__ clen,
                                           const int * __restrict__ cboff,
                                           const double * __restrict__ wm,
                                           double * __restrict__ y,
                                           const int ncol, const int lx)
{
  /* wm = wsq * mask, one strided array in the gather and the scatter */
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = gid >> 3;
  const int t = gid & 7;
  if (c >= ncol) return;
  RS_DECL_MASK;
  const int len = clen[c];
  const int off = coff[c];
  const int kd = lx - 1;
  const int ci = c & 7;
  const size_t yo = (size_t) cboff[c >> 3] * 8 + ci;   /* work entry of row 0 */
  const size_t fo = yo * lx;                            /* factor row 0 */
  /* gather, eight entries at a time */
  for (int q = t; q < len; q += 8) {
    const int idx = cnode[off + q];
    y[yo + (size_t) q * 8] = wm[idx] * r[idx];
  }
  RS_SYNCGROUP();
  /* forward: L v = b; lane t holds v_{j-kd+t} (0 when j-kd+t < 0) */
  double w = 0.0;
  for (int j = 0; j < len; j++) {
    const double f = (t < lx) ? (double) cabf[fo + (size_t) j * 8 * lx + t] : 0.0;
    double prod = (t < kd) ? f * w : 0.0;
    prod += RS_SHFL_XOR(prod, 4);
    prod += RS_SHFL_XOR(prod, 2);
    prod += RS_SHFL_XOR(prod, 1);
    const double dinv = RS_SHFL(f, kd);
    const double vj = (y[yo + (size_t) j * 8] - prod) * dinv;
    /* shift the window: w_t <- w_{t+1}, w_{kd-1} <- v_j */
    const double wn = RS_SHFL(w, (t + 1) & 7);
    w = (t == kd - 1) ? vj : ((t < kd - 1) ? wn : 0.0);
    if (t == 0) y[yo + (size_t) j * 8] = vj;
  }
  RS_SYNCGROUP();
  /* backward: L^T x = v; lane t holds x_{j+1+t} (0 beyond the end) */
  w = 0.0;
  for (int j = len - 1; j >= 0; j--) {
    const double f = (t < lx) ? (double) cabb[fo + (size_t) j * 8 * lx + t] : 0.0;
    double prod = (t < kd) ? f * w : 0.0;
    prod += RS_SHFL_XOR(prod, 4);
    prod += RS_SHFL_XOR(prod, 2);
    prod += RS_SHFL_XOR(prod, 1);
    const double dinv = RS_SHFL(f, kd);
    const double xj = (y[yo + (size_t) j * 8] - prod) * dinv;
    /* shift: w_t <- w_{t-1}, w_0 <- x_j */
    const double wn = RS_SHFL(w, (t + 7) & 7);
    w = (t == 0) ? xj : wn;
    if (t == 0) y[yo + (size_t) j * 8] = xj;
  }
  RS_SYNCGROUP();
  /* scatter */
  for (int q = t; q < len; q += 8) {
    if (cwrite[off + q]) {
      const int idx = cnode[off + q];
      z[idx] = wm[idx] * y[yo + (size_t) q * 8];
    }
  }
}

/* owner side, group version: in-place band solve of the owned columns,
   y[goff[c] + j], factors in the cabf / cabb layouts at goff[c]*lx */
template <typename F>
__global__ void rs_line_gsolve_group_kernel(double * __restrict__ y,
                                            const F * __restrict__ cabf,
                                            const F * __restrict__ cabb,
                                            const int * __restrict__ goff,
                                            const int * __restrict__ glen,
                                            const int ncol, const int lx)
{
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int c = gid >> 3;
  const int t = gid & 7;
  if (c >= ncol) return;
  RS_DECL_MASK;
  const int len = glen[c];
  const int off = goff[c];
  const int kd = lx - 1;
  const size_t fo = (size_t) off * lx;
  double w = 0.0;
  for (int j = 0; j < len; j++) {
    const double f = (t < lx) ? (double) cabf[fo + (size_t) j * lx + t] : 0.0;
    double prod = (t < kd) ? f * w : 0.0;
    prod += RS_SHFL_XOR(prod, 4);
    prod += RS_SHFL_XOR(prod, 2);
    prod += RS_SHFL_XOR(prod, 1);
    const double dinv = RS_SHFL(f, kd);
    const double vj = (y[off + j] - prod) * dinv;
    const double wn = RS_SHFL(w, (t + 1) & 7);
    w = (t == kd - 1) ? vj : ((t < kd - 1) ? wn : 0.0);
    if (t == 0) y[off + j] = vj;
  }
  RS_SYNCGROUP();
  w = 0.0;
  for (int j = len - 1; j >= 0; j--) {
    const double f = (t < lx) ? (double) cabb[fo + (size_t) j * lx + t] : 0.0;
    double prod = (t < kd) ? f * w : 0.0;
    prod += RS_SHFL_XOR(prod, 4);
    prod += RS_SHFL_XOR(prod, 2);
    prod += RS_SHFL_XOR(prod, 1);
    const double dinv = RS_SHFL(f, kd);
    const double xj = (y[off + j] - prod) * dinv;
    const double wn = RS_SHFL(w, (t + 7) & 7);
    w = (t == 0) ? xj : wn;
    if (t == 0) y[off + j] = xj;
  }
}

/* Schur-complement exchange of the cut columns: per sent piece, the two
   junction scalars s_start = b.y[0:kd), s_end = a.y[len-kd:len) from the
   interior solution y of its local column (blocked layout; written in
   the slot order of the send buffer), and the end junction's weighted
   residual; then the correction z -= xJstart * v + xJend * w at the
   interior nodes, and the end-junction value written by the piece that
   owns it */
__global__ void rs_line_schur_s_kernel(const double * __restrict__ y, const int * __restrict__ cboff,
                                       const int * __restrict__ clen, const int * __restrict__ pc_cl,
                                       const double * __restrict__ pc_b, const double * __restrict__ pc_a,
                                       const int * __restrict__ pc_jend, const double * __restrict__ r,
                                       const double * __restrict__ wsq, const double * __restrict__ mask,
                                       const int * __restrict__ slot, double * __restrict__ pbuf,
                                       const int np, const int lx)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= np) return;
  const int cl = pc_cl[p];
  const int len = clen[cl];
  const int kd = lx - 1;
  const size_t yo = (size_t) cboff[cl >> 3] * 8 + (cl & 7);
  double ss = 0.0, se = 0.0;
  for (int t = 0; t < kd; t++) {
    if (t < len) ss += pc_b[p * kd + t] * y[yo + (size_t) t * 8];
    const int q = len - kd + t;
    if (q >= 0) se += pc_a[p * kd + t] * y[yo + (size_t) q * 8];
  }
  const int je = pc_jend[p];
  const double rj = (je >= 0) ? wsq[je] * mask[je] * r[je] : 0.0;
  const int sl = slot[p];
  pbuf[3 * sl] = ss; pbuf[3 * sl + 1] = se; pbuf[3 * sl + 2] = rj;
}

__global__ void rs_line_schur_corr_kernel(double * __restrict__ z, const int * __restrict__ idx,
                                          const double * __restrict__ v, const double * __restrict__ w,
                                          const int * __restrict__ pp, const int * __restrict__ slot,
                                          const double * __restrict__ xj, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const int sl = slot[pp[i]];
    z[idx[i]] -= xj[2 * sl] * v[i] + xj[2 * sl + 1] * w[i];
  }
}

__global__ void rs_line_schur_jun_kernel(double * __restrict__ z, const double * __restrict__ wsq,
                                         const double * __restrict__ mask, const int * __restrict__ pc_jend,
                                         const int * __restrict__ slot, const double * __restrict__ xj,
                                         const int np)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < np) {
    const int je = pc_jend[p];
    if (je >= 0) z[je] = wsq[je] * mask[je] * xj[2 * slot[p] + 1];
  }
}

/* Owner side of the Schur exchange on the device: one thread per owned
   column, the tridiagonal junction system (diagonal sd, off-diagonal se,
   right-hand side from the received piece scalars rs) by the Thomas
   algorithm in the scratch arrays dd / xj (junction-indexed), the junction
   values (start, end) of every piece of the column into xs (slot order) */
__global__ void rs_line_schur_own_kernel(const double * __restrict__ rs, double * __restrict__ xs,
                                         const int * __restrict__ pr_slot, const int * __restrict__ own_p0,
                                         const int * __restrict__ own_np, const int * __restrict__ jun_off,
                                         const double * __restrict__ sd, const double * __restrict__ se,
                                         double * __restrict__ dd, double * __restrict__ xj, const int nown)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nown) return;
  const int mk = own_np[j], p0 = own_p0[j], jo = jun_off[j];
  for (int k = 0; k < mk - 1; k++) {
    const int s = pr_slot[p0 + k], s1 = pr_slot[p0 + k + 1];
    xj[jo + k] = rs[3 * s + 2] - rs[3 * s + 1] - rs[3 * s1];
    dd[jo + k] = sd[jo + k];
  }
  for (int k = 1; k < mk - 1; k++) {
    const double m = se[jo + k - 1] / dd[jo + k - 1];
    dd[jo + k] -= m * se[jo + k - 1];
    xj[jo + k] -= m * xj[jo + k - 1];
  }
  if (mk > 1) xj[jo + mk - 2] /= dd[jo + mk - 2];
  for (int k = mk - 3; k >= 0; k--)
    xj[jo + k] = (xj[jo + k] - se[jo + k] * xj[jo + k + 1]) / dd[jo + k];
  for (int k = 0; k < mk; k++) {
    const int s = pr_slot[p0 + k];
    xs[2 * s] = (k > 0) ? xj[jo + k - 1] : 0.0;
    xs[2 * s + 1] = (k < mk - 1) ? xj[jo + k] : 0.0;
  }
}


/* Plane strips: in-plane couplings of the plane elements from one comb
   response (comb node off, 0-based): pblk(p + lx t, p0 + lx t0, k0, pe)
   (Fortran order, all 0-based here) = w(node(p, t, k0) of element e).
   pinfo[4 pe .. 4 pe + 3] = element (0-based), d, dc, dn (0-based dirs). */
__global__ void rs_line_plane_extract_kernel(const double * __restrict__ w, double * __restrict__ pblk,
                                             const int * __restrict__ pinfo, const int lx,
                                             const int off, const int npel)
{
  const int nl = lx * lx;
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= npel * nl) return;
  const int pe = g / nl, q = g % nl;
  const int p = q % lx, t = q / lx;
  const int e = pinfo[4 * pe], d = pinfo[4 * pe + 1], dc = pinfo[4 * pe + 2], dn = pinfo[4 * pe + 3];
  int cs[3] = {off % lx, (off / lx) % lx, off / nl};
  const int p0 = cs[d], t0 = cs[dc], k0 = cs[dn];
  int c[3];
  c[d] = p; c[dc] = t; c[dn] = k0;
  const size_t lx3 = (size_t) nl * lx;
  const double a = w[(size_t) e * lx3 + c[0] + lx * (c[1] + lx * c[2])];
  pblk[(size_t) q + nl * ((size_t) (p0 + lx * t0) + nl * ((size_t) k0 + lx * (size_t) pe))] = a;
}

/* Plane strips: one thread block per strip, banded Cholesky solve with
   half bandwidth kd (lower storage, ldab = kd + 1, column-major as dpbtrf),
   the lanes working on the band, the strip vector in shared memory.
   snode: local node (0-based) of every strip row, each row exactly once. */
__global__ void rs_line_strip_solve_kernel(double * __restrict__ z, const double * __restrict__ r,
                                           const double * __restrict__ wsq, const double * __restrict__ mask,
                                           const double * __restrict__ sab, const int * __restrict__ sbo,
                                           const int * __restrict__ snode, const int kd)
{
  extern __shared__ double y[];
  const int s = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const int o = sbo[s], ns = sbo[s + 1] - o, ld = kd + 1;
  const double * __restrict__ L = sab + (size_t) o * ld;
  for (int i = tid; i < ns; i += nt) {
    const int idx = snode[o + i];
    y[i] = wsq[idx] * mask[idx] * r[idx];
  }
  __syncthreads();
  for (int j = 0; j < ns; j++) {                   /* L y = b */
    if (tid == 0) y[j] /= L[(size_t) j * ld];
    __syncthreads();
    const double yj = y[j];
    for (int l = tid + 1; l <= kd && j + l < ns; l += nt)
      y[j + l] -= L[(size_t) j * ld + l] * yj;
    __syncthreads();
  }
  for (int j = ns - 1; j >= 0; j--) {              /* L^T x = y */
    if (tid == 0) y[j] /= L[(size_t) j * ld];
    __syncthreads();
    const double xj = y[j];
    for (int l = tid + 1; l <= kd && j - l >= 0; l += nt)
      y[j - l] -= L[(size_t) (j - l) * ld + l] * xj;
    __syncthreads();
  }
  for (int i = tid; i < ns; i += nt) {
    const int idx = snode[o + i];
    z[idx] = wsq[idx] * mask[idx] * y[i];
  }
}

__global__ void rs_line_addcol3_kernel(double * __restrict__ a,
                                       const double * __restrict__ b,
                                       const double * __restrict__ c,
                                       const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i] * c[i];
}

/* comb pass on the device: x = unit vector at local node off of every
   element; after w = A x, extract per element the three line responses
   through off and the sum of |w| over the element */
/* Neko's working precision (real(rp)): float in single-precision Neko
   builds (compile with -DRS_REAL_SP), double otherwise. The line and coarse
   data (factors, work vectors, Schur pieces) are always double; these
   kernels convert at the boundary and handle the double work vectors. */
#ifdef RS_REAL_SP
typedef float rs_real;
#else
typedef double rs_real;
#endif

__global__ void rs_r2d_kernel(double * __restrict__ d, const rs_real * __restrict__ s, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = (double) s[i];
}

__global__ void rs_d2r_kernel(rs_real * __restrict__ d, const double * __restrict__ s, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = (rs_real) s[i];
}

__global__ void rs_dzero_kernel(double * __restrict__ a, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = 0.0;
}

__global__ void rs_dadd2_kernel(double * __restrict__ a, const double * __restrict__ b, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

__global__ void rs_dcopy_kernel(double * __restrict__ a, const double * __restrict__ b, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = b[i];
}

__global__ void rs_line_comb_set_kernel(double * __restrict__ x, const int lx3,
                                        const int off, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = ((i % lx3) == off) ? 1.0 : 0.0;
}

/* per element: blk3(p, p0, L, d, e) = w at position p of the d-line
   through the comb node (L, p0 = line_of(d, off)); lat3(off, d, e) =
   sum |w| over the element minus the sum over that line; dl = w(off).
   Layouts match the Fortran arrays blk3(lx,lx,nl,3,nelv),
   lat3(lx3,3,nelv), dl(n). */
__global__ void rs_line_comb_extract_kernel(const double * __restrict__ w,
                                            double * __restrict__ blk3,
                                            double * __restrict__ lat3,
                                            double * __restrict__ dl,
                                            const int lx, const int off)
{
  __shared__ double sh[512];
  __shared__ double sl[3];
  const int e = blockIdx.x;
  const int t = threadIdx.x;
  const int lx3 = lx * lx * lx;
  const int nl = lx * lx;
  double v = (t < lx3) ? fabs(w[(size_t) e * lx3 + t]) : 0.0;
  sh[t] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  const double stot = sh[0];
  const int i0 = off % lx, j0 = (off / lx) % lx, k0 = off / (lx * lx);
  if (t < 3) sl[t] = 0.0;
  __syncthreads();
  if (t < 3 * lx) {
    const int d = t / lx, p = t % lx;
    int idx, L, p0;
    if (d == 0) { idx = p + j0 * lx + k0 * nl; p0 = i0; L = j0 + k0 * lx; }
    else if (d == 1) { idx = i0 + p * lx + k0 * nl; p0 = j0; L = i0 + k0 * lx; }
    else { idx = i0 + j0 * lx + p * nl; p0 = k0; L = i0 + j0 * lx; }
    const double a = w[(size_t) e * lx3 + idx];
    blk3[p + lx * (p0 + lx * ((size_t) L + nl * (d + 3 * (size_t) e)))] = a;
    atomicAdd(&sl[d], fabs(a));
  }
  __syncthreads();
  if (t < 3) lat3[off + lx3 * (t + 3 * (size_t) e)] = stot - sl[t];
  if (t == 0) dl[(size_t) e * lx3 + off] = w[(size_t) e * lx3 + off];
}

/* coarse level of rs_colagg on the device: restriction (sum over the
   aggregates with the multiplicity weights), prolongation, CSR matvec */
/* the aggregation level works in Neko's precision (rs_real) */
__global__ void rs_colagg_restrict_kernel(const rs_real * __restrict__ w,
                                          const rs_real * __restrict__ mult,
                                          const int * __restrict__ agg,
                                          rs_real * __restrict__ rc, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&rc[agg[i]], mult[i] * w[i]);
}

__global__ void rs_colagg_prolong_kernel(rs_real * __restrict__ z,
                                         const rs_real * __restrict__ xc,
                                         const int * __restrict__ agg, const int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) z[i] += xc[agg[i]];
}

__global__ void rs_colagg_spmv_kernel(const int * __restrict__ ia,
                                      const int * __restrict__ ja,
                                      const rs_real * __restrict__ av,
                                      const rs_real * __restrict__ p,
                                      rs_real * __restrict__ q, const int m)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    double acc = 0.0;
    for (int k = ia[i]; k < ia[i + 1]; k++) acc += av[k] * p[ja[k]];
    q[i] = (rs_real) acc;
  }
}

/* dense coarse solve of rs_colagg on the device: Cholesky factorization
   (lower) of the replicated coarse operator once, two triangular solves
   per call, through the vendor BLAS / solver libraries */
#ifdef RS_HAVE_HIPBLAS
static rocblas_handle rs_blas = NULL;
static int rs_colagg_potrf_impl(double *A, int n)
{
  if (!rs_blas) rocblas_create_handle(&rs_blas);
  rocblas_int *dinfo; rocblas_int info = -1;
  cudaMalloc((void **) &dinfo, sizeof(rocblas_int));
  rocsolver_dpotrf(rs_blas, rocblas_fill_lower, n, A, n, dinfo);
  cudaMemcpy(&info, dinfo, sizeof(rocblas_int), cudaMemcpyDeviceToHost);
  cudaFree(dinfo);
  return (int) info;
}
static void rs_colagg_trsv2_impl(const double *A, double *x, int n)
{
  if (!rs_blas) rocblas_create_handle(&rs_blas);
  rocblas_dtrsv(rs_blas, rocblas_fill_lower, rocblas_operation_none, rocblas_diagonal_non_unit, n, A, n, x, 1);
  rocblas_dtrsv(rs_blas, rocblas_fill_lower, rocblas_operation_transpose, rocblas_diagonal_non_unit, n, A, n, x, 1);
}
#else
static cublasHandle_t rs_blas = NULL;
static cusolverDnHandle_t rs_solver = NULL;
static int rs_colagg_potrf_impl(double *A, int n)
{
  if (!rs_solver) cusolverDnCreate(&rs_solver);
  int lwork = 0, info = -1, *dinfo; double *work;
  cusolverDnDpotrf_bufferSize(rs_solver, CUBLAS_FILL_MODE_LOWER, n, A, n, &lwork);
  cudaMalloc((void **) &work, (size_t) lwork * sizeof(double));
  cudaMalloc((void **) &dinfo, sizeof(int));
  cusolverDnDpotrf(rs_solver, CUBLAS_FILL_MODE_LOWER, n, A, n, work, lwork, dinfo);
  cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(work); cudaFree(dinfo);
  return info;
}
static void rs_colagg_trsv2_impl(const double *A, double *x, int n)
{
  if (!rs_blas) cublasCreate(&rs_blas);
  cublasDtrsv(rs_blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, A, n, x, 1);
  cublasDtrsv(rs_blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, A, n, x, 1);
}
#endif

extern "C" {
  int rs_line_cuda_backend(void) { return RS_LINE_BACKEND; }

  /* in-place lower Cholesky of the column-major n x n matrix A on the device; returns LAPACK's info */
  int rs_colagg_cuda_potrf(void *A, int n) { return rs_colagg_potrf_impl((double *) A, n); }

  /* x <- L^-T L^-1 x with the factor from rs_colagg_cuda_potrf */
  void rs_colagg_cuda_trsv2(void *A, void *x, int n) { rs_colagg_trsv2_impl((const double *) A, (double *) x, n); }

  void rs_colagg_cuda_restrict(void *w, void *mult, void *agg, void *rc, int n)
  {
    rs_colagg_restrict_kernel<<<(n + 255) / 256, 256>>>((const rs_real *) w, (const rs_real *) mult,
                                                        (const int *) agg, (rs_real *) rc, n);
  }
  void rs_colagg_cuda_prolong(void *z, void *xc, void *agg, int n)
  {
    rs_colagg_prolong_kernel<<<(n + 255) / 256, 256>>>((rs_real *) z, (const rs_real *) xc,
                                                       (const int *) agg, n);
  }
  void rs_colagg_cuda_spmv(void *ia, void *ja, void *av, void *p, void *q, int m)
  {
    rs_colagg_spmv_kernel<<<(m + 255) / 256, 256>>>((const int *) ia, (const int *) ja,
                                                    (const rs_real *) av, (const rs_real *) p,
                                                    (rs_real *) q, m);
  }

  void rs_line_cuda_comb_set(void *x, int lx3, int off, int n)
  {
    const int nb = (n + 255) / 256;
    rs_line_comb_set_kernel<<<nb, 256>>>((double *) x, lx3, off, n);
  }

  void rs_line_cuda_comb_extract(void *w, void *blk3, void *lat3, void *dl, int lx, int off, int nelv)
  {
    rs_line_comb_extract_kernel<<<nelv, 512>>>((const double *) w, (double *) blk3,
                                               (double *) lat3, (double *) dl, lx, off);
  }

  void rs_line_cuda_solve(void *z, void *r, void *cab, void *cnode, void *cwrite,
                          void *coff, void *clen, void *wsq, void *mask, void *y,
                          int ncol, int lx)
  {
    if (ncol <= 0) return;
    const int nb = (ncol + 127) / 128;
    rs_line_solve_kernel<<<nb, 128>>>((double *) z, (const double *) r,
                                      (const double *) cab, (const int *) cnode,
                                      (const int *) cwrite, (const int *) coff,
                                      (const int *) clen, (const double *) wsq,
                                      (const double *) mask, (double *) y, ncol, lx);
  }

  void rs_line_cuda_gpack(void *r, void *wsq, void *mask, void *idx, void *sb, int n)
  {
    if (n <= 0) return;
    rs_line_gpack_kernel<<<(n + 255) / 256, 256>>>((const double *) r, (const double *) wsq,
                                                   (const double *) mask, (const int *) idx,
                                                   (double *) sb, n);
  }

  void rs_line_cuda_gunpack(void *z, void *wsq, void *mask, void *idx, void *wr, void *sb, int n)
  {
    if (n <= 0) return;
    rs_line_gunpack_kernel<<<(n + 255) / 256, 256>>>((double *) z, (const double *) wsq,
                                                     (const double *) mask, (const int *) idx,
                                                     (const int *) wr, (const double *) sb, n);
  }

  void rs_line_cuda_gfill(void *gy, void *rb, void *map, int n)
  {
    if (n <= 0) return;
    rs_line_gfill_kernel<<<(n + 255) / 256, 256>>>((double *) gy, (const double *) rb,
                                                   (const int *) map, n);
  }

  void rs_line_cuda_gunfill(void *rb, void *gy, void *map, int n)
  {
    if (n <= 0) return;
    rs_line_gunfill_kernel<<<(n + 255) / 256, 256>>>((double *) rb, (const double *) gy,
                                                     (const int *) map, n);
  }

  void rs_line_cuda_gsolve(void *gy, void *cab, void *glen, int ncol, int lx)
  {
    if (ncol <= 0) return;
    rs_line_gsolve_kernel<<<(ncol + 127) / 128, 128>>>((double *) gy, (const double *) cab,
                                                       (const int *) glen, ncol, lx);
  }

  /* fsp != 0: the factors cabf / cabb are float */
  void rs_line_cuda_solve_group(void *z, void *r, void *cabf, void *cabb, void *cnode, void *cwrite,
                                void *coff, void *clen, void *cboff, void *wm, void *y,
                                int ncol, int lx, int fsp)
  {
    if (ncol <= 0) return;
    const int nthr = ncol * 8;
    const int nb = (nthr + 255) / 256;
    if (fsp)
      rs_line_solve_group_kernel<float><<<nb, 256>>>((double *) z, (const double *) r,
          (const float *) cabf, (const float *) cabb, (const int *) cnode, (const int *) cwrite,
          (const int *) coff, (const int *) clen, (const int *) cboff,
          (const double *) wm, (double *) y, ncol, lx);
    else
      rs_line_solve_group_kernel<double><<<nb, 256>>>((double *) z, (const double *) r,
          (const double *) cabf, (const double *) cabb, (const int *) cnode, (const int *) cwrite,
          (const int *) coff, (const int *) clen, (const int *) cboff,
          (const double *) wm, (double *) y, ncol, lx);
  }

  void rs_line_cuda_gsolve_group(void *gy, void *cabf, void *cabb, void *goff, void *glen, int ncol, int lx,
                                 int fsp)
  {
    if (ncol <= 0) return;
    const int nb = (ncol * 8 + 255) / 256;
    if (fsp)
      rs_line_gsolve_group_kernel<float><<<nb, 256>>>((double *) gy, (const float *) cabf,
          (const float *) cabb, (const int *) goff, (const int *) glen, ncol, lx);
    else
      rs_line_gsolve_group_kernel<double><<<nb, 256>>>((double *) gy, (const double *) cabf,
          (const double *) cabb, (const int *) goff, (const int *) glen, ncol, lx);
  }

  void rs_line_cuda_schur_s(void *y, void *cboff, void *clen, void *pc_cl, void *pc_b, void *pc_a,
                            void *pc_jend, void *r, void *wsq, void *mask, void *slot, void *pbuf,
                            int np, int lx)
  {
    if (np <= 0) return;
    rs_line_schur_s_kernel<<<(np + 255) / 256, 256>>>((const double *) y, (const int *) cboff,
        (const int *) clen, (const int *) pc_cl, (const double *) pc_b, (const double *) pc_a,
        (const int *) pc_jend, (const double *) r, (const double *) wsq, (const double *) mask,
        (const int *) slot, (double *) pbuf, np, lx);
  }

  void rs_line_cuda_schur_corr(void *z, void *idx, void *v, void *w, void *pp, void *slot, void *xj, int n)
  {
    if (n <= 0) return;
    rs_line_schur_corr_kernel<<<(n + 255) / 256, 256>>>((double *) z, (const int *) idx, (const double *) v,
        (const double *) w, (const int *) pp, (const int *) slot, (const double *) xj, n);
  }

  void rs_line_cuda_schur_jun(void *z, void *wsq, void *mask, void *pc_jend, void *slot, void *xj, int np)
  {
    if (np <= 0) return;
    rs_line_schur_jun_kernel<<<(np + 255) / 256, 256>>>((double *) z, (const double *) wsq,
        (const double *) mask, (const int *) pc_jend, (const int *) slot, (const double *) xj, np);
  }

  void rs_line_cuda_schur_own(void *rs, void *xs, void *pr_slot, void *own_p0, void *own_np,
                              void *jun_off, void *sd, void *se, void *dd, void *xj, int nown)
  {
    if (nown <= 0) return;
    rs_line_schur_own_kernel<<<(nown + 127) / 128, 128>>>((const double *) rs, (double *) xs,
        (const int *) pr_slot, (const int *) own_p0, (const int *) own_np, (const int *) jun_off,
        (const double *) sd, (const double *) se, (double *) dd, (double *) xj, nown);
  }


  void rs_line_cuda_plane_extract(void *w, void *pblk, void *pinfo, int lx, int off, int npel)
  {
    if (npel <= 0) return;
    const int n = npel * lx * lx;
    rs_line_plane_extract_kernel<<<(n + 255) / 256, 256>>>((const double *) w, (double *) pblk,
        (const int *) pinfo, lx, off, npel);
  }

  void rs_line_cuda_strip_solve(void *z, void *r, void *wsq, void *mask, void *sab, void *sbo,
                                void *snode, int nstrip, int kd, int maxns)
  {
    if (nstrip <= 0) return;
    rs_line_strip_solve_kernel<<<nstrip, 64, (size_t) maxns * sizeof(double)>>>((double *) z,
        (const double *) r, (const double *) wsq, (const double *) mask, (const double *) sab,
        (const int *) sbo, (const int *) snode, kd);
  }

  /* copies with 64-bit byte counts (Neko's device_memcpy takes a 32-bit
     element count): dir 1 host -> device, 2 device -> host */
  void rs_line_cuda_copy(void *dst, void *src, size_t bytes, int dir)
  {
    if (bytes == 0) return;
    cudaError_t e = cudaMemcpy(dst, src, bytes, dir == 1 ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
      fprintf(stderr, "rs_line_cuda: copy of %zu bytes failed: %s\n", bytes, cudaGetErrorString(e));
      abort();
    }
  }

  void rs_line_cuda_addcol3(void *a, void *b, void *c, int n)
  {
    if (n <= 0) return;
    const int nb = (n + 255) / 256;
    rs_line_addcol3_kernel<<<nb, 256>>>((double *) a, (const double *) b,
                                        (const double *) c, n);
  }

  /* precision boundary and double work vectors (see rs_r2d_kernel) */
  int rs_line_cuda_real_bytes(void) { return (int) sizeof(rs_real); }
  void rs_line_cuda_r2d(void *d, void *s, int n)
  {
    if (n > 0) rs_r2d_kernel<<<(n + 255) / 256, 256>>>((double *) d, (const rs_real *) s, n);
  }
  void rs_line_cuda_d2r(void *d, void *s, int n)
  {
    if (n > 0) rs_d2r_kernel<<<(n + 255) / 256, 256>>>((rs_real *) d, (const double *) s, n);
  }
  void rs_line_cuda_dzero(void *a, int n)
  {
    if (n > 0) rs_dzero_kernel<<<(n + 255) / 256, 256>>>((double *) a, n);
  }
  void rs_line_cuda_dadd2(void *a, void *b, int n)
  {
    if (n > 0) rs_dadd2_kernel<<<(n + 255) / 256, 256>>>((double *) a, (const double *) b, n);
  }
  void rs_line_cuda_dcopy(void *a, void *b, int n)
  {
    if (n > 0) rs_dcopy_kernel<<<(n + 255) / 256, 256>>>((double *) a, (const double *) b, n);
  }

  /* pin host buffers that cross to the device every call (the packed MPI
     buffers): pageable copies are several times slower on both vendors */
  void rs_line_cuda_pin(void *p, size_t bytes)
  {
    if (p == NULL || bytes == 0) return;
    cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterPortable);
    if (e != cudaSuccess) {
      fprintf(stderr, "rs_line_cuda: host pinning failed (%s), continuing unpinned\n", cudaGetErrorString(e));
      (void) cudaGetLastError();
    }
  }

  void rs_line_cuda_unpin(void *p)
  {
    if (p == NULL) return;
    if (cudaHostUnregister(p) != cudaSuccess) (void) cudaGetLastError();
  }

  void rs_line_cuda_sync(void)
  {
    cudaError_t e1 = cudaGetLastError();
    cudaError_t e2 = cudaDeviceSynchronize();
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
      fprintf(stderr, "rs_line_cuda: CUDA error: %s / %s\n", cudaGetErrorString(e1),
              cudaGetErrorString(e2));
      abort();
    }
  }
}
