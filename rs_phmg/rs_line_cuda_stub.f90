!> CPU-only builds: the GPU entry points of rs_line are never called
!! (NEKO_BCKND_DEVICE = 0) but must link. Use this file instead of
!! rs_line_cuda.cu (CUDA) or rs_line_hip.cpp (HIP) on the makeneko line of a
!! CPU build.
module rs_line_cuda_stub
  use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_size_t
  implicit none
contains
  function rs_line_cuda_backend() bind(c, name = 'rs_line_cuda_backend') result(b)
    integer(c_int) :: b
    b = 0
  end function rs_line_cuda_backend
  subroutine rs_line_cuda_solve(z, r, cab, cnode, cwrite, coff, clen, wsq, mask, y, &
       ncol, lx) bind(c, name = 'rs_line_cuda_solve')
    type(c_ptr), value :: z, r, cab, cnode, cwrite, coff, clen, wsq, mask, y
    integer(c_int), value :: ncol, lx
  end subroutine rs_line_cuda_solve
  subroutine rs_line_cuda_gpack(r, wsq, mask, idx, sb, n) bind(c, name = 'rs_line_cuda_gpack')
    type(c_ptr), value :: r, wsq, mask, idx, sb
    integer(c_int), value :: n
  end subroutine rs_line_cuda_gpack
  subroutine rs_line_cuda_gunpack(z, wsq, mask, idx, wr, sb, n) bind(c, name = 'rs_line_cuda_gunpack')
    type(c_ptr), value :: z, wsq, mask, idx, wr, sb
    integer(c_int), value :: n
  end subroutine rs_line_cuda_gunpack
  subroutine rs_line_cuda_gfill(gy, rb, map, n) bind(c, name = 'rs_line_cuda_gfill')
    type(c_ptr), value :: gy, rb, map
    integer(c_int), value :: n
  end subroutine rs_line_cuda_gfill
  subroutine rs_line_cuda_gunfill(rb, gy, map, n) bind(c, name = 'rs_line_cuda_gunfill')
    type(c_ptr), value :: rb, gy, map
    integer(c_int), value :: n
  end subroutine rs_line_cuda_gunfill
  subroutine rs_line_cuda_gsolve(gy, cab, glen, ncol, lx) bind(c, name = 'rs_line_cuda_gsolve')
    type(c_ptr), value :: gy, cab, glen
    integer(c_int), value :: ncol, lx
  end subroutine rs_line_cuda_gsolve
  subroutine rs_line_cuda_solve_group(z, r, cabf, cabb, cnode, cwrite, coff, clen, cboff, wm, y, &
       ncol, lx, fsp) bind(c, name = 'rs_line_cuda_solve_group')
    type(c_ptr), value :: z, r, cabf, cabb, cnode, cwrite, coff, clen, cboff, wm, y
    integer(c_int), value :: ncol, lx, fsp
  end subroutine rs_line_cuda_solve_group
  subroutine rs_line_cuda_gsolve_group(gy, cabf, cabb, goff, glen, ncol, lx, fsp) &
       bind(c, name = 'rs_line_cuda_gsolve_group')
    type(c_ptr), value :: gy, cabf, cabb, goff, glen
    integer(c_int), value :: ncol, lx, fsp
  end subroutine rs_line_cuda_gsolve_group
  subroutine rs_line_cuda_schur_s(y, cboff, clen, pc_cl, pc_b, pc_a, pc_jend, r, wsq, mask, slot, pbuf, &
       np, lx) bind(c, name = 'rs_line_cuda_schur_s')
    type(c_ptr), value :: y, cboff, clen, pc_cl, pc_b, pc_a, pc_jend, r, wsq, mask, slot, pbuf
    integer(c_int), value :: np, lx
  end subroutine rs_line_cuda_schur_s
  subroutine rs_line_cuda_schur_corr(z, idx, v, w, pp, slot, xj, n) bind(c, name = 'rs_line_cuda_schur_corr')
    type(c_ptr), value :: z, idx, v, w, pp, slot, xj
    integer(c_int), value :: n
  end subroutine rs_line_cuda_schur_corr
  subroutine rs_line_cuda_schur_jun(z, wsq, mask, pc_jend, slot, xj, np) bind(c, name = 'rs_line_cuda_schur_jun')
    type(c_ptr), value :: z, wsq, mask, pc_jend, slot, xj
    integer(c_int), value :: np
  end subroutine rs_line_cuda_schur_jun
  subroutine rs_line_cuda_copy(dst, src, bytes, dir) bind(c, name = 'rs_line_cuda_copy')
    type(c_ptr), value :: dst, src
    integer(c_size_t), value :: bytes
    integer(c_int), value :: dir
  end subroutine rs_line_cuda_copy
  subroutine rs_line_cuda_plane_extract(w, pblk, pinfo, lx, off, npel) bind(c, name = 'rs_line_cuda_plane_extract')
    type(c_ptr), value :: w, pblk, pinfo
    integer(c_int), value :: lx, off, npel
  end subroutine rs_line_cuda_plane_extract
  subroutine rs_line_cuda_strip_solve(z, r, wsq, mask, sab, sbo, snode, nstrip, kd, maxns) &
       bind(c, name = 'rs_line_cuda_strip_solve')
    type(c_ptr), value :: z, r, wsq, mask, sab, sbo, snode
    integer(c_int), value :: nstrip, kd, maxns
  end subroutine rs_line_cuda_strip_solve
  subroutine rs_line_cuda_schur_own(rs, xs, pr_slot, own_p0, own_np, jun_off, sd, se, dd, xj, nown) &
       bind(c, name = 'rs_line_cuda_schur_own')
    type(c_ptr), value :: rs, xs, pr_slot, own_p0, own_np, jun_off, sd, se, dd, xj
    integer(c_int), value :: nown
  end subroutine rs_line_cuda_schur_own
  function rs_line_cuda_real_bytes() bind(c, name = 'rs_line_cuda_real_bytes') result(b)
    integer(c_int) :: b
    b = 0
  end function rs_line_cuda_real_bytes
  subroutine rs_line_cuda_r2d(d, s, n) bind(c, name = 'rs_line_cuda_r2d')
    type(c_ptr), value :: d, s
    integer(c_int), value :: n
  end subroutine rs_line_cuda_r2d
  subroutine rs_line_cuda_d2r(d, s, n) bind(c, name = 'rs_line_cuda_d2r')
    type(c_ptr), value :: d, s
    integer(c_int), value :: n
  end subroutine rs_line_cuda_d2r
  subroutine rs_line_cuda_dzero(a, n) bind(c, name = 'rs_line_cuda_dzero')
    type(c_ptr), value :: a
    integer(c_int), value :: n
  end subroutine rs_line_cuda_dzero
  subroutine rs_line_cuda_dadd2(a, b, n) bind(c, name = 'rs_line_cuda_dadd2')
    type(c_ptr), value :: a, b
    integer(c_int), value :: n
  end subroutine rs_line_cuda_dadd2
  subroutine rs_line_cuda_dcopy(a, b, n) bind(c, name = 'rs_line_cuda_dcopy')
    type(c_ptr), value :: a, b
    integer(c_int), value :: n
  end subroutine rs_line_cuda_dcopy
  subroutine rs_line_cuda_addcol3(a, b, c, n) bind(c, name = 'rs_line_cuda_addcol3')
    type(c_ptr), value :: a, b, c
    integer(c_int), value :: n
  end subroutine rs_line_cuda_addcol3
  subroutine rs_line_cuda_pin(p, bytes) bind(c, name = 'rs_line_cuda_pin')
    type(c_ptr), value :: p
    integer(c_size_t), value :: bytes
  end subroutine rs_line_cuda_pin
  subroutine rs_line_cuda_unpin(p) bind(c, name = 'rs_line_cuda_unpin')
    type(c_ptr), value :: p
  end subroutine rs_line_cuda_unpin
  subroutine rs_line_cuda_sync() bind(c, name = 'rs_line_cuda_sync')
    ! reached only on a device build linked with the stub (HIP/OpenCL): refuse
    error stop 'rs_line: device backend without the GPU kernels (rs_line_cuda.cu / rs_line_hip.cpp)'
  end subroutine rs_line_cuda_sync
  subroutine rs_line_cuda_comb_set(x, lx3, off, n) bind(c, name = 'rs_line_cuda_comb_set')
    type(c_ptr), value :: x
    integer(c_int), value :: lx3, off, n
    error stop 'rs_line: device backend without the GPU kernels (rs_line_cuda.cu / rs_line_hip.cpp)'
  end subroutine rs_line_cuda_comb_set
  subroutine rs_line_cuda_comb_extract(w, blk3, lat3, dl, lx, off, nelv) &
       bind(c, name = 'rs_line_cuda_comb_extract')
    type(c_ptr), value :: w, blk3, lat3, dl
    integer(c_int), value :: lx, off, nelv
  end subroutine rs_line_cuda_comb_extract
  function rs_colagg_cuda_potrf(a, n) bind(c, name = 'rs_colagg_cuda_potrf') result(info)
    type(c_ptr), value :: a
    integer(c_int), value :: n
    integer(c_int) :: info
    info = -1
  end function rs_colagg_cuda_potrf
  subroutine rs_colagg_cuda_trsv2(a, x, n) bind(c, name = 'rs_colagg_cuda_trsv2')
    type(c_ptr), value :: a, x
    integer(c_int), value :: n
  end subroutine rs_colagg_cuda_trsv2
  subroutine rs_colagg_cuda_restrict(w, mult, agg, rc, n) bind(c, name = 'rs_colagg_cuda_restrict')
    type(c_ptr), value :: w, mult, agg, rc
    integer(c_int), value :: n
  end subroutine rs_colagg_cuda_restrict
  subroutine rs_colagg_cuda_prolong(z, xc, agg, n) bind(c, name = 'rs_colagg_cuda_prolong')
    type(c_ptr), value :: z, xc, agg
    integer(c_int), value :: n
  end subroutine rs_colagg_cuda_prolong
  subroutine rs_colagg_cuda_spmv(ia, ja, av, p, q, m) bind(c, name = 'rs_colagg_cuda_spmv')
    type(c_ptr), value :: ia, ja, av, p, q
    integer(c_int), value :: m
  end subroutine rs_colagg_cuda_spmv
end module rs_line_cuda_stub
