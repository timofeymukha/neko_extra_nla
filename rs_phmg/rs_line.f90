!> Line (block) relaxation along the stiff direction of each element, as a
!! `pc_t` for use as the accelerator of Neko's Chebyshev smoother inside
!! `rs_phmg` (`smoother_cheby_acc = "line"`).
!!
!! The stiff reference direction of each element is chosen algebraically
!! (default): the comb vectors applied to the element-local Ax give the
!! couplings along all three reference directions, and the direction with
!! the largest off-diagonal line coupling wins ("geometric" uses the
!! geometric factors instead). For every GLL line along it, the block of the
!! *assembled* operator restricted to the line (local line couplings,
!! assembled diagonal, Dirichlet nodes decoupled) is used in two modes:
!!
!!  - element mode: each element line is solved on its own (LU of lx x lx);
!!  - column mode: lines are chained across element faces normal to the
!!    stiff direction into column lines spanning all stacked elements of
!!    the same direction on this rank (neighbours found by matching unique
!!    dof ids); each column line is a banded system, factored once.
!!
!! Application: z = sum_lines R^T W^{1/2} A_line^{-1} W^{1/2} R r, with
!! W = 1 / (number of lines containing the node), one write per unique node
!! and line, then gather-scatter (sum). Symmetric; reduces to Neko's Jacobi
!! for point blocks. CPU, double precision (LAPACK) only.
module rs_line
  use num_types, only : rp, sp, dp, i8
  use precon, only : pc_t
  use coefs, only : coef_t
  use bc_list, only : bc_list_t
  use json_module, only : json_file
  use ax_product, only : ax_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use scalar_bc_projector, only : scalar_bc_projector_t
  use utils, only : neko_error
  use logger, only : neko_log, LOG_SIZE
  use comm, only : pe_rank, pe_size, NEKO_COMM
  use, intrinsic :: iso_c_binding, only : c_ptr, c_int, c_size_t, c_null_ptr, c_associated, c_loc, &
       c_f_pointer, c_intptr_t
  use device, only : device_map, device_memcpy, device_get_ptr, device_free, device_alloc, &
       device_deassociate, device_sync, HOST_TO_DEVICE, DEVICE_TO_HOST
  use device_math, only : device_rzero, device_add2, device_addcol3, device_copy
  use neko_config, only : NEKO_BCKND_DEVICE, NEKO_BCKND_CUDA, NEKO_BCKND_HIP, NEKO_BCKND_OPENCL
  use gs_device_mpi, only : gs_device_mpi_t
  use mpi_f08, only : MPI_Allgather, MPI_Allgatherv, MPI_Alltoallv, MPI_Allreduce, &
       MPI_INTEGER, MPI_INTEGER8, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_MAX, MPI_IN_PLACE, &
       MPI_Wtime, MPI_LOGICAL, MPI_LAND
  implicit none
  private

  interface
     !> which GPU kernels are linked: 0 stub, 1 CUDA (rs_line_cuda.cu), 2 HIP (rs_line_hip.cpp)
     function rs_line_cuda_backend() bind(c, name = 'rs_line_cuda_backend') result(b)
       import c_int
       integer(c_int) :: b
     end function rs_line_cuda_backend
     subroutine rs_line_cuda_solve(z, r, cab, cnode, cwrite, coff, clen, wsq, mask, y, &
          ncol, lx) bind(c, name = 'rs_line_cuda_solve')
       import c_ptr, c_int
       type(c_ptr), value :: z, r, cab, cnode, cwrite, coff, clen, wsq, mask, y
       integer(c_int), value :: ncol, lx
     end subroutine rs_line_cuda_solve
     !> size in bytes of the kernels' rs_real (Neko's rp: -DRS_REAL_SP for single precision)
     function rs_line_cuda_real_bytes() bind(c, name = 'rs_line_cuda_real_bytes') result(b)
       import c_int
       integer(c_int) :: b
     end function rs_line_cuda_real_bytes
     !> real(rp) -> double and back on the device; double work-vector helpers
     subroutine rs_line_cuda_r2d(d, s, n) bind(c, name = 'rs_line_cuda_r2d')
       import c_ptr, c_int
       type(c_ptr), value :: d, s
       integer(c_int), value :: n
     end subroutine rs_line_cuda_r2d
     subroutine rs_line_cuda_d2r(d, s, n) bind(c, name = 'rs_line_cuda_d2r')
       import c_ptr, c_int
       type(c_ptr), value :: d, s
       integer(c_int), value :: n
     end subroutine rs_line_cuda_d2r
     subroutine rs_line_cuda_dzero(a, n) bind(c, name = 'rs_line_cuda_dzero')
       import c_ptr, c_int
       type(c_ptr), value :: a
       integer(c_int), value :: n
     end subroutine rs_line_cuda_dzero
     subroutine rs_line_cuda_dadd2(a, b, n) bind(c, name = 'rs_line_cuda_dadd2')
       import c_ptr, c_int
       type(c_ptr), value :: a, b
       integer(c_int), value :: n
     end subroutine rs_line_cuda_dadd2
     subroutine rs_line_cuda_dcopy(a, b, n) bind(c, name = 'rs_line_cuda_dcopy')
       import c_ptr, c_int
       type(c_ptr), value :: a, b
       integer(c_int), value :: n
     end subroutine rs_line_cuda_dcopy
     subroutine rs_line_cuda_addcol3(a, b, c, n) bind(c, name = 'rs_line_cuda_addcol3')
       import c_ptr, c_int
       type(c_ptr), value :: a, b, c
       integer(c_int), value :: n
     end subroutine rs_line_cuda_addcol3
     subroutine rs_line_cuda_sync() bind(c, name = 'rs_line_cuda_sync')
     end subroutine rs_line_cuda_sync
     subroutine rs_line_cuda_pin(p, bytes) bind(c, name = 'rs_line_cuda_pin')
       import c_ptr, c_size_t
       type(c_ptr), value :: p
       integer(c_size_t), value :: bytes
     end subroutine rs_line_cuda_pin
     subroutine rs_line_cuda_unpin(p) bind(c, name = 'rs_line_cuda_unpin')
       import c_ptr
       type(c_ptr), value :: p
     end subroutine rs_line_cuda_unpin
     subroutine rs_line_cuda_solve_group(z, r, cabf, cabb, cnode, cwrite, coff, clen, cboff, wm, &
          y, ncol, lx, fsp) bind(c, name = 'rs_line_cuda_solve_group')
       import c_ptr, c_int
       type(c_ptr), value :: z, r, cabf, cabb, cnode, cwrite, coff, clen, cboff, wm, y
       integer(c_int), value :: ncol, lx, fsp
     end subroutine rs_line_cuda_solve_group
     subroutine rs_line_cuda_gpack(r, wsq, mask, idx, sb, n) bind(c, name = 'rs_line_cuda_gpack')
       import c_ptr, c_int
       type(c_ptr), value :: r, wsq, mask, idx, sb
       integer(c_int), value :: n
     end subroutine rs_line_cuda_gpack
     subroutine rs_line_cuda_gunpack(z, wsq, mask, idx, wr, sb, n) bind(c, name = 'rs_line_cuda_gunpack')
       import c_ptr, c_int
       type(c_ptr), value :: z, wsq, mask, idx, wr, sb
       integer(c_int), value :: n
     end subroutine rs_line_cuda_gunpack
     subroutine rs_line_cuda_gfill(gy, rb, map, n) bind(c, name = 'rs_line_cuda_gfill')
       import c_ptr, c_int
       type(c_ptr), value :: gy, rb, map
       integer(c_int), value :: n
     end subroutine rs_line_cuda_gfill
     subroutine rs_line_cuda_gunfill(rb, gy, map, n) bind(c, name = 'rs_line_cuda_gunfill')
       import c_ptr, c_int
       type(c_ptr), value :: rb, gy, map
       integer(c_int), value :: n
     end subroutine rs_line_cuda_gunfill
     subroutine rs_line_cuda_gsolve(gy, cab, glen, ncol, lx) bind(c, name = 'rs_line_cuda_gsolve')
       import c_ptr, c_int
       type(c_ptr), value :: gy, cab, glen
       integer(c_int), value :: ncol, lx
     end subroutine rs_line_cuda_gsolve
     subroutine rs_line_cuda_schur_s(y, cboff, clen, pc_cl, pc_b, pc_a, pc_jend, r, wsq, mask, &
          slot, pbuf, np, lx) bind(c, name = 'rs_line_cuda_schur_s')
       import c_ptr, c_int
       type(c_ptr), value :: y, cboff, clen, pc_cl, pc_b, pc_a, pc_jend, r, wsq, mask, slot, pbuf
       integer(c_int), value :: np, lx
     end subroutine rs_line_cuda_schur_s
     subroutine rs_line_cuda_schur_corr(z, idx, v, w, pp, slot, xj, n) bind(c, name = 'rs_line_cuda_schur_corr')
       import c_ptr, c_int
       type(c_ptr), value :: z, idx, v, w, pp, slot, xj
       integer(c_int), value :: n
     end subroutine rs_line_cuda_schur_corr
     subroutine rs_line_cuda_schur_jun(z, wsq, mask, pc_jend, slot, xj, np) bind(c, name = 'rs_line_cuda_schur_jun')
       import c_ptr, c_int
       type(c_ptr), value :: z, wsq, mask, pc_jend, slot, xj
       integer(c_int), value :: np
     end subroutine rs_line_cuda_schur_jun
     subroutine rs_line_cuda_copy(dst, src, bytes, dir) bind(c, name = 'rs_line_cuda_copy')
       import c_ptr, c_size_t, c_int
       type(c_ptr), value :: dst, src
       integer(c_size_t), value :: bytes
       integer(c_int), value :: dir
     end subroutine rs_line_cuda_copy
     subroutine rs_line_cuda_plane_extract(w, pblk, pinfo, lx, off, npel) &
          bind(c, name = 'rs_line_cuda_plane_extract')
       import c_ptr, c_int
       type(c_ptr), value :: w, pblk, pinfo
       integer(c_int), value :: lx, off, npel
     end subroutine rs_line_cuda_plane_extract
     subroutine rs_line_cuda_strip_solve(z, r, wsq, mask, sab, sbo, snode, nstrip, kd, maxns) &
          bind(c, name = 'rs_line_cuda_strip_solve')
       import c_ptr, c_int
       type(c_ptr), value :: z, r, wsq, mask, sab, sbo, snode
       integer(c_int), value :: nstrip, kd, maxns
     end subroutine rs_line_cuda_strip_solve
     subroutine rs_line_cuda_schur_own(rs, xs, pr_slot, own_p0, own_np, jun_off, sd, se, dd, xj, nown) &
          bind(c, name = 'rs_line_cuda_schur_own')
       import c_ptr, c_int
       type(c_ptr), value :: rs, xs, pr_slot, own_p0, own_np, jun_off, sd, se, dd, xj
       integer(c_int), value :: nown
     end subroutine rs_line_cuda_schur_own
     !> Neko's point-to-point helpers (src/gs/bcknd/device/device_mpi.c, in
     !! every libneko): MPI_Isend / MPI_Irecv on NEKO_COMM of a raw pointer,
     !! device memory with a CUDA/ROCm-aware MPI or host memory
     subroutine rs_mpi_init_reqs(n, reqs) bind(c, name = 'device_mpi_init_reqs')
       import c_ptr, c_int
       integer(c_int), value :: n
       type(c_ptr) :: reqs
     end subroutine rs_mpi_init_reqs
     subroutine rs_mpi_free_reqs(reqs) bind(c, name = 'device_mpi_free_reqs')
       import c_ptr
       type(c_ptr) :: reqs
     end subroutine rs_mpi_free_reqs
     subroutine rs_mpi_isend(buf, offset, nbytes, rank, tag, reqs, i) bind(c, name = 'device_mpi_isend')
       import c_ptr, c_int
       integer(c_int), value :: offset, nbytes, rank, tag, i
       type(c_ptr), value :: buf, reqs
     end subroutine rs_mpi_isend
     subroutine rs_mpi_irecv(buf, offset, nbytes, rank, tag, reqs, i) bind(c, name = 'device_mpi_irecv')
       import c_ptr, c_int
       integer(c_int), value :: offset, nbytes, rank, tag, i
       type(c_ptr), value :: buf, reqs
     end subroutine rs_mpi_irecv
     subroutine rs_mpi_waitall(n, reqs) bind(c, name = 'device_mpi_waitall')
       import c_ptr, c_int
       integer(c_int), value :: n
       type(c_ptr), value :: reqs
     end subroutine rs_mpi_waitall
     subroutine rs_line_cuda_gsolve_group(gy, cabf, cabb, goff, glen, ncol, lx, fsp) &
          bind(c, name = 'rs_line_cuda_gsolve_group')
       import c_ptr, c_int
       type(c_ptr), value :: gy, cabf, cabb, goff, glen
       integer(c_int), value :: ncol, lx, fsp
     end subroutine rs_line_cuda_gsolve_group
     subroutine rs_line_cuda_comb_set(x, lx3, off, n) bind(c, name = 'rs_line_cuda_comb_set')
       import c_ptr, c_int
       type(c_ptr), value :: x
       integer(c_int), value :: lx3, off, n
     end subroutine rs_line_cuda_comb_set
     subroutine rs_line_cuda_comb_extract(w, blk3, lat3, dl, lx, off, nelv) &
          bind(c, name = 'rs_line_cuda_comb_extract')
       import c_ptr, c_int
       type(c_ptr), value :: w, blk3, lat3, dl
       integer(c_int), value :: lx, off, nelv
     end subroutine rs_line_cuda_comb_extract
  end interface

  public :: hgs, hbcp, rs_line_cuda_sync, rs_mem_report

  type, public, extends(pc_t) :: rs_line_t
     !> device (CUDA) port: the rank-local columns packed for one thread
     !! per column; the distributed columns go through the host
     integer :: ncol_loc = 0
     integer, allocatable :: cnode(:), cwrite(:), coff(:), clen(:)
     real(kind=dp), allocatable :: cabd(:), ywork(:), isocoef(:), zh(:)
     type(c_ptr) :: cnode_d = c_null_ptr, cwrite_d = c_null_ptr, coff_d = c_null_ptr
     type(c_ptr) :: clen_d = c_null_ptr, cabd_d = c_null_ptr, ywork_d = c_null_ptr
     !> double work copies of r and z on the device when Neko's rp is not dp
     !! (the line data and kernels are double throughout)
     type(c_ptr) :: rw_d = c_null_ptr, zw_d = c_null_ptr
     !> wsq * mask on the device: one strided array in the group kernel
     real(kind=dp), allocatable :: wm(:)
     type(c_ptr) :: wm_d = c_null_ptr
     type(c_ptr) :: isocoef_d = c_null_ptr, wsq_d = c_null_ptr, mask_d = c_null_ptr
     type(c_ptr) :: zh_d = c_null_ptr
     logical :: check_device = .false.
     !> device solve timers (env RS_TIMERS): local columns (+ isotropic), distributed columns, gs
     logical :: timers = .false.
     real(kind=dp) :: tsol(3) = 0.0_dp
     real(kind=dp) :: tstrip = 0.0_dp                     ! device solve: strip part (inside "distributed")
     integer :: nsol = 0
     !> sub-timers of the distributed columns (device path): pack kernel,
     !! copy to host, Alltoallv, copy to device, owners' kernels, copy to
     !! host, Alltoallv back, copy to device, unpack kernel
     real(kind=dp) :: tdist(9) = 0.0_dp
     integer :: lx = 0
     integer :: nelv = 0
     integer :: n = 0
     logical :: columns = .false.
     !> link the chains across ranks into global columns (owner-solved);
     !! false = every chain stops at the rank boundary (no exchange, the
     !! coupling across the cut is dropped)
     logical :: global_columns = .true.
     !> coupling-ratio threshold below which an element is treated as
     !! isotropic (diagonal smoothing, no lines, no chaining); 0 = off
     real(kind=dp) :: iso_ratio = 0.0_dp
     integer :: niso = 0
     logical, allocatable :: iso(:)
     !> plane strips (option planes): an element whose middle coupling
     !! dominates its weakest by plane_ratio is a
     !! plane element: no column lines; its node layers normal to the
     !! weakest direction are chained along the strongest direction (dir)
     !! into strips (bundles of lx lines of one layer), each a banded block
     !! with half bandwidth lx**2 - 1 assembled from the in-plane couplings
     logical :: planes = .false.
     real(kind=dp) :: plane_ratio = 16.0_dp
     logical, allocatable :: pel(:)
     integer, allocatable :: pnorm(:), pid(:)
     integer :: npel = 0
     real(kind=dp), allocatable :: pblk(:,:,:,:)          ! (lx*lx, lx*lx, lx, npel)
     integer :: nstrip = 0, smaxseg = 0, smaxn = 0
     integer, allocatable :: snseg(:), sseg_e(:,:), sseg_k(:,:), sseg_o(:,:), sseg_x(:,:,:)
     integer, allocatable :: sbo(:)                       ! first band column of each strip
     real(kind=dp), allocatable :: sab(:,:)               ! (lx*lx, sum of strip sizes)
     integer, allocatable :: snode(:)                     ! local node (0-based) of every strip row
     type(c_ptr) :: sab_d = c_null_ptr, sbo_d = c_null_ptr, snode_d = c_null_ptr
     logical :: strips_dev = .false.
     real(kind=dp), allocatable :: diag(:)
     !> 1 at nodes of isotropic elements that lie on no column line (they
     !! get diagonal smoothing); 0 elsewhere (columns own their nodes)
     real(kind=dp), allocatable :: jmask(:)
     integer, allocatable :: dir(:)
     !> raw line blocks of the chosen direction (lx, lx, lx*lx, nelv)
     real(kind=dp), allocatable :: blk(:,:,:,:)
     !> element mode: LU factors and pivots
     real(kind=dp), allocatable :: lu(:,:,:,:)
     integer, allocatable :: ipiv(:,:,:)
     !> sqrt of the partition-of-unity weights, Dirichlet mask
     real(kind=dp), allocatable :: wsq(:)
     real(kind=dp), allocatable :: mask(:)
     !> assembled sum of |couplings| of each node to the nodes off its
     !! stiff line (lateral strength, used by rs_colagg's aggregation)
     real(kind=dp), allocatable :: lat(:)
     !> column mode
     integer :: ncol = 0
     integer :: maxseg = 0
     integer :: maxn = 0
     integer, allocatable :: nseg(:)
     integer, allocatable :: cseg_e(:,:)
     integer, allocatable :: cseg_L(:,:)
     integer, allocatable :: cseg_o(:,:)      ! +1 increasing, -1 decreasing
     !> banded Cholesky factors (lx, sum of column lengths), column c at
     !! nodes cbo(c)+1 .. cbo(c)+len(c) (packed: one long column must not
     !! pad all the others to maxn)
     real(kind=dp), allocatable :: cab(:,:)
     integer, allocatable :: cbo(:)
     type(gs_t), pointer :: gs_h => null()
     !> distributed columns: local chains linked across ranks into global
     !! columns, each solved on one owner rank (see link_columns_global)
     integer :: nglob = 0                      ! global columns (all ranks)
     integer, allocatable :: cg_id(:)          ! local chain -> global column (0: local only)
     logical, allocatable :: cg_rev(:)         ! chain traversed backwards in the global order
     logical, allocatable :: cg_last(:)        ! chain is the last piece of its global column
     integer, allocatable :: cg_off(:)         ! global index of the chain's first segment
     integer, allocatable :: cg_own(:)         ! owner rank of the global column
     integer :: nown = 0                       ! global columns owned by this rank
     integer :: gmaxn = 0
     integer, allocatable :: own_len(:)        ! nodes of each owned column
     !> band Cholesky factors and work of the owned columns, packed by
     !! column (column c at nodes hoff(c)+1 .. hoff(c)+own_len(c)); not
     !! padded to gmaxn, which a single long column makes huge
     real(kind=dp), allocatable :: gcab(:,:)
     real(kind=dp), allocatable :: gy(:)
     integer, allocatable :: hoff(:)
     integer :: nps = 0, npr = 0               ! send / receive pieces
     integer, allocatable :: ps_c(:), ps_pos(:), ps_len(:)
     integer, allocatable :: pr_own(:), pr_q0(:), pr_pos(:), pr_len(:)
     integer, allocatable :: sendcnt(:), senddsp(:), recvcnt(:), recvdsp(:)
     !> Schur-complement exchange of the cut columns (device path): every
     !! piece solves its interior (junction nodes excluded) with the local
     !! kernel and sends two junction scalars and the junction residual to
     !! the owner, who solves a tridiagonal system over the junctions of
     !! each column; the pieces then correct with their two spike vectors
     logical :: schur = .false.
     integer, allocatable :: ps_slot(:), pr_slot(:)        ! piece slot in the per-piece buffers (0-based)
     integer, allocatable :: pscnt(:), psdsp(:), prcnt(:), prdsp(:)
     integer, allocatable :: own_p0(:), own_np(:)          ! first pr piece and piece count of each owned column
     integer, allocatable :: c2p(:)                        ! chain -> send piece (0: local chain)
     integer, allocatable :: pn_off(:), pn_len(:), pn_idx(:) ! interior node lists of the pieces (0-based)
     integer, allocatable :: pc_cl(:), pc_jend(:)          ! piece -> local column (0-based), end-junction node (0-based, -1)
     real(kind=dp), allocatable :: pc_b(:), pc_a(:)        ! (lx-1) junction couplings per piece, flattened
     integer :: npcor = 0
     integer, allocatable :: pcor_idx(:), pcor_p(:)        ! correction entries: node, piece (0-based)
     real(kind=dp), allocatable :: pcor_v(:), pcor_w(:)    ! weighted spike vectors at those nodes
     real(kind=dp), allocatable :: pbuf(:), xjbuf(:)       ! 3 and 2 per sent piece
     real(kind=dp), allocatable :: rsbuf(:), xsbuf(:)      ! 3 and 2 per received piece
     real(kind=dp), allocatable :: sd(:), se(:)            ! tridiagonal (diag, off) per owned column, by junction
     integer, allocatable :: jun_off(:)                    ! junction offset of each owned column
     type(c_ptr) :: pc_cl_d = c_null_ptr, pc_jend_d = c_null_ptr, pc_b_d = c_null_ptr, pc_a_d = c_null_ptr
     type(c_ptr) :: pcor_idx_d = c_null_ptr, pcor_p_d = c_null_ptr, pcor_v_d = c_null_ptr, pcor_w_d = c_null_ptr
     type(c_ptr) :: pbuf_d = c_null_ptr, xjbuf_d = c_null_ptr, ps_slot_d = c_null_ptr
     !> per-call exchange of the junction scalars: point-to-point with the
     !! ranks that share cut columns (snb: owners of this rank's pieces,
     !! rnb: ranks whose pieces this rank owns). schur_dmpi: device buffers
     !! straight into MPI and the junction systems on the device (when
     !! Neko's own gs runs on device MPI, or env RS_LINE_DMPI=1); otherwise
     !! the host stages the buffers and solves the junction systems
     logical :: schur_dmpi = .false., schur_pinned = .false.
     integer :: nsnb = 0, nrnb = 0
     integer, allocatable :: snb(:), rnb(:)
     type(c_ptr) :: schur_reqs = c_null_ptr
     integer, allocatable :: own_p0z(:)                    ! own_p0 - 1 (0-based, for the device)
     real(kind=dp), allocatable :: sdd(:), sxj(:)          ! device scratch of the junction solves
     type(c_ptr) :: xblk_d = c_null_ptr                   ! one block: pbuf_d, xjbuf_d, rsbuf_d, xsbuf_d
     type(c_ptr) :: rsbuf_d = c_null_ptr, xsbuf_d = c_null_ptr, pr_slot_d = c_null_ptr
     type(c_ptr) :: own_p0z_d = c_null_ptr, own_np_d = c_null_ptr, jun_off_d = c_null_ptr
     type(c_ptr) :: sd_d = c_null_ptr, se_d = c_null_ptr, sdd_d = c_null_ptr, sxj_d = c_null_ptr
     real(kind=dp), allocatable :: sbuf(:), rbuf(:)
     !> device path of the distributed columns: one entry per sbuf slot,
     !! its node (0-based) and whether the unpack writes it (global skip
     !! rule); only the packed buffer crosses the host-device boundary
     integer :: ngp = 0
     integer, allocatable :: gp_idx(:), gp_wr(:)
     type(c_ptr) :: gp_idx_d = c_null_ptr, gp_wr_d = c_null_ptr, sbuf_d = c_null_ptr
     !> owner side on the device: rbuf entry -> packed gy index, packed band
     !! factors of the owned columns (layout of cabd), lengths, work
     logical :: gown_dev = .false.
     logical :: sbuf_pinned = .false., rbuf_pinned = .false.
     !> group kernel (eight lanes per column): factors packed per column
     !! without padding in a forward (rows) and a backward (columns) layout
     logical :: group_kernel = .true.
     real(kind=dp), allocatable :: cabf(:), cabb(:)
     integer, allocatable :: cboff(:)            ! row offset of each block of 8 columns (0-based)
     type(c_ptr) :: cabf_d = c_null_ptr, cabb_d = c_null_ptr, cboff_d = c_null_ptr
     !> device copies of the group-kernel factors (cabf/cabb, gcabf/gcabb) in
     !! single precision (option line_factor_precision single); host data
     !! and arithmetic stay double
     logical :: fac_sp = .false.
     integer :: nrb = 0
     integer, allocatable :: rb_map(:), glen(:)
     real(kind=dp), allocatable :: gcabd(:), gyw(:)
     type(c_ptr) :: rb_map_d = c_null_ptr, glen_d = c_null_ptr, gcabd_d = c_null_ptr
     type(c_ptr) :: gyw_d = c_null_ptr, rbuf_d = c_null_ptr
     !> owner side, group kernel: per-column offsets and the two factor layouts
     integer, allocatable :: goff(:)
     real(kind=dp), allocatable :: gcabf(:), gcabb(:)
     type(c_ptr) :: goff_d = c_null_ptr, gcabf_d = c_null_ptr, gcabb_d = c_null_ptr
   contains
     procedure, pass(this) :: init => rs_line_init
     procedure, pass(this) :: init_from_components => rs_line_init_from_components
     procedure, pass(this) :: solve => rs_line_solve
     procedure, pass(this) :: update => rs_line_update
     procedure, pass(this) :: free => rs_line_free
     procedure, pass(this) :: release_host => rs_line_release_host
  end type rs_line_t

contains

  subroutine rs_line_init(this, coef, bclst, json)
    class(rs_line_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json

    call neko_error('rs_line: use init_from_components (needs Ax)')
  end subroutine rs_line_init

  !> Local node index (1..lx**3) of position p on line L along direction d.
  pure function node_index(lx, d, L, p) result(idx)
    integer, intent(in) :: lx, d, L, p
    integer :: idx, a, b
    a = mod(L - 1, lx) + 1
    b = (L - 1) / lx + 1
    select case (d)
    case (1)
       idx = p + (a - 1) * lx + (b - 1) * lx * lx
    case (2)
       idx = a + (p - 1) * lx + (b - 1) * lx * lx
    case default
       idx = a + (b - 1) * lx + (p - 1) * lx * lx
    end select
  end function node_index

  !> Inverse: (line, position) of local node index idx along direction d.
  pure subroutine line_of(lx, d, idx, L, p)
    integer, intent(in) :: lx, d, idx
    integer, intent(out) :: L, p
    integer :: i, j, k
    i = mod(idx - 1, lx) + 1
    j = mod((idx - 1) / lx, lx) + 1
    k = (idx - 1) / (lx * lx) + 1
    select case (d)
    case (1)
       p = i; L = j + (k - 1) * lx
    case (2)
       p = j; L = i + (k - 1) * lx
    case default
       p = k; L = i + (j - 1) * lx
    end select
  end subroutine line_of

  subroutine rs_line_init_from_components(this, ax, coef, gs_h, bc_projector, &
       columns, geometric, force_dir, iso_ratio, global_columns, planes, plane_ratio, factor_single)
    class(rs_line_t), intent(inout), target :: this
    class(ax_t), intent(in) :: ax
    type(coef_t), intent(inout), target :: coef
    type(gs_t), intent(inout), target :: gs_h
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    logical, intent(in) :: columns
    logical, intent(in) :: geometric
    integer, intent(in) :: force_dir
    real(kind=rp), intent(in), optional :: iso_ratio
    logical, intent(in), optional :: global_columns
    logical, intent(in), optional :: planes
    real(kind=rp), intent(in), optional :: plane_ratio
    logical, intent(in), optional :: factor_single
    real(kind=rp), allocatable :: x(:), w(:)
    real(kind=dp), allocatable :: cnt(:), s3(:,:)
    real(kind=dp), allocatable, target :: dl(:), blk3(:,:,:,:,:), lat3(:,:,:)
    real(kind=dp) :: g(3), s(3), aniso_min, aniso_max, stot_e, sl
    type(c_ptr) :: blk3_d, lat3_d, dl_d
    real(kind=dp) :: tc(0:3), tcs(4)
    integer :: lx, nelv, n, e, i0, j0, k0, i, L, p, p0, d, off, idx, info
    integer :: nl, ndir(3), lx3, istat
    character(len=LOG_SIZE) :: log_buf
    character(len=16) :: env
    real(kind=dp) :: tm(0:6), tpl
    type(c_ptr) :: x_d, w_d, xd_d, wd_d

    call this%free()
    tm(0) = MPI_Wtime()
    tcs = 0.0_dp
    lx = coef%Xh%lx
    nelv = coef%msh%nelv
    lx3 = lx**3
    n = lx3 * nelv
    nl = lx * lx
    this%lx = lx
    this%nelv = nelv
    this%n = n
    this%columns = columns
    this%gs_h => gs_h
    this%iso_ratio = 0.0_dp
    if (present(iso_ratio)) this%iso_ratio = iso_ratio
    this%global_columns = .true.
    if (present(global_columns)) this%global_columns = global_columns
    this%planes = .false.
    if (present(planes)) this%planes = planes .and. columns
    this%plane_ratio = 16.0_dp
    if (present(plane_ratio)) this%plane_ratio = plane_ratio
    this%fac_sp = .false.
    if (present(factor_single)) this%fac_sp = factor_single
    allocate(this%dir(nelv), this%blk(lx, lx, nl, nelv), this%iso(nelv), this%diag(n))
    this%iso = .false.
    allocate(this%wsq(n), this%mask(n), x(n), w(n), dl(n))
    allocate(blk3(lx, lx, nl, 3, nelv), lat3(lx3, 3, nelv), this%lat(n))

    ! Dirichlet mask
    do i = 1, n
       this%mask(i) = 1.0_dp
    end do
    call hbcp(bc_projector, this%mask, n)
    if (NEKO_BCKND_DEVICE .eq. 1) then
       ! the linked kernels must match Neko's backend (the stubs return 0)
       block
         integer(c_int) :: kb
         kb = rs_line_cuda_backend()
         if ((NEKO_BCKND_CUDA .eq. 1 .and. kb .ne. 1) .or. &
              (NEKO_BCKND_HIP .eq. 1 .and. kb .ne. 2) .or. NEKO_BCKND_OPENCL .eq. 1) then
            call neko_error('rs_line: the linked GPU kernels do not match the Neko backend ' // &
                 '(rs_line_cuda.cu for CUDA, rs_line_hip.cpp for HIP, no OpenCL)')
         end if
       end block
    end if
    if (NEKO_BCKND_DEVICE .eq. 1) then
       ! the comb responses are reduced on the device into blk3, lat3 and
       ! dl (device copies), fetched once after the loop
       x_d = c_null_ptr; w_d = c_null_ptr; blk3_d = c_null_ptr; lat3_d = c_null_ptr
       dl_d = c_null_ptr
       call device_map(x, x_d, n)
       call device_map(w, w_d, n)
       ! the comb kernels work in double: own buffers when rp is not dp
       xd_d = x_d; wd_d = w_d
       if (rp .ne. dp) then
          call device_alloc(xd_d, int(n, c_size_t) * 8_c_size_t)
          call device_alloc(wd_d, int(n, c_size_t) * 8_c_size_t)
       end if
       ! blk3 has 3 lx**4 entries per element: more than 2^31 above ~175 000
       ! elements per rank at lx = 8, beyond Neko's 32-bit device_map count
       call device_alloc(blk3_d, int(size(blk3, kind=i8), c_size_t) * int(storage_size(1.0_dp) / 8, c_size_t))
       call device_map(lat3, lat3_d, size(lat3))
       call device_map(dl, dl_d, n)
    end if

    ! comb vectors: local couplings along all three directions, local diagonal
    blk3 = 0.0_dp
    do k0 = 1, lx
       do j0 = 1, lx
          do i0 = 1, lx
             off = i0 + (j0 - 1) * lx + (k0 - 1) * nl
             if (NEKO_BCKND_DEVICE .ne. 1) then
                do i = 1, n
                   x(i) = 0.0_dp
                end do
                do e = 1, nelv
                   x(off + (e - 1) * lx3) = 1.0_dp
                end do
             end if
             if (NEKO_BCKND_DEVICE .eq. 1) then
                ! comb on the device, extraction by a kernel: only 3 lx + 1
                ! values per element come back
                tc(0) = MPI_Wtime()
                call rs_line_cuda_comb_set(xd_d, int(lx3, c_int), int(off - 1, c_int), int(n, c_int))
                if (rp .ne. dp) call rs_line_cuda_d2r(x_d, xd_d, int(n, c_int))
                call rs_line_cuda_sync()
                tc(1) = MPI_Wtime()
                call ax%compute(w, x, coef, coef%msh, coef%Xh)
                call device_sync()
                tc(2) = MPI_Wtime()
                if (rp .ne. dp) call rs_line_cuda_r2d(wd_d, w_d, int(n, c_int))
                call rs_line_cuda_comb_extract(wd_d, blk3_d, lat3_d, dl_d, int(lx, c_int), &
                     int(off - 1, c_int), int(nelv, c_int))
                call rs_line_cuda_sync()
                tc(3) = MPI_Wtime()
                tcs(1) = tcs(1) + tc(1) - tc(0); tcs(2) = tcs(2) + tc(2) - tc(1)
                tcs(3) = tcs(3) + tc(3) - tc(2)
             else
             call ax%compute(w, x, coef, coef%msh, coef%Xh)
             do e = 1, nelv
                stot_e = 0.0_dp
                do idx = 1, lx3
                   stot_e = stot_e + abs(w(idx + (e - 1) * lx3))
                end do
                do d = 1, 3
                   call line_of(lx, d, off, L, p0)
                   sl = 0.0_dp
                   do p = 1, lx
                      idx = node_index(lx, d, L, p) + (e - 1) * lx3
                      blk3(p, p0, L, d, e) = w(idx)
                      sl = sl + abs(w(idx))
                   end do
                   ! |couplings| of node off to the nodes of e off its d-line
                   lat3(off, d, e) = stot_e - sl
                end do
                dl(off + (e - 1) * lx3) = w(off + (e - 1) * lx3)
             end do
             end if
          end do
       end do
    end do
    if (NEKO_BCKND_DEVICE .eq. 1) then
       tc(0) = MPI_Wtime()
       call rs_line_cuda_copy(c_loc(blk3), blk3_d, int(size(blk3, kind=i8), c_size_t) * &
            int(storage_size(1.0_dp) / 8, c_size_t), 2_c_int)
       call device_memcpy(lat3, lat3_d, size(lat3), DEVICE_TO_HOST, sync = .true.)
       call device_memcpy(dl, dl_d, n, DEVICE_TO_HOST, sync = .true.)
       tcs(4) = MPI_Wtime() - tc(0)
       call device_deassociate(x); call device_free(x_d)
       call device_deassociate(w); call device_free(w_d)
       if (rp .ne. dp) then
          call device_free(xd_d); call device_free(wd_d)
       end if
       call device_free(blk3_d)
       call device_deassociate(lat3); call device_free(lat3_d)
       call device_deassociate(dl); call device_free(dl_d)
    end if
    call hgs(gs_h, dl, n, GS_OP_ADD)
    this%diag = dl
    tm(1) = MPI_Wtime()
    call rs_mem_report('rs_line lx '//achar(48 + lx)//' after comb')

    ! direction per element
    allocate(s3(3, nelv))
    ndir = 0
    aniso_min = huge(1.0_dp)
    aniso_max = 0.0_dp
    !$omp parallel do private(e, g, s, d, L, p0, p, idx) reduction(+:ndir) &
    !$omp reduction(min:aniso_min) reduction(max:aniso_max) schedule(static)
    do e = 1, nelv
       if (geometric) then
          g(1) = sum(coef%G11(:,:,:,e))
          g(2) = sum(coef%G22(:,:,:,e))
          g(3) = sum(coef%G33(:,:,:,e))
       else
          ! total off-diagonal line coupling per direction
          do d = 1, 3
             s(d) = 0.0_dp
             do L = 1, nl
                do p0 = 1, lx
                   do p = 1, lx
                      if (p .ne. p0) s(d) = s(d) + abs(blk3(p, p0, L, d, e))
                   end do
                end do
             end do
          end do
          g = s
       end if
       if (geometric) s = g
       s3(:, e) = s
       this%dir(e) = maxloc(g, 1)
       if (force_dir .ge. 1 .and. force_dir .le. 3) this%dir(e) = force_dir
       ndir(this%dir(e)) = ndir(this%dir(e)) + 1
       g(this%dir(e)) = -1.0_dp
       aniso_min = min(aniso_min, maxval(s) / max(maxval(g), tiny(1.0_dp)))
       aniso_max = max(aniso_max, maxval(s) / max(maxval(g), tiny(1.0_dp)))
       if (this%iso_ratio .gt. 0.0_dp) then
          this%iso(e) = maxval(s) .lt. this%iso_ratio * max(maxval(g), tiny(1.0_dp))
       end if
       this%blk(:,:,:,e) = blk3(:,:,:,this%dir(e),e)
       do idx = 1, lx3
          this%lat(idx + (e - 1) * lx3) = lat3(idx, this%dir(e), e)
       end do
    end do
    !$omp end parallel do
    call hgs(gs_h, this%lat, n, GS_OP_ADD)
    deallocate(lat3)
    if (this%planes) call classify_planes(this, s3)
    deallocate(s3)

    ! Assemble the line blocks: the off-diagonal coupling between two line
    ! nodes is the sum over all elements that contain both (lines on
    ! lateral faces and edges are shared), the diagonal is the assembled
    ! diagonal; then decouple Dirichlet nodes.
    tm(2) = MPI_Wtime()
    call rs_mem_report('rs_line lx '//achar(48 + lx)//' after directions')
    ! RS_LINE_ASSEMBLE = G: gather-scatter by reference position (needs
    ! consistent element frames); L: rank-local pairs; default: by node ids
    call get_environment_variable('RS_LINE_ASSEMBLE', env, status = istat)
    if (istat .eq. 0 .and. env(1:1) .eq. 'G') then
       call assemble_line_blocks_gs(this, coef, gs_h, blk3)
    else if (istat .eq. 0 .and. env(1:1) .eq. 'L') then
       call assemble_line_blocks(this, coef, blk3)
    else
       call assemble_line_blocks_id(this, coef, blk3)
    end if
    deallocate(blk3)
    tm(3) = MPI_Wtime()
    call rs_mem_report('rs_line lx '//achar(48 + lx)//' after assembly')
    !$omp parallel do private(e, d, L, p, idx) schedule(static)
    do e = 1, nelv
       d = this%dir(e)
       do L = 1, nl
          do p = 1, lx
             idx = node_index(lx, d, L, p) + (e - 1) * lx3
             this%blk(p, p, L, e) = dl(idx)
          end do
          do p = 1, lx
             idx = node_index(lx, d, L, p) + (e - 1) * lx3
             if (this%mask(idx) .lt. 0.5_dp) then
                this%blk(p, :, L, e) = 0.0_dp
                this%blk(:, p, L, e) = 0.0_dp
                this%blk(p, p, L, e) = 1.0_dp
             end if
          end do
       end do
    end do
    !$omp end parallel do

    if (this%npel .gt. 0) call comb_planes(this, ax, coef, x, w)
    if (this%columns) then
       call build_columns(this, coef)
       if (this%planes) then
          tpl = MPI_Wtime()
          if (this%npel .gt. 0) call build_strips(this, coef)
          ! collective: every rank, also those without plane elements
          call strip_stats(this, MPI_Wtime() - tpl)
       end if
    else
       allocate(this%lu(lx, lx, nl, nelv), this%ipiv(lx, nl, nelv))
       this%lu = this%blk
       do e = 1, nelv
          do L = 1, nl
             call dgetrf(lx, lx, this%lu(:,:,L,e), lx, this%ipiv(:,L,e), info)
             if (info .ne. 0) call neko_error('rs_line: dgetrf failed')
          end do
       end do
    end if

    tm(4) = MPI_Wtime()
    call rs_mem_report('rs_line lx '//achar(48 + lx)//' after columns+link')
    ! partition-of-unity weights: 1 / number of (column) lines holding a node
    allocate(cnt(n))
    do i = 1, n
       cnt(i) = 0.0_dp
    end do
    allocate(this%jmask(n))
    this%jmask = 0.0_dp
    if (this%columns) then
       call column_mark(this, cnt)
       call strip_mark(this, cnt)
       if (count(this%iso) .gt. 0 .or. pe_size .gt. 1) then
          ! nodes on no column line (after summing the copies) belong to
          ! isotropic elements only: diagonal smoothing there
          call hgs(gs_h, cnt, n, GS_OP_ADD)
          do e = 1, nelv
             if (.not. this%iso(e)) cycle
             do i = 1, lx3
                idx = i + (e - 1) * lx3
                if (cnt(idx) .eq. 0.0_dp) this%jmask(idx) = 1.0_dp
             end do
          end do
          ! rebuild the count: column lines plus one per diagonal copy
          do i = 1, n
             cnt(i) = 0.0_dp
          end do
          call column_mark(this, cnt)
          call strip_mark(this, cnt)
          do i = 1, n
             cnt(i) = cnt(i) + this%jmask(i)
          end do
       end if
    else
       do i = 1, n
          cnt(i) = 1.0_dp
       end do
    end if
    call hgs(gs_h, cnt, n, GS_OP_ADD)
    do i = 1, n
       this%wsq(i) = 1.0_dp / sqrt(max(cnt(i), 1.0_dp))
    end do
    deallocate(cnt)
    ! experiment (env RS_LINE_SPFACT): factors computed in double, then
    ! rounded to single precision, i.e. what single-precision storage of
    ! the line / strip factors would give (solves still in double)
    call get_environment_variable('RS_LINE_SPFACT', env, status = istat)
    if (istat .eq. 0) then
       if (allocated(this%cab)) this%cab = real(real(this%cab, sp), dp)
       if (allocated(this%gcab)) this%gcab = real(real(this%gcab, sp), dp)
       if (allocated(this%sab)) this%sab = real(real(this%sab, sp), dp)
       if (allocated(this%lu)) this%lu = real(real(this%lu, sp), dp)
       if (pe_rank .eq. 0) write(*, '(A)') 'RS_LINE factors rounded to single precision (RS_LINE_SPFACT)'
    end if
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call rs_line_device_setup(this)
       call get_environment_variable('RS_LINE_CHECK', env, status = istat)
       this%check_device = (istat .eq. 0)
       call get_environment_variable('RS_TIMERS', env, status = istat)
       this%timers = (istat .eq. 0)
       this%tsol = 0.0_dp; this%nsol = 0
    end if

    tm(5) = MPI_Wtime()
    call rs_mem_report('rs_line lx '//achar(48 + lx)//' after end')
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,5F8.3)') 'RS_LINE setup lx ', lx, ' comb/dir/assembly/columns+link/weights ', &
            tm(1) - tm(0), tm(2) - tm(1), tm(3) - tm(2), tm(4) - tm(3), tm(5) - tm(4)
       if (NEKO_BCKND_DEVICE .eq. 1) then
          write(*, '(A,4F8.3)') 'RS_LINE comb (device) set/ax/extract/final copies ', tcs(1), tcs(2), &
               tcs(3), tcs(4)
       end if
       write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A,L1,A,L1)') 'rs_line: lx ', lx, &
            ' dir(r,s,t) ', ndir(1), ' ', ndir(2), ' ', ndir(3), &
            ' col ', this%columns, ' geo ', geometric
       call neko_log%message(log_buf)
       if (.not. geometric) then
          write(log_buf, '(A,ES9.2,A,ES9.2)') &
               'rs_line: aniso ratio min ', aniso_min, ' max ', aniso_max
          call neko_log%message(log_buf)
       end if
       if (this%columns) then
          write(log_buf, '(A,I0,A,I0,A,I0,A,I0)') 'rs_line: columns ', &
               this%ncol, ' maxseg ', this%maxseg, ' maxlen ', this%maxn, &
               ' iso elements(rank0) ', count(this%iso)
          call neko_log%message(log_buf)
          if (this%planes) then
             write(log_buf, '(A,I0,A,I0,A,I0,A,ES9.2)') 'rs_line: plane el. ', &
                  this%npel, ' strips ', this%nstrip, ' maxel ', this%smaxseg, &
                  ' ratio ', this%plane_ratio
             call neko_log%message(log_buf)
          end if
       end if
    end if
    deallocate(x, w, dl)
  end subroutine rs_line_init_from_components

  !> Add the couplings of the other elements sharing each pair of line
  !! nodes (off-diagonals only; the diagonal is set afterwards).
  subroutine assemble_line_blocks(this, coef, blk3)
    class(rs_line_t), intent(inout) :: this
    type(coef_t), intent(in), target :: coef
    real(kind=dp), intent(in) :: blk3(:,:,:,:,:)
    integer, parameter :: MAXSH = 8
    integer :: lx, nelv, nl, lx3, e, L, p, pb, idx, idxb, d, k, e2, ia, ib
    integer :: i1, j1, k1, i2, j2, k2, d2, L2, pa2, pb2, ndiff
    integer(kind=i8) :: id, idb, maxid
    integer, allocatable :: nsh(:), she(:,:), shi(:,:)

    lx = this%lx; nelv = this%nelv; nl = lx * lx; lx3 = lx**3
    maxid = maxval(coef%dof%dof)
    allocate(nsh(maxid + 1), she(MAXSH, maxid + 1), shi(MAXSH, maxid + 1))
    nsh = 0
    do e = 1, nelv
       do idx = 1, lx3
          ! interior (element-private) nodes carry id 0; only the others
          ! have unique ids
          id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
          if (id .eq. 0) cycle
          if (nsh(id + 1) .lt. MAXSH) then
             nsh(id + 1) = nsh(id + 1) + 1
             she(nsh(id + 1), id + 1) = e
             shi(nsh(id + 1), id + 1) = idx
          end if
       end do
    end do
    do e = 1, nelv
       d = this%dir(e)
       do L = 1, nl
          do p = 1, lx
             idx = node_index(lx, d, L, p)
             id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
             if (id .eq. 0) cycle
             if (nsh(id + 1) .le. 1) cycle
             do pb = 1, lx
                if (pb .eq. p) cycle
                idxb = node_index(lx, d, L, pb)
                idb = coef%dof%dof(mod(idxb-1,lx)+1, mod((idxb-1)/lx,lx)+1, (idxb-1)/nl+1, e)
                if (idb .eq. 0) cycle
                ! other elements containing both nodes
                do k = 1, nsh(id + 1)
                   e2 = she(k, id + 1)
                   if (e2 .eq. e) cycle
                   ia = shi(k, id + 1)
                   ib = 0
                   do i2 = 1, nsh(idb + 1)
                      if (she(i2, idb + 1) .eq. e2) ib = shi(i2, idb + 1)
                   end do
                   if (ib .eq. 0) cycle
                   ! direction along which ia and ib differ in e2
                   i1 = mod(ia-1,lx)+1; j1 = mod((ia-1)/lx,lx)+1; k1 = (ia-1)/nl+1
                   i2 = mod(ib-1,lx)+1; j2 = mod((ib-1)/lx,lx)+1; k2 = (ib-1)/nl+1
                   ndiff = 0; d2 = 0
                   if (i1 .ne. i2) then
                      ndiff = ndiff + 1; d2 = 1
                   end if
                   if (j1 .ne. j2) then
                      ndiff = ndiff + 1; d2 = 2
                   end if
                   if (k1 .ne. k2) then
                      ndiff = ndiff + 1; d2 = 3
                   end if
                   if (ndiff .ne. 1) cycle
                   call line_of(lx, d2, ia, L2, pa2)
                   call line_of(lx, d2, ib, L2, pb2)
                   this%blk(p, pb, L, e) = this%blk(p, pb, L, e) + blk3(pa2, pb2, L2, d2, e2)
                end do
             end do
          end do
       end do
    end do
    deallocate(nsh, she, shi)
  end subroutine assemble_line_blocks

  !> Assemble the line couplings by node ids, independently of the element
  !! frames: every element line on a face or edge (a segment) is keyed by
  !! the ids of its two end nodes; the blocks of all elements holding the
  !! same segment are summed in the canonical (id-ordered) orientation,
  !! locally by sorting, and across ranks by exchanging the segments whose
  !! nodes are all MPI-shared (Allgatherv: prototype; a neighbour exchange
  !! is the scalable form). Interior lines belong to one element only.
  subroutine assemble_line_blocks_id(this, coef, blk3)
    class(rs_line_t), intent(inout) :: this
    type(coef_t), intent(in), target :: coef
    real(kind=dp), intent(in) :: blk3(:,:,:,:,:)
    integer :: lx, nelv, nl, lx3, d, e, L, p, p2, idx, i, k, g, ierr, nent, ngrp
    integer :: nb, ntot, lo, hi, mid, c
    integer(kind=i8) :: ida, idb
    integer(kind=i8), allocatable :: key(:,:), gkey(:,:), skey(:,:), rkey(:,:)
    real(kind=dp), allocatable :: blkc(:,:,:), sumb(:,:,:), rblk(:,:,:), stri(:,:), rtri(:,:)
    integer :: ntri
    integer, allocatable :: ent_e(:), ent_L(:), ent_d(:), perm(:), grp(:), counts(:), displs(:)
    logical, allocatable :: flip(:), bnd(:)
    logical :: shared_line, any_shared

    lx = this%lx; nelv = this%nelv; nl = lx * lx; lx3 = lx**3
    allocate(counts(pe_size), displs(pe_size))
    ! the segments of all three reference directions in one list: a
    ! neighbour may hold the same segment along another reference direction
    allocate(key(2, 3 * nl * nelv), blkc(lx, lx, 3 * nl * nelv), ent_e(3 * nl * nelv), &
         ent_L(3 * nl * nelv), ent_d(3 * nl * nelv), flip(3 * nl * nelv), bnd(3 * nl * nelv))
    ! entries: one slot per (d, e, L); the interior lines stay unused
    block
      logical, allocatable :: used(:)
      integer, allocatable :: slot(:)
      integer :: ne, nslot, ii
      nslot = 3 * nelv * nl
      allocate(used(nslot))
      !$omp parallel do private(d, e, L, p, idx, any_shared, shared_line, ida, idb, ii, p2) schedule(static)
      do ii = 1, nslot
         d = (ii - 1) / (nelv * nl) + 1
         e = mod((ii - 1) / nl, nelv) + 1
         L = mod(ii - 1, nl) + 1
         any_shared = lx .eq. 2
         shared_line = .true.
         do p = 1, lx
            idx = node_index(lx, d, L, p)
            if (p .gt. 1 .and. p .lt. lx) then
               if (nid(idx, e) .ne. 0_i8) any_shared = .true.
            end if
            if (.not. coef%dof%shared_dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, &
                 (idx-1)/nl+1, e)) shared_line = .false.
         end do
         used(ii) = any_shared
         if (.not. any_shared) cycle
         ida = nid(node_index(lx, d, L, 1), e)
         idb = nid(node_index(lx, d, L, lx), e)
         ent_e(ii) = e; ent_L(ii) = L; ent_d(ii) = d
         flip(ii) = ida .gt. idb
         key(1, ii) = min(ida, idb); key(2, ii) = max(ida, idb)
         bnd(ii) = shared_line
         do p2 = 1, lx
            do p = 1, lx
               if (flip(ii)) then
                  blkc(p, p2, ii) = blk3(lx + 1 - p, lx + 1 - p2, L, d, e)
               else
                  blkc(p, p2, ii) = blk3(p, p2, L, d, e)
               end if
            end do
         end do
      end do
      !$omp end parallel do
      ! compact the used slots into fresh arrays (a parallel in-place
      ! compaction would race: slot(i) >= i)
      allocate(slot(nslot))
      nent = 0
      do ii = 1, nslot
         if (used(ii)) then
            nent = nent + 1; slot(nent) = ii
         end if
      end do
      block
        integer(kind=i8), allocatable :: key2(:,:)
        real(kind=dp), allocatable :: blkc2(:,:,:)
        integer, allocatable :: ent_e2(:), ent_L2(:), ent_d2(:)
        logical, allocatable :: flip2(:), bnd2(:)
        allocate(key2(2, max(nent, 1)), blkc2(lx, lx, max(nent, 1)), ent_e2(max(nent, 1)), &
             ent_L2(max(nent, 1)), ent_d2(max(nent, 1)), flip2(max(nent, 1)), bnd2(max(nent, 1)))
        !$omp parallel do private(i, ii) schedule(static)
        do i = 1, nent
           ii = slot(i)
           ent_e2(i) = ent_e(ii); ent_L2(i) = ent_L(ii); ent_d2(i) = ent_d(ii)
           flip2(i) = flip(ii); key2(:, i) = key(:, ii); bnd2(i) = bnd(ii)
           blkc2(:, :, i) = blkc(:, :, ii)
        end do
        !$omp end parallel do
        call move_alloc(key2, key); call move_alloc(blkc2, blkc)
        call move_alloc(ent_e2, ent_e); call move_alloc(ent_L2, ent_L); call move_alloc(ent_d2, ent_d)
        call move_alloc(flip2, flip); call move_alloc(bnd2, bnd)
      end block
      deallocate(used, slot)
    end block
    block
       ! local groups of equal keys: radix sort on the exact packed key
       ! key1 * 2^32 + key2 (ids < 2^32 checked), so that the groups are
       ! in lexicographic order and the binary search below is valid
       integer(kind=i8), allocatable :: hk(:)
       allocate(perm(max(nent, 1)), grp(max(nent, 1)), sumb(lx, lx, max(nent, 1)), &
            gkey(2, max(nent, 1)), hk(max(nent, 1)))
       if (maxval(coef%dof%dof) .ge. 2_i8**32) call neko_error('rs_line: dof ids >= 2^32')
       do i = 1, nent
          hk(i) = ishft(key(1, i), 32) + key(2, i)
       end do
       if (nent .gt. 0) call sort_ki(nent, hk, perm)
       deallocate(hk)
       ngrp = 0
       do i = 1, nent
          k = perm(i)
          if (i .gt. 1) then
             if (key(1, k) .eq. key(1, perm(i-1)) .and. key(2, k) .eq. key(2, perm(i-1))) then
                sumb(:, :, ngrp) = sumb(:, :, ngrp) + blkc(:, :, k)
                grp(k) = ngrp
                cycle
             end if
          end if
          ngrp = ngrp + 1
          sumb(:, :, ngrp) = blkc(:, :, k)
          gkey(:, ngrp) = key(:, k)
          grp(k) = ngrp
       end do
       ! cross-rank: exchange the segments whose nodes are all MPI-shared
       nb = count(bnd(1:nent))
       ntri = (lx * (lx + 1)) / 2
       allocate(skey(3, max(nb, 1)), stri(ntri, max(nb, 1)))
       nb = 0
       do i = 1, nent
          if (.not. bnd(i)) cycle
          nb = nb + 1
          skey(1, nb) = key(1, i); skey(2, nb) = key(2, i); skey(3, nb) = int(pe_rank, i8)
          ! lower triangle of the symmetric block, column by column
          k = 0
          do p2 = 1, lx
             do p = p2, lx
                k = k + 1
                stri(k, nb) = blkc(p, p2, i)
             end do
          end do
       end do
       call MPI_Allgather(nb, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, NEKO_COMM, ierr)
       ntot = sum(counts)
       allocate(rkey(3, max(ntot, 1)), rtri(ntri, max(ntot, 1)), rblk(lx, lx, 1))
       displs(1) = 0
       do i = 2, pe_size
          displs(i) = displs(i-1) + counts(i-1)
       end do
       call MPI_Allgatherv(skey, 3 * nb, MPI_INTEGER8, rkey, 3 * counts, 3 * displs, &
            MPI_INTEGER8, NEKO_COMM, ierr)
       call MPI_Allgatherv(stri, ntri * nb, MPI_DOUBLE_PRECISION, rtri, &
            ntri * counts, ntri * displs, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
       do i = 1, ntot
          if (rkey(3, i) .eq. int(pe_rank, i8)) cycle
          ! binary search of the key among my groups
          lo = 1; hi = ngrp; g = 0
          do while (lo .le. hi)
             mid = (lo + hi) / 2
             if (gkey(1, mid) .lt. rkey(1, i)) then
                c = -1
             else if (gkey(1, mid) .gt. rkey(1, i)) then
                c = 1
             else if (gkey(2, mid) .lt. rkey(2, i)) then
                c = -1
             else if (gkey(2, mid) .gt. rkey(2, i)) then
                c = 1
             else
                c = 0
             end if
             if (c .eq. 0) then
                g = mid; exit
             else if (c .lt. 0) then
                lo = mid + 1
             else
                hi = mid - 1
             end if
          end do
          if (g .gt. 0) then
             k = 0
             do p2 = 1, lx
                do p = p2, lx
                   k = k + 1
                   sumb(p, p2, g) = sumb(p, p2, g) + rtri(k, i)
                   if (p .ne. p2) sumb(p2, p, g) = sumb(p2, p, g) + rtri(k, i)
                end do
             end do
          end if
       end do
       ! write back to the elements whose stiff direction is the entry's
       !$omp parallel do private(i, e, L, g, p, p2) schedule(static)
       do i = 1, nent
          e = ent_e(i)
          if (this%dir(e) .ne. ent_d(i)) cycle
          L = ent_L(i); g = grp(i)
          do p2 = 1, lx
             do p = 1, lx
                if (flip(i)) then
                   this%blk(p, p2, L, e) = sumb(lx + 1 - p, lx + 1 - p2, g)
                else
                   this%blk(p, p2, L, e) = sumb(p, p2, g)
                end if
             end do
          end do
       end do
       !$omp end parallel do
       deallocate(perm, grp, sumb, gkey, skey, stri, rkey, rtri, rblk)
    end block
    deallocate(key, blkc, ent_e, ent_L, ent_d, flip, bnd, counts, displs)

  contains

    function nid(idx, e) result(id)
      integer, intent(in) :: idx, e
      integer(kind=i8) :: id
      id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
    end function nid

  end subroutine assemble_line_blocks_id

  !> Assemble the line couplings over all elements and ranks with
  !! gather-scatter (consistent element frames assumed).
  subroutine assemble_line_blocks_gs(this, coef, gs_h, blk3)
    class(rs_line_t), intent(inout) :: this
    type(coef_t), intent(in), target :: coef
    type(gs_t), intent(inout) :: gs_h
    real(kind=dp), intent(in) :: blk3(:,:,:,:,:)
    real(kind=dp), allocatable :: q(:)
    integer :: lx, nelv, nl, lx3, n, e, d, L, p, pidx, idx

    lx = this%lx; nelv = this%nelv; nl = lx * lx; lx3 = lx**3; n = lx3 * nelv
    allocate(q(n))
    do d = 1, 3
       do p = 1, lx
          do pidx = 1, lx
             if (pidx .eq. p) cycle
             ! q(idx) = coupling between node idx, if it sits at position
             ! pidx of its d-line, and the p-th node of that line; nodes at
             ! other positions carry 0. The gather-scatter sum then runs
             ! over the elements that share the whole line (lateral
             ! neighbours) and excludes the stacked neighbours, which share
             ! the end node but not the line.
             do idx = 1, n
                q(idx) = 0.0_dp
             end do
             do e = 1, nelv
                do L = 1, nl
                   idx = node_index(lx, d, L, pidx) + (e - 1) * lx3
                   q(idx) = blk3(p, pidx, L, d, e)
                end do
             end do
             call hgs(gs_h, q, n, GS_OP_ADD)
             do e = 1, nelv
                if (this%dir(e) .ne. d) cycle
                do L = 1, nl
                   idx = node_index(lx, d, L, pidx) + (e - 1) * lx3
                   this%blk(p, pidx, L, e) = q(idx)
                end do
             end do
          end do
       end do
    end do
    deallocate(q)
  end subroutine assemble_line_blocks_gs

  !> Chain element lines across faces normal to the stiff direction.
  subroutine build_columns(this, coef)
    class(rs_line_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    integer :: lx, nelv, nl, lx3, e, L, p, idx, d, i, f, m, q, qp, info
    integer :: e2, idx2, L2, p2, d2, o, c, nseg, lenc, ldab, kl, ku
    integer(kind=i8) :: id, maxid
    integer, allocatable :: first_e(:), first_idx(:), second_e(:), second_idx(:)
    integer, allocatable :: nb_e(:,:), nb_o(:,:), nb_L(:,:,:), col(:,:)
    logical, allocatable :: visited(:,:)
    integer :: pl, pl2, idxf
    integer :: a, b, ii, jj, kk, ext(3), nextr
    real(kind=dp) :: tb(0:3)
    logical, allocatable :: sel(:)

    lx = this%lx; nelv = this%nelv; nl = lx * lx; lx3 = lx**3
    tb(0) = MPI_Wtime()
    allocate(sel(nelv))
    sel = .not. this%iso
    if (allocated(this%pel)) sel = sel .and. .not. this%pel
    call face_neighbours(this, coef, sel, nb_e, nb_o, nb_L)
    deallocate(sel)
    tb(1) = MPI_Wtime()

    ! walk the columns: a column starts at an element line whose low face
    ! (f = 1) has no chaining neighbour, and proceeds through high faces.
    allocate(col(nl, nelv), visited(nl, nelv))
    col = 0; visited = .false.
    ! isotropic and plane elements have no column lines
    do e = 1, nelv
       if (this%iso(e)) visited(:, e) = .true.
       if (allocated(this%pel)) then
          if (this%pel(e)) visited(:, e) = .true.
       end if
    end do
    ! first pass: count columns and segments. Columns start at a line
    ! whose low face has no chaining neighbour (walked upwards) or, with
    ! inconsistent frames, whose high face has none (walked downwards).
    this%ncol = 0; this%maxseg = 1
    do e = 1, nelv
       do L = 1, nl
          if (visited(L, e)) cycle
          if (nb_e(1, e) .ne. 0) cycle
          call walk(this, nb_e, nb_o, nb_L, visited, col, e, L, nl, nelv, .false., 1, c)
       end do
    end do
    do e = 1, nelv
       do L = 1, nl
          if (visited(L, e)) cycle
          if (nb_e(2, e) .ne. 0) cycle
          call walk(this, nb_e, nb_o, nb_L, visited, col, e, L, nl, nelv, .false., -1, c)
       end do
    end do
    ! leftovers (loops): single-segment columns
    do e = 1, nelv
       do L = 1, nl
          if (.not. visited(L, e)) then
             this%ncol = this%ncol + 1
             visited(L, e) = .true.
             col(L, e) = this%ncol
          end if
       end do
    end do
    allocate(this%nseg(this%ncol))
    allocate(this%cseg_e(this%maxseg, this%ncol), this%cseg_L(this%maxseg, this%ncol), &
         this%cseg_o(this%maxseg, this%ncol))
    this%nseg = 0
    ! second pass: fill the segment lists (same traversal order)
    visited = .false.
    do e = 1, nelv
       if (this%iso(e)) visited(:, e) = .true.
       if (allocated(this%pel)) then
          if (this%pel(e)) visited(:, e) = .true.
       end if
    end do
    this%ncol = 0
    do e = 1, nelv
       do L = 1, nl
          if (visited(L, e)) cycle
          if (nb_e(1, e) .ne. 0) cycle
          call walk(this, nb_e, nb_o, nb_L, visited, col, e, L, nl, nelv, .true., 1, c)
       end do
    end do
    do e = 1, nelv
       do L = 1, nl
          if (visited(L, e)) cycle
          if (nb_e(2, e) .ne. 0) cycle
          call walk(this, nb_e, nb_o, nb_L, visited, col, e, L, nl, nelv, .true., -1, c)
       end do
    end do
    do e = 1, nelv
       do L = 1, nl
          if (.not. visited(L, e)) then
             this%ncol = this%ncol + 1
             visited(L, e) = .true.
             this%nseg(this%ncol) = 1
             this%cseg_e(1, this%ncol) = e
             this%cseg_L(1, this%ncol) = L
             this%cseg_o(1, this%ncol) = 1
          end if
       end do
    end do
    deallocate(nb_e, nb_o, nb_L, col, visited)
    tb(2) = MPI_Wtime()

    ! banded matrices
    ! symmetric positive definite band (principal submatrix of A): lower
    ! Cholesky storage ab(1 + q - qp, qp), ldab = kd + 1 (dpbtrf/dpbtrs)
    kl = lx - 1; ku = lx - 1; ldab = kl + 1
    this%maxn = this%maxseg * (lx - 1) + 1
    allocate(this%cbo(max(this%ncol, 1)))
    this%cbo(1) = 0
    do c = 2, this%ncol
       this%cbo(c) = this%cbo(c - 1) + this%nseg(c - 1) * (lx - 1) + 1
    end do
    lenc = 0
    if (this%ncol .gt. 0) lenc = this%cbo(this%ncol) + this%nseg(this%ncol) * (lx - 1) + 1
    allocate(this%cab(ldab, max(lenc, 1)))
    !$omp parallel do private(c) schedule(static)
    do c = 1, this%ncol
       this%cab(:, this%cbo(c) + 1 : this%cbo(c) + this%nseg(c) * (lx - 1) + 1) = 0.0_dp
    end do
    !$omp end parallel do
    !$omp parallel do private(c, nseg, lenc, m, e, L, o, p, q, i, qp, pl, pl2, info) schedule(dynamic, 64)
    do c = 1, this%ncol
       nseg = this%nseg(c)
       lenc = nseg * (lx - 1) + 1
       do m = 1, nseg
          e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
          do p = 1, lx
             q = (m - 1) * (lx - 1) + p
             do i = 1, lx
                qp = (m - 1) * (lx - 1) + i
                if (o .eq. 1) then
                   pl = p; pl2 = i
                else
                   pl = lx + 1 - p; pl2 = lx + 1 - i
                end if
                if (q .lt. qp) cycle
                if (p .eq. i) then
                   ! diagonal: assembled, set once
                   this%cab(1 + q - qp, this%cbo(c) + qp) = this%blk(pl, pl2, L, e)
                else
                   this%cab(1 + q - qp, this%cbo(c) + qp) = &
                        this%cab(1 + q - qp, this%cbo(c) + qp) + this%blk(pl, pl2, L, e)
                end if
             end do
          end do
       end do
       call dpbtrf('L', lenc, kl, this%cab(1, this%cbo(c) + 1), ldab, info)
       if (info .ne. 0) then
          !$omp critical
          call neko_error('rs_line: dpbtrf failed (column not SPD)')
          !$omp end critical
       end if
    end do
    !$omp end parallel do
    tb(3) = MPI_Wtime()
    if (pe_rank .eq. 0) write(*, '(A,3F8.3)') 'RS_LINE columns: tables+faces/walk/band+factor ', &
         tb(1) - tb(0), tb(2) - tb(1), tb(3) - tb(2)

    allocate(this%cg_id(this%ncol), this%cg_rev(this%ncol), this%cg_off(this%ncol), &
         this%cg_own(this%ncol), this%cg_last(this%ncol))
    this%cg_id = 0; this%cg_rev = .false.; this%cg_off = 1; this%cg_own = -1; this%cg_last = .true.
    block
      real(kind=dp) :: t0
      t0 = MPI_Wtime()
      if (pe_size .gt. 1 .and. this%global_columns) call link_columns_global(this, coef)
      if (pe_rank .eq. 0) write(*, '(A,F8.3)') 'RS_LINE setup   link_columns_global ', MPI_Wtime() - t0
    end block
  end subroutine build_columns

  !> Neighbours across the two faces normal to each selected element's
  !! direction dir(e): the selected element e2 that shares all lx**2 face
  !! node ids and whose own direction is normal to that face; nb_o = e2's
  !! traversal orientation when entered through the face, nb_L = map of the
  !! lines through the face nodes (e line -> e2 line).
  subroutine face_neighbours(this, coef, sel, nb_e, nb_o, nb_L)
    class(rs_line_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    logical, intent(in) :: sel(:)
    integer, allocatable, intent(out) :: nb_e(:,:), nb_o(:,:), nb_L(:,:,:)
    integer :: lx, nelv, nl, lx3, e, L, idx, d, i, f, e2, idx2, L2, p2, d2, pl
    integer(kind=i8) :: id, maxid
    integer, allocatable :: first_e(:), first_idx(:), second_e(:), second_idx(:)
    integer :: ext(3)

    lx = this%lx; nelv = this%nelv; nl = lx * lx; lx3 = lx**3
    ! unique-id lookup (first two elements holding each id)
    maxid = maxval(coef%dof%dof)
    ! ids start at 0: index the lookup tables with id + 1
    allocate(first_e(maxid + 1), first_idx(maxid + 1), second_e(maxid + 1), &
         second_idx(maxid + 1))
    first_e = 0; second_e = 0; first_idx = 0; second_idx = 0
    do e = 1, nelv
       do idx = 1, lx3
          id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
          if (id .eq. 0) cycle
          if (first_e(id + 1) .eq. 0) then
             first_e(id + 1) = e; first_idx(id + 1) = idx
          else if (second_e(id + 1) .eq. 0 .and. first_e(id + 1) .ne. e) then
             second_e(id + 1) = e; second_idx(id + 1) = idx
          end if
       end do
    end do

    ! neighbours across the two faces (f = 1: p = 1, f = 2: p = lx) of each
    ! element: the element (other than e) that contains ALL lx**2 node ids
    ! of the face (works for lx = 2, where every face node is a vertex).
    ! nb_o = the neighbour's traversal orientation when entered through that
    ! face (+1: its p increases away from the face, -1: decreases); nb_L:
    ! lateral line map by matching ids.
    allocate(nb_e(2, nelv), nb_o(2, nelv), nb_L(nl, 2, nelv))
    nb_e = 0; nb_o = 0; nb_L = 0
    block
      integer, parameter :: MAXSH = 8
      integer, allocatable :: nsh(:), she(:,:), shi(:,:)
      integer(kind=i8), allocatable :: fid(:)
      integer :: k, cand, nmatch, L0, ii0, jj0, kk0, c1, c2, c3, idx0, ia0
      allocate(nsh(maxid + 1), she(MAXSH, maxid + 1), shi(MAXSH, maxid + 1), fid(nl))
      nsh = 0
      do e = 1, nelv
         do idx = 1, lx3
            id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
            if (id .eq. 0) cycle
            if (nsh(id + 1) .lt. MAXSH) then
               nsh(id + 1) = nsh(id + 1) + 1
               she(nsh(id + 1), id + 1) = e
               shi(nsh(id + 1), id + 1) = idx
            end if
         end do
      end do
      do e = 1, nelv
         d = this%dir(e)
         do f = 1, 2
            if (f .eq. 1) then
               pl = 1
            else
               pl = lx
            end if
            do L = 1, nl
               idx = node_index(lx, d, L, pl)
               fid(L) = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
            end do
            if (any(fid .eq. 0)) cycle
            ! candidates: elements containing the first face node
            e2 = 0
            do k = 1, nsh(fid(1) + 1)
               cand = she(k, fid(1) + 1)
               if (cand .eq. e) cycle
               nmatch = 0
               do L = 1, nl
                  do i = 1, nsh(fid(L) + 1)
                     if (she(i, fid(L) + 1) .eq. cand) then
                        nmatch = nmatch + 1
                        exit
                     end if
                  end do
               end do
               if (nmatch .eq. nl) then
                  e2 = cand
                  exit
               end if
            end do
            if (e2 .eq. 0) cycle
            if (.not. sel(e) .or. .not. sel(e2)) cycle
            d2 = this%dir(e2)
            ! the face in e2: the index that is constant (1 or lx) over the
            ! matched nodes; take the first face node's position in e2
            ia0 = 0
            do i = 1, nsh(fid(1) + 1)
               if (she(i, fid(1) + 1) .eq. e2) ia0 = shi(i, fid(1) + 1)
            end do
            c1 = mod(ia0-1,lx)+1; c2 = mod((ia0-1)/lx,lx)+1; c3 = (ia0-1)/nl+1
            ! which of the three indices equals 1 or lx for ALL face nodes:
            ext = 0
            ! test index d2 only: the column continues only if the shared
            ! face is normal to e2's own stiff direction
            select case (d2)
            case (1); ii0 = c1
            case (2); ii0 = c2
            case default; ii0 = c3
            end select
            if (ii0 .ne. 1 .and. ii0 .ne. lx) cycle
            ! verify that all face nodes have the same d2-index ii0 in e2
            nmatch = 0
            do L = 1, nl
               do i = 1, nsh(fid(L) + 1)
                  if (she(i, fid(L) + 1) .eq. e2) then
                     idx2 = shi(i, fid(L) + 1)
                     call line_of(lx, d2, idx2, L2, p2)
                     if (p2 .eq. ii0) nmatch = nmatch + 1
                     exit
                  end if
               end do
            end do
            if (nmatch .ne. nl) cycle
            nb_e(f, e) = e2
            if (ii0 .eq. 1) then
               nb_o(f, e) = 1
            else
               nb_o(f, e) = -1
            end if
            ! lateral line map: the e2 line through each face node
            do L = 1, nl
               do i = 1, nsh(fid(L) + 1)
                  if (she(i, fid(L) + 1) .eq. e2) then
                     idx2 = shi(i, fid(L) + 1)
                     call line_of(lx, d2, idx2, L2, p2)
                     nb_L(L, f, e) = L2
                     exit
                  end if
               end do
            end do
         end do
      end do
      deallocate(nsh, she, shi, fid)
    end block
    deallocate(first_e, first_idx, second_e, second_idx)
  end subroutine face_neighbours


  ! ---------------------------------------------------------------------------
  ! plane strips
  ! ---------------------------------------------------------------------------

  !> Coordinate along direction x (/= d) of line L of direction d (the line
  !! index is a + lx (b - 1), a along the smaller other direction).
  pure function lcoord(lx, d, x, L) result(c)
    integer, intent(in) :: lx, d, x, L
    integer :: c, o1
    o1 = merge(2, 1, d .eq. 1)
    if (x .eq. o1) then
       c = mod(L - 1, lx) + 1
    else
       c = (L - 1) / lx + 1
    end if
  end function lcoord

  !> Line of direction d through coordinate cx along x and cy along y.
  pure function lline(lx, d, x, cx, cy) result(L)
    integer, intent(in) :: lx, d, x, cx, cy
    integer :: L, o1
    o1 = merge(2, 1, d .eq. 1)
    if (x .eq. o1) then
       L = cx + lx * (cy - 1)
    else
       L = cy + lx * (cx - 1)
    end if
  end function lline

  !> Local node index with coordinate p along d, t along dc, k along dn.
  pure function pnode(lx, d, dc, dn, p, t, k) result(idx)
    integer, intent(in) :: lx, d, dc, dn, p, t, k
    integer :: idx, c(3)
    c(d) = p; c(dc) = t; c(dn) = k
    idx = c(1) + lx * (c(2) - 1) + lx * lx * (c(3) - 1)
  end function pnode

  !> Plane elements: the middle coupling dominates the weakest by
  !! plane_ratio (review 6: the former pencil-like branch with a fixed 1.44
  !! made nearly cubic elements planes). Normal = weakest.
  subroutine classify_planes(this, s3)
    class(rs_line_t), intent(inout) :: this
    real(kind=dp), intent(in) :: s3(:,:)
    integer :: e, dn, d
    real(kind=dp) :: hi, lo, mid
    allocate(this%pel(this%nelv), this%pnorm(this%nelv), this%pid(this%nelv))
    this%pel = .false.; this%pnorm = 0; this%pid = 0; this%npel = 0
    do e = 1, this%nelv
       if (this%iso(e)) cycle
       d = this%dir(e)
       dn = minloc(s3(:, e), 1)
       if (dn .eq. d) cycle
       hi = s3(d, e); lo = max(s3(dn, e), tiny(1.0_dp)); mid = s3(6 - d - dn, e)
       if (mid .ge. this%plane_ratio * lo) then
          this%pel(e) = .true.; this%pnorm(e) = dn
          this%npel = this%npel + 1; this%pid(e) = this%npel
       end if
    end do
  end subroutine classify_planes

  !> Second comb pass: in-plane couplings of the plane elements,
  !! pblk(p + lx (t - 1), p0 + lx (t0 - 1), k, pid(e)) = A_e(node(p, t, k), node(p0, t0, k)).
  subroutine comb_planes(this, ax, coef, x, w)
    class(rs_line_t), intent(inout), target :: this
    class(ax_t), intent(in) :: ax
    type(coef_t), intent(inout), target :: coef
    real(kind=rp), intent(inout) :: x(:), w(:)
    integer :: lx, nl, lx3, n, e, off, cs(3), pe, d, dn, dc, p0, t0, k0, p, t, i
    integer, allocatable :: plist(:)
    type(c_ptr) :: x_d, w_d, xd_d, wd_d

    lx = this%lx; nl = lx * lx; lx3 = lx**3; n = this%n
    allocate(this%pblk(nl, nl, lx, this%npel), plist(this%npel))
    this%pblk = 0.0_dp
    do e = 1, this%nelv
       if (this%pid(e) .gt. 0) plist(this%pid(e)) = e
    end do
    if (NEKO_BCKND_DEVICE .eq. 1) then
       ! comb, Ax and the extraction on the device; one copy back at the end
       block
         integer, allocatable :: pinfo(:)
         type(c_ptr) :: pblk_d, pinfo_d
         integer(c_size_t) :: nbyt
         x_d = c_null_ptr; w_d = c_null_ptr; pblk_d = c_null_ptr; pinfo_d = c_null_ptr
         call device_map(x, x_d, n)
         call device_map(w, w_d, n)
         xd_d = x_d; wd_d = w_d
         if (rp .ne. dp) then
            call device_alloc(xd_d, int(n, c_size_t) * 8_c_size_t)
            call device_alloc(wd_d, int(n, c_size_t) * 8_c_size_t)
         end if
         ! more than 2^31 entries on large meshes: allocate and copy by bytes
         nbyt = int(size(this%pblk, kind=i8), c_size_t) * int(storage_size(1.0_dp) / 8, c_size_t)
         call device_alloc(pblk_d, nbyt)
         allocate(pinfo(4 * this%npel))
         do pe = 1, this%npel
            e = plist(pe); d = this%dir(e); dn = this%pnorm(e); dc = 6 - d - dn
            pinfo(4 * pe - 3) = e - 1; pinfo(4 * pe - 2) = d - 1
            pinfo(4 * pe - 1) = dc - 1; pinfo(4 * pe) = dn - 1
         end do
         call device_map(pinfo, pinfo_d, size(pinfo))
         call device_memcpy(pinfo, pinfo_d, size(pinfo), HOST_TO_DEVICE, sync = .true.)
         do off = 1, lx3
            call rs_line_cuda_comb_set(xd_d, int(lx3, c_int), int(off - 1, c_int), int(n, c_int))
            if (rp .ne. dp) call rs_line_cuda_d2r(x_d, xd_d, int(n, c_int))
            call rs_line_cuda_sync()
            call ax%compute(w, x, coef, coef%msh, coef%Xh)
            call device_sync()
            if (rp .ne. dp) call rs_line_cuda_r2d(wd_d, w_d, int(n, c_int))
            call rs_line_cuda_plane_extract(wd_d, pblk_d, pinfo_d, int(lx, c_int), int(off - 1, c_int), &
                 int(this%npel, c_int))
            call rs_line_cuda_sync()
         end do
         call rs_line_cuda_copy(c_loc(this%pblk), pblk_d, nbyt, 2_c_int)
         call device_free(pblk_d)
         call device_deassociate(pinfo); call device_free(pinfo_d)
         call device_deassociate(x); call device_free(x_d)
         call device_deassociate(w); call device_free(w_d)
         if (rp .ne. dp) then
            call device_free(xd_d); call device_free(wd_d)
         end if
       end block
       deallocate(plist)
       return
    end if
    do off = 1, lx3
       cs(1) = mod(off - 1, lx) + 1; cs(2) = mod((off - 1) / lx, lx) + 1; cs(3) = (off - 1) / nl + 1
       do i = 1, n
          x(i) = 0.0_dp
       end do
       do e = 1, this%nelv
          x(off + (e - 1) * lx3) = 1.0_dp
       end do
       call ax%compute(w, x, coef, coef%msh, coef%Xh)
       !$omp parallel do private(pe, e, d, dn, dc, p0, t0, k0, p, t) schedule(static)
       do pe = 1, this%npel
          e = plist(pe); d = this%dir(e); dn = this%pnorm(e); dc = 6 - d - dn
          p0 = cs(d); t0 = cs(dc); k0 = cs(dn)
          do t = 1, lx
             do p = 1, lx
                this%pblk(p + lx * (t - 1), p0 + lx * (t0 - 1), k0, pe) = &
                     w(pnode(lx, d, dc, dn, p, t, k0) + (e - 1) * lx3)
             end do
          end do
       end do
       !$omp end parallel do
    end do
    deallocate(plist)
  end subroutine comb_planes

  !> Strips: bundles of the lx lines of one node layer (normal pnorm) of
  !! plane elements, chained along dir across faces to plane elements in
  !! which the continued lines again form one layer; banded factors.
  subroutine build_strips(this, coef)
    class(rs_line_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    integer, allocatable :: nb_e(:,:), nb_o(:,:), nb_L(:,:,:)
    logical, allocatable :: vis(:,:)
    integer :: lx, nelv, lx3, e, k, s, kd, ldab, info, ntot, m, ns
    integer :: pass

    lx = this%lx; nelv = this%nelv; lx3 = lx**3
    call face_neighbours(this, coef, this%pel, nb_e, nb_o, nb_L)
    allocate(vis(lx, nelv))
    do pass = 1, 2
       vis = .false.
       this%nstrip = 0
       if (pass .eq. 1) this%smaxseg = 1
       ! starts: no consistent predecessor through the low face (walk up),
       ! then through the high face (walk down), then leftovers (loops)
       do e = 1, nelv
          if (.not. this%pel(e)) cycle
          do k = 1, lx
             if (vis(k, e)) cycle
             if (layer_next(this, nb_e, nb_L, e, k, 1) .ne. 0) cycle
             call walk_strip(e, k, 1, pass .eq. 2)
          end do
       end do
       do e = 1, nelv
          if (.not. this%pel(e)) cycle
          do k = 1, lx
             if (vis(k, e)) cycle
             if (layer_next(this, nb_e, nb_L, e, k, 2) .ne. 0) cycle
             call walk_strip(e, k, -1, pass .eq. 2)
          end do
       end do
       do e = 1, nelv
          if (.not. this%pel(e)) cycle
          do k = 1, lx
             if (vis(k, e)) cycle
             call walk_strip(e, k, 1, pass .eq. 2)
          end do
       end do
       if (pass .eq. 1) then
          allocate(this%snseg(this%nstrip), this%sseg_e(this%smaxseg, this%nstrip), &
               this%sseg_k(this%smaxseg, this%nstrip), this%sseg_o(this%smaxseg, this%nstrip), &
               this%sseg_x(lx, this%smaxseg, this%nstrip))
          this%snseg = 0
       end if
    end do
    deallocate(vis, nb_e, nb_o, nb_L)
    ! banded blocks
    kd = lx * lx - 1; ldab = kd + 1
    allocate(this%sbo(this%nstrip + 1))
    this%sbo(1) = 0
    this%smaxn = 0
    do s = 1, this%nstrip
       ns = lx * (this%snseg(s) * (lx - 1) + 1)
       this%sbo(s + 1) = this%sbo(s) + ns
       this%smaxn = max(this%smaxn, ns)
    end do
    ntot = this%sbo(this%nstrip + 1)
    allocate(this%sab(ldab, max(ntot, 1)))
    !$omp parallel do private(s, info) schedule(dynamic, 16)
    do s = 1, this%nstrip
       call strip_assemble(this, s)
       call dpbtrf('L', this%sbo(s + 1) - this%sbo(s), kd, this%sab(1, this%sbo(s) + 1), ldab, info)
       if (info .ne. 0) then
          !$omp critical
          write(*, '(A,I0,A,I0)') 'rs_line: strip ', s, ' not SPD, dpbtrf info ', info
          call neko_error('rs_line: dpbtrf failed (strip not SPD)')
          !$omp end critical
       end if
    end do
    !$omp end parallel do
    ! the in-plane couplings are only needed for the assembly
    if (allocated(this%pblk)) deallocate(this%pblk)
  contains
    subroutine walk_strip(e0, k0, o0, fill)
      integer, intent(in) :: e0, k0, o0
      logical, intent(in) :: fill
      integer :: e, k, o, f, m, e2, k2, t, d, dc, dn, sidx
      integer :: Lc(lx), L2(lx)
      this%nstrip = this%nstrip + 1
      sidx = this%nstrip
      e = e0; k = k0; o = o0
      d = this%dir(e); dn = this%pnorm(e); dc = 6 - d - dn
      do t = 1, lx
         Lc(t) = lline(lx, d, dc, t, k)
      end do
      m = 0
      do
         m = m + 1
         vis(k, e) = .true.
         if (fill) then
            d = this%dir(e); dn = this%pnorm(e); dc = 6 - d - dn
            this%sseg_e(m, sidx) = e; this%sseg_k(m, sidx) = k; this%sseg_o(m, sidx) = o
            do t = 1, lx
               this%sseg_x(t, m, sidx) = lcoord(lx, d, dc, Lc(t))
            end do
            this%snseg(sidx) = m
         end if
         f = merge(2, 1, o .eq. 1)
         e2 = nb_e(f, e)
         if (e2 .eq. 0) exit
         do t = 1, lx
            L2(t) = nb_L(Lc(t), f, e)
         end do
         k2 = lcoord(lx, this%dir(e2), this%pnorm(e2), L2(1))
         if (any([(lcoord(lx, this%dir(e2), this%pnorm(e2), L2(t)) .ne. k2, t = 1, lx)])) exit
         if (vis(k2, e2)) exit
         o = nb_o(f, e); e = e2; k = k2; Lc = L2
         if (m .ge. nelv) exit
      end do
      if (.not. fill) this%smaxseg = max(this%smaxseg, m)
    end subroutine walk_strip
  end subroutine build_strips

  !> Element across face f of plane element e whose continued lines of
  !! layer k form one layer there (0 if none).
  function layer_next(this, nb_e, nb_L, e, k, f) result(e2)
    class(rs_line_t), intent(in) :: this
    integer, intent(in) :: nb_e(:,:), nb_L(:,:,:), e, k, f
    integer :: e2, t, d, dn, dc, k2, L2
    e2 = nb_e(f, e)
    if (e2 .eq. 0) return
    d = this%dir(e); dn = this%pnorm(e); dc = 6 - d - dn
    k2 = -1
    do t = 1, this%lx
       L2 = nb_L(lline(this%lx, d, dc, t, k), f, e)
       if (k2 .lt. 0) k2 = lcoord(this%lx, this%dir(e2), this%pnorm(e2), L2)
       if (lcoord(this%lx, this%dir(e2), this%pnorm(e2), L2) .ne. k2) then
          e2 = 0
          return
       end if
    end do
  end function layer_next

  !> Strip statistics over all ranks (sums and maxima; the strips are
  !! rank-local). Collective: called on every rank when planes are on.
  subroutine strip_stats(this, tsetup)
    class(rs_line_t), intent(in) :: this
    real(kind=dp), intent(in) :: tsetup
    integer(kind=i8) :: st(3), stmax(3)
    integer :: m, ierr2
    st = 0_i8
    if (this%nstrip .gt. 0) st = [int(this%npel, i8), int(this%nstrip, i8), &
         int(this%sbo(this%nstrip + 1), i8) * int(this%lx * this%lx, i8)]
    stmax = st
    call MPI_Allreduce(MPI_IN_PLACE, st, 3, MPI_INTEGER8, MPI_SUM, NEKO_COMM, ierr2)
    call MPI_Allreduce(MPI_IN_PLACE, stmax, 3, MPI_INTEGER8, MPI_MAX, NEKO_COMM, ierr2)
    m = this%smaxseg
    call MPI_Allreduce(MPI_IN_PLACE, m, 1, MPI_INTEGER, MPI_MAX, NEKO_COMM, ierr2)
    if (pe_rank .eq. 0) write(*, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,F8.3)') &
         'RS_LINE strips (all ranks): plane elements ', st(1), ' (max/rank ', stmax(1), &
         ') strips ', st(2), ' (max/rank ', stmax(2), ') max strip elements ', m, &
         ' band entries ', st(3), ' (max/rank ', stmax(3), ') setup(rank0) ', tsetup
  end subroutine strip_stats

  !> Local node and strip row of strip position (m, p, t).
  pure subroutine strip_node(this, s, m, p, t, idx, irow)
    class(rs_line_t), intent(in) :: this
    integer, intent(in) :: s, m, p, t
    integer, intent(out) :: idx, irow
    integer :: lx, e, d, dn, dc
    lx = this%lx
    e = this%sseg_e(m, s); d = this%dir(e); dn = this%pnorm(e); dc = 6 - d - dn
    idx = pnode(lx, d, dc, dn, pos_of(lx, this%sseg_o(m, s), p), this%sseg_x(t, m, s), &
         this%sseg_k(m, s)) + (e - 1) * lx**3
    irow = t + lx * ((m - 1) * (lx - 1) + p - 1)
  end subroutine strip_node

  !> Band matrix of strip s: in-plane couplings of its elements (summed on
  !! the junction faces), assembled diagonal, Dirichlet nodes decoupled.
  subroutine strip_assemble(this, s)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: s
    integer :: lx, m, e, k, o, pe, p1, t1, p2, t2, i1, i2, j1, j2, o0, kd, ns, idx, irow, jc
    lx = this%lx; kd = lx * lx - 1
    o0 = this%sbo(s); ns = this%sbo(s + 1) - o0
    this%sab(:, o0 + 1 : o0 + ns) = 0.0_dp
    do m = 1, this%snseg(s)
       e = this%sseg_e(m, s); k = this%sseg_k(m, s); o = this%sseg_o(m, s); pe = this%pid(e)
       do t2 = 1, lx
          do p2 = 1, lx
             j1 = pos_of(lx, o, p2) + lx * (this%sseg_x(t2, m, s) - 1)
             j2 = t2 + lx * ((m - 1) * (lx - 1) + p2 - 1)
             do t1 = 1, lx
                do p1 = 1, lx
                   i2 = t1 + lx * ((m - 1) * (lx - 1) + p1 - 1)
                   if (i2 .le. j2) cycle
                   i1 = pos_of(lx, o, p1) + lx * (this%sseg_x(t1, m, s) - 1)
                   this%sab(1 + i2 - j2, o0 + j2) = this%sab(1 + i2 - j2, o0 + j2) + &
                        this%pblk(i1, j1, k, pe)
                end do
             end do
          end do
       end do
    end do
    ! assembled diagonal, Dirichlet rows and columns
    do m = 1, this%snseg(s)
       do p1 = 1, lx
          do t1 = 1, lx
             call strip_node(this, s, m, p1, t1, idx, irow)
             this%sab(1, o0 + irow) = this%diag(idx)
             if (this%mask(idx) .lt. 0.5_dp) then
                do jc = max(1, irow - kd), irow - 1
                   this%sab(1 + irow - jc, o0 + jc) = 0.0_dp
                end do
                do jc = irow + 1, min(ns, irow + kd)
                   this%sab(1 + jc - irow, o0 + irow) = 0.0_dp
                end do
                this%sab(1, o0 + irow) = 1.0_dp
             end if
          end do
       end do
    end do
  end subroutine strip_assemble

  !> One local copy per (strip, unique node).
  subroutine strip_mark(this, cnt)
    class(rs_line_t), intent(in) :: this
    real(kind=dp), intent(inout) :: cnt(:)
    integer :: s, m, p, t, idx, irow
    do s = 1, this%nstrip
       do m = 1, this%snseg(s)
          do p = merge(1, 2, m .eq. 1), this%lx
             do t = 1, this%lx
                call strip_node(this, s, m, p, t, idx, irow)
                cnt(idx) = cnt(idx) + 1.0_dp
             end do
          end do
       end do
    end do
  end subroutine strip_mark

  !> z(strip nodes) = W^1/2 A_strip^-1 W^1/2 r (one local copy per node).
  subroutine strip_solve(this, z, r, n)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=dp), intent(inout) :: z(n)
    real(kind=dp), intent(in) :: r(n)
    real(kind=dp), allocatable :: y(:)
    integer :: s, m, p, t, idx, irow, ns, kd, info
    kd = this%lx * this%lx - 1
    !$omp parallel private(y, s, m, p, t, idx, irow, ns, info)
    allocate(y(max(this%smaxn, 1)))
    !$omp do schedule(dynamic, 16)
    do s = 1, this%nstrip
       ns = this%sbo(s + 1) - this%sbo(s)
       do m = 1, this%snseg(s)
          do p = 1, this%lx
             do t = 1, this%lx
                call strip_node(this, s, m, p, t, idx, irow)
                y(irow) = this%wsq(idx) * this%mask(idx) * r(idx)
             end do
          end do
       end do
       call dpbtrs('L', ns, kd, 1, this%sab(1, this%sbo(s) + 1), kd + 1, y, ns, info)
       do m = 1, this%snseg(s)
          do p = merge(1, 2, m .eq. 1), this%lx
             do t = 1, this%lx
                call strip_node(this, s, m, p, t, idx, irow)
                z(idx) = this%wsq(idx) * this%mask(idx) * y(irow)
             end do
          end do
       end do
    end do
    !$omp end do
    deallocate(y)
    !$omp end parallel
  end subroutine strip_solve

  !> Position (in the element's own frame) of the p-th node of a segment
  !! traversed with orientation o.
  pure function pos_of(lx, o, p) result(pl)
    integer, intent(in) :: lx, o, p
    integer :: pl
    if (o .eq. 1) then
       pl = p
    else
       pl = lx + 1 - p
    end if
  end function pos_of

  !> Segment m_g (1..nseg, in the global traversal order) of local chain c:
  !! element, line, and orientation in the global order.
  subroutine gseg(this, c, mg, e, L, og)
    class(rs_line_t), intent(in) :: this
    integer, intent(in) :: c, mg
    integer, intent(out) :: e, L, og
    integer :: m
    if (this%cg_rev(c)) then
       m = this%nseg(c) + 1 - mg
       e = this%cseg_e(m, c); L = this%cseg_L(m, c); og = -this%cseg_o(m, c)
    else
       m = mg
       e = this%cseg_e(m, c); L = this%cseg_L(m, c); og = this%cseg_o(m, c)
    end if
  end subroutine gseg

  !> Link the local chains across ranks into global columns. Chain ends on
  !! MPI-shared faces are published (face key = min, max and sum of the
  !! face node ids, plus the line's end node id) and matched; the matched
  !! chains are walked into global columns, each owned by one of its ranks
  !! (round robin). The owner receives the segment blocks once (banded LU),
  !! and at every application the pieces of the right-hand side (one
  !! MPI_Alltoallv each way).
  subroutine link_columns_global(this, coef)
    class(rs_line_t), intent(inout) :: this
    type(coef_t), intent(in), target :: coef
    integer, parameter :: NR = 8
    integer :: lx, nl, lx3, c, s, e, L, o, pl, idx, i, j, k, ierr, nrec, ntot
    integer :: nchain, g, side, sx, r, cc, nsg, offs, kk, p, mg, q, qp, info
    integer :: kl, ku, ldab, lenc, pos, nb, nblk, gtoth
    integer(kind=i8) :: id, k1, k2, k3, ck
    integer(kind=i8), allocatable :: rec(:,:), allrec(:,:), hkey(:), ckey(:)
    integer, allocatable :: counts(:), displs(:), perm(:), permc(:), partner(:)
    logical, allocatable :: used(:)
    integer, allocatable :: gl_rank(:), gl_c(:), gl_off(:), gl_nseg(:), gl_rev(:)
    integer, allocatable :: gl_first(:), gl_nch(:), gl_owner(:), gl_total(:)
    integer :: ngl, nchmax, cur, rec_in, rec_out, side_in
    real(kind=dp), allocatable :: bs(:), br(:)
    integer, allocatable :: bsendcnt(:), bsenddsp(:), brecvcnt(:), brecvdsp(:)
    integer, allocatable :: ps_g(:), ps_off(:), pr_g(:), pr_off(:), pr_src(:), ord(:)
    integer :: nps, npr, nmulti
    character(len=LOG_SIZE) :: log_buf
    real(kind=dp) :: tl(0:8)

    lx = this%lx; nl = lx * lx; lx3 = lx**3
    tl(0) = MPI_Wtime()
    ! ---- records of the chain ends on shared faces ----------------------------
    allocate(rec(NR, 2 * max(this%ncol, 1)))
    nrec = 0
    do c = 1, this%ncol
       do s = 1, 2
          if (s .eq. 1) then
             e = this%cseg_e(1, c); L = this%cseg_L(1, c); o = this%cseg_o(1, c)
             pl = pos_of(lx, o, 1)
          else
             e = this%cseg_e(this%nseg(c), c); L = this%cseg_L(this%nseg(c), c)
             o = this%cseg_o(this%nseg(c), c)
             pl = pos_of(lx, o, lx)
          end if
          k1 = huge(1_i8); k2 = -1_i8; k3 = 0_i8
          do i = 1, nl
             idx = node_index(lx, this%dir(e), i, pl)
             if (.not. coef%dof%shared_dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, &
                  (idx-1)/nl+1, e)) then
                k2 = -2_i8
                exit
             end if
             id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
             k1 = min(k1, id); k2 = max(k2, id); k3 = k3 + id
          end do
          if (k2 .lt. 0_i8) cycle
          idx = node_index(lx, this%dir(e), L, pl)
          id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
          nrec = nrec + 1
          rec(1, nrec) = k1; rec(2, nrec) = k2; rec(3, nrec) = k3; rec(4, nrec) = id
          rec(5, nrec) = int(pe_rank, i8); rec(6, nrec) = int(c, i8)
          rec(7, nrec) = int(s, i8); rec(8, nrec) = int(this%nseg(c), i8)
       end do
    end do
    allocate(counts(pe_size), displs(pe_size))
    call MPI_Allgather(nrec * NR, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    displs(1) = 0
    do i = 2, pe_size
       displs(i) = displs(i-1) + counts(i-1)
    end do
    ntot = sum(counts) / NR
    allocate(allrec(NR, max(ntot, 1)))
    tl(1) = MPI_Wtime()
    call MPI_Allgatherv(rec, nrec * NR, MPI_INTEGER8, allrec, counts, displs, &
         MPI_INTEGER8, NEKO_COMM, ierr)
    deallocate(rec)
    tl(2) = MPI_Wtime()
    if (ntot .eq. 0) then
       deallocate(counts, displs, allrec)
       return
    end if

    ! ---- matching: sort by a 64-bit hash of (face key, node id) in a
    ! contiguous array (the full keys are compared on equal hashes) ----------
    allocate(perm(ntot), partner(ntot), hkey(ntot))
    do i = 1, ntot
       hkey(i) = mix4(allrec(1, i), allrec(2, i), allrec(3, i), allrec(4, i))
    end do
    call sort_ki(ntot, hkey, perm)
    partner = 0; nmulti = 0
    i = 1
    do while (i .le. ntot)
       j = i
       do while (j .lt. ntot)
          if (hkey(j+1) .eq. hkey(i)) then
             j = j + 1
          else
             exit
          end if
       end do
       if (j - i + 1 .eq. 2) then
          if (same_key(allrec(:, perm(i)), allrec(:, perm(j)), 4)) then
             partner(perm(i)) = perm(j); partner(perm(j)) = perm(i)
          end if
       else if (j - i + 1 .gt. 2) then
          nmulti = nmulti + 1
       end if
       i = j + 1
    end do
    tl(3) = MPI_Wtime()
    ! chain lookup: records sorted by the packed (rank, chain, side)
    allocate(permc(ntot), ckey(ntot))
    do i = 1, ntot
       ckey(i) = pack_rcs(allrec(5, i), allrec(6, i), allrec(7, i))
    end do
    call sort_ki(ntot, ckey, permc)
    tl(4) = MPI_Wtime()

    ! ---- walk the global columns ------------------------------------------------
    allocate(used(ntot))
    used = .false.
    nchmax = ntot
    allocate(gl_rank(nchmax), gl_c(nchmax), gl_off(nchmax), gl_nseg(nchmax), gl_rev(nchmax))
    allocate(gl_first(ntot + 1), gl_nch(ntot), gl_owner(ntot), gl_total(ntot))
    ngl = 0; nchain = 0
    do kk = 1, 2
       ! pass 1: start from chains with an unmatched side; pass 2: loops
       do i = 1, ntot
          r = permc(i)
          if (used(r)) cycle
          if (partner(r) .eq. 0 .and. kk .eq. 1) then
             ! an unmatched record: start here if the chain's other side is matched
             rec_out = other_side(r)
             if (rec_out .eq. 0) cycle
             if (partner(rec_out) .eq. 0) cycle
             side = int(allrec(7, r))      ! the unmatched side: we start from it
          else if (kk .eq. 1) then
             ! matched record: is the other side unmatched / unpublished?
             rec_out = other_side(r)
             if (rec_out .ne. 0) then
                if (partner(rec_out) .ne. 0) cycle
             end if
             ! start from the (unpublished or unmatched) other side
             side = 3 - int(allrec(7, r))
             rec_out = r
          else
             if (partner(r) .eq. 0) cycle
             side = 3 - int(allrec(7, r))
             rec_out = r
          end if
          ! walk: current chain = that of record r, entered at 'side'
          ngl = ngl + 1
          gl_first(ngl) = nchain + 1
          offs = 0
          cur = r
          side_in = side
          do
             nchain = nchain + 1
             gl_rank(nchain) = int(allrec(5, cur)); gl_c(nchain) = int(allrec(6, cur))
             gl_nseg(nchain) = int(allrec(8, cur))
             gl_off(nchain) = offs + 1
             if (side_in .eq. 1) then
                gl_rev(nchain) = 0
             else
                gl_rev(nchain) = 1
             end if
             offs = offs + gl_nseg(nchain)
             ! mark both records of this chain as used
             used(cur) = .true.
             k = other_side(cur)
             if (k .ne. 0) used(k) = .true.
             ! exit record: the side opposite to side_in
             if (int(allrec(7, cur)) .eq. 3 - side_in) then
                rec_out = cur
             else
                rec_out = other_side(cur)
             end if
             if (rec_out .eq. 0) exit
             if (partner(rec_out) .eq. 0) exit
             rec_in = partner(rec_out)
             if (used(rec_in)) exit          ! loop closed
             cur = rec_in
             side_in = int(allrec(7, cur))
             if (nchain .ge. nchmax) exit
          end do
          gl_nch(ngl) = nchain - gl_first(ngl) + 1
          gl_total(ngl) = offs
          ! owner: round robin over the chain positions
          gl_owner(ngl) = gl_rank(gl_first(ngl) + mod(ngl - 1, gl_nch(ngl)))
       end do
    end do
    gl_first(ngl + 1) = nchain + 1
    this%nglob = ngl
    tl(5) = MPI_Wtime()

    ! ---- my chains in global columns; pieces ----------------------------------
    nps = 0; npr = 0; this%nown = 0; this%gmaxn = 0
    do g = 1, ngl
       if (gl_owner(g) .eq. pe_rank) then
          this%nown = this%nown + 1
          this%gmaxn = max(this%gmaxn, gl_total(g) * (lx - 1) + 1)
          npr = npr + gl_nch(g)
       end if
       do k = gl_first(g), gl_first(g+1) - 1
          if (gl_rank(k) .eq. pe_rank) then
             c = gl_c(k)
             this%cg_id(c) = g; this%cg_rev(c) = gl_rev(k) .eq. 1
             this%cg_off(c) = gl_off(k); this%cg_own(c) = gl_owner(g)
             this%cg_last(c) = (gl_off(k) + gl_nseg(k) - 1 .eq. gl_total(g))
             nps = nps + 1
          end if
       end do
    end do
    this%nps = nps; this%npr = npr
    allocate(this%ps_c(max(nps,1)), this%ps_pos(max(nps,1)), this%ps_len(max(nps,1)), &
         this%ps_slot(max(nps,1)), this%pr_slot(max(npr,1)), this%pscnt(pe_size), this%psdsp(pe_size), &
         this%prcnt(pe_size), this%prdsp(pe_size), this%own_p0(max(this%nown,1)), this%own_np(max(this%nown,1)))
    this%pscnt = 0; this%prcnt = 0
    allocate(this%pr_own(max(npr,1)), this%pr_q0(max(npr,1)), this%pr_pos(max(npr,1)), &
         this%pr_len(max(npr,1)), pr_src(max(npr,1)))
    allocate(this%sendcnt(pe_size), this%senddsp(pe_size), this%recvcnt(pe_size), &
         this%recvdsp(pe_size))
    allocate(bsendcnt(pe_size), bsenddsp(pe_size), brecvcnt(pe_size), brecvdsp(pe_size))
    allocate(this%own_len(max(this%nown,1)))
    ! send pieces in (global column, offset) order: iterate g increasing, chains in order
    nps = 0
    this%sendcnt = 0; bsendcnt = 0
    do g = 1, ngl
       do k = gl_first(g), gl_first(g+1) - 1
          if (gl_rank(k) .ne. pe_rank) cycle
          nps = nps + 1
          this%ps_c(nps) = gl_c(k)
          this%ps_len(nps) = gl_nseg(k) * (lx - 1) + 1
          this%sendcnt(gl_owner(g) + 1) = this%sendcnt(gl_owner(g) + 1) + this%ps_len(nps)
          bsendcnt(gl_owner(g) + 1) = bsendcnt(gl_owner(g) + 1) + gl_nseg(k) * lx * lx
          this%pscnt(gl_owner(g) + 1) = this%pscnt(gl_owner(g) + 1) + 1
       end do
    end do
    this%psdsp(1) = 0
    do i = 2, pe_size
       this%psdsp(i) = this%psdsp(i-1) + this%pscnt(i-1)
    end do
    this%senddsp(1) = 0; bsenddsp(1) = 0
    do i = 2, pe_size
       this%senddsp(i) = this%senddsp(i-1) + this%sendcnt(i-1)
       bsenddsp(i) = bsenddsp(i-1) + bsendcnt(i-1)
    end do
    ! positions of the send pieces within their destination segments
    block
      integer, allocatable :: fillp(:), fillb(:), fillq(:)
      allocate(fillp(pe_size), fillb(pe_size), fillq(pe_size), ps_off(max(nps,1)))
      fillp = this%senddsp; fillb = bsenddsp; fillq = this%psdsp
      nps = 0
      do g = 1, ngl
         do k = gl_first(g), gl_first(g+1) - 1
            if (gl_rank(k) .ne. pe_rank) cycle
            nps = nps + 1
            r = gl_owner(g) + 1
            this%ps_pos(nps) = fillp(r) + 1; fillp(r) = fillp(r) + this%ps_len(nps)
            ps_off(nps) = fillb(r) + 1; fillb(r) = fillb(r) + gl_nseg(k) * lx * lx
            this%ps_slot(nps) = fillq(r); fillq(r) = fillq(r) + 1
         end do
      end do
      ! receive pieces (owned columns) in the same (g, offset) order per source
      this%recvcnt = 0; brecvcnt = 0
      npr = 0; this%nown = 0
      do g = 1, ngl
         if (gl_owner(g) .ne. pe_rank) cycle
         this%nown = this%nown + 1
         this%own_len(this%nown) = gl_total(g) * (lx - 1) + 1
         this%own_p0(this%nown) = npr + 1
         this%own_np(this%nown) = gl_first(g+1) - gl_first(g)
         do k = gl_first(g), gl_first(g+1) - 1
            npr = npr + 1
            this%pr_own(npr) = this%nown
            this%pr_q0(npr) = (gl_off(k) - 1) * (lx - 1) + 1
            this%pr_len(npr) = gl_nseg(k) * (lx - 1) + 1
            pr_src(npr) = gl_rank(k) + 1
            this%recvcnt(pr_src(npr)) = this%recvcnt(pr_src(npr)) + this%pr_len(npr)
            brecvcnt(pr_src(npr)) = brecvcnt(pr_src(npr)) + gl_nseg(k) * lx * lx
         end do
      end do
      this%recvdsp(1) = 0; brecvdsp(1) = 0
      do i = 2, pe_size
         this%recvdsp(i) = this%recvdsp(i-1) + this%recvcnt(i-1)
         brecvdsp(i) = brecvdsp(i-1) + brecvcnt(i-1)
      end do
      fillp = this%recvdsp; fillb = brecvdsp
      allocate(pr_off(max(npr,1)))
      do i = 1, npr
         this%prcnt(pr_src(i)) = this%prcnt(pr_src(i)) + 1
      end do
      this%prdsp(1) = 0
      do i = 2, pe_size
         this%prdsp(i) = this%prdsp(i-1) + this%prcnt(i-1)
      end do
      fillq = this%prdsp
      do i = 1, npr
         r = pr_src(i)
         this%pr_pos(i) = fillp(r) + 1; fillp(r) = fillp(r) + this%pr_len(i)
         pr_off(i) = fillb(r) + 1; fillb(r) = fillb(r) + (this%pr_len(i) - 1) / (lx - 1) * lx * lx
         this%pr_slot(i) = fillq(r); fillq(r) = fillq(r) + 1
      end do
      deallocate(fillp, fillb, fillq)
    end block
    allocate(this%sbuf(max(sum(this%sendcnt), 1)), this%rbuf(max(sum(this%recvcnt), 1)))
    allocate(this%hoff(max(this%nown, 1)))
    this%hoff(1) = 0
    do i = 2, this%nown
       this%hoff(i) = this%hoff(i - 1) + this%own_len(i - 1)
    end do
    gtoth = 0
    if (this%nown .gt. 0) gtoth = this%hoff(this%nown) + this%own_len(this%nown)
    allocate(this%gy(max(gtoth, 1)))

    tl(6) = MPI_Wtime()
    ! ---- exchange the segment blocks (global orientation) and factor ----------
    allocate(bs(max(sum(bsendcnt), 1)), br(max(sum(brecvcnt), 1)))
    do i = 1, this%nps
       c = this%ps_c(i)
       pos = ps_off(i)
       do mg = 1, this%nseg(c)
          call gseg(this, c, mg, e, L, o)
          do qp = 1, lx
             do q = 1, lx
                bs(pos) = this%blk(pos_of(lx, o, q), pos_of(lx, o, qp), L, e)
                pos = pos + 1
             end do
          end do
       end do
    end do
    call MPI_Alltoallv(bs, bsendcnt, bsenddsp, MPI_DOUBLE_PRECISION, br, brecvcnt, &
         brecvdsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    kl = lx - 1; ku = lx - 1; ldab = kl + 1
    allocate(this%gcab(ldab, max(gtoth, 1)))
    this%gcab = 0.0_dp
    do i = 1, this%npr
       pos = pr_off(i)
       nsg = (this%pr_len(i) - 1) / (lx - 1)
       do mg = 1, nsg
          do qp = 1, lx
             do q = 1, lx
                k = this%hoff(this%pr_own(i)) + this%pr_q0(i) + (mg - 1) * (lx - 1)  ! first node of the segment
                if (q .eq. qp) then
                   this%gcab(1, k + qp - 1) = br(pos)
                else if (q .gt. qp) then
                   this%gcab(1 + q - qp, k + qp - 1) = this%gcab(1 + q - qp, k + qp - 1) + br(pos)
                end if
                pos = pos + 1
             end do
          end do
       end do
    end do
    do i = 1, this%nown
       lenc = this%own_len(i)
       call dpbtrf('L', lenc, kl, this%gcab(1, this%hoff(i) + 1), ldab, info)
       if (info .ne. 0) call neko_error('rs_line: dpbtrf failed (global column not SPD)')
    end do
    deallocate(bs, br, bsendcnt, bsenddsp, brecvcnt, brecvdsp, ps_off, pr_off, pr_src)
    deallocate(counts, displs, allrec, perm, permc, partner, used, hkey, ckey)
    deallocate(gl_rank, gl_c, gl_off, gl_nseg, gl_rev, gl_first, gl_nch, gl_owner, gl_total)
    tl(7) = MPI_Wtime()
    if (pe_rank .eq. 0) then
       write(log_buf, '(A,I0,A,I0,A,I0,A,I0)') 'rs_line: global columns ', ngl, &
            ' chains ', nchain, ' max nodes ', this%gmaxn, ' multi-matches ', nmulti
       call neko_log%message(log_buf)
       write(*, '(A,I0,A,7F8.3)') 'RS_LINE link records ', ntot, &
            ' records/gather/sort1+match/sort2/walk/pieces/blocks+factor ', &
            tl(1) - tl(0), tl(2) - tl(1), tl(3) - tl(2), tl(4) - tl(3), tl(5) - tl(4), &
            tl(6) - tl(5), tl(7) - tl(6)
    end if

  contains

    !> The record of the other side of the chain of record r (0 if none).
    function other_side(r) result(k)
      integer, intent(in) :: r
      integer :: k, lo, hi, mid
      integer(kind=i8) :: key
      key = pack_rcs(allrec(5, r), allrec(6, r), 3_i8 - allrec(7, r))
      k = 0; lo = 1; hi = ntot
      do while (lo .le. hi)
         mid = (lo + hi) / 2
         if (ckey(mid) .eq. key) then
            k = permc(mid); return
         else if (ckey(mid) .lt. key) then
            lo = mid + 1
         else
            hi = mid - 1
         end if
      end do
    end function other_side

  end subroutine link_columns_global

  !> Exact packing of (rank < 2^20, chain < 2^40, side < 4).
  pure function pack_rcs(rank, chain, side) result(k)
    integer(kind=i8), intent(in) :: rank, chain, side
    integer(kind=i8) :: k
    k = ishft(rank, 42) + ishft(chain, 2) + side
  end function pack_rcs

  !> 64-bit mixing hash of four keys (splitmix-style).
  pure function mix4(a, b, c, d) result(h)
    integer(kind=i8), intent(in) :: a, b, c, d
    integer(kind=i8) :: h
    h = a
    h = ieor(h, ishft(h, -30)) * (-4658895280553007687_i8)
    h = h + b
    h = ieor(h, ishft(h, -27)) * (-7723592293110705685_i8)
    h = h + c
    h = ieor(h, ishft(h, -30)) * (-4658895280553007687_i8)
    h = h + d
    h = ieor(h, ishft(h, -31))
  end function mix4

  !> LSD radix sort (8 passes of 8 bits, sign bit flipped so that negative
  !! hashes order correctly) of contiguous 64-bit keys, carrying the
  !! original index. Linear in n; stable.
  subroutine sort_ki(n, k, idx)
    integer, intent(in) :: n
    integer(kind=i8), intent(inout) :: k(:)
    integer, intent(out) :: idx(:)
    integer(kind=i8), allocatable :: k2(:)
    integer, allocatable :: idx2(:)
    integer :: i, pass, b, cnt(0:255), pos(0:255)
    integer(kind=i8), parameter :: signbit = ishft(1_i8, 63)
    do i = 1, n
       idx(i) = i
    end do
    if (n .lt. 2) return
    allocate(k2(n), idx2(n))
    do i = 1, n
       k(i) = ieor(k(i), signbit)
    end do
    do pass = 0, 7
       cnt = 0
       do i = 1, n
          b = int(iand(ishft(k(i), -8 * pass), 255_i8))
          cnt(b) = cnt(b) + 1
       end do
       pos(0) = 1
       do b = 1, 255
          pos(b) = pos(b-1) + cnt(b-1)
       end do
       do i = 1, n
          b = int(iand(ishft(k(i), -8 * pass), 255_i8))
          k2(pos(b)) = k(i); idx2(pos(b)) = idx(i)
          pos(b) = pos(b) + 1
       end do
       k(1:n) = k2(1:n); idx(1:n) = idx2(1:n)
    end do
    do i = 1, n
       k(i) = ieor(k(i), signbit)
    end do
    deallocate(k2, idx2)
  end subroutine sort_ki

  pure function cmp3(a, b) result(c)
    integer(kind=i8), intent(in) :: a(3), b(3)
    integer :: c, i
    c = 0
    do i = 1, 3
       if (a(i) .lt. b(i)) then
          c = -1; return
       else if (a(i) .gt. b(i)) then
          c = 1; return
       end if
    end do
  end function cmp3

  pure function same_key(a, b, nk) result(eq)
    integer(kind=i8), intent(in) :: a(:), b(:)
    integer, intent(in) :: nk
    logical :: eq
    integer :: i
    eq = .true.
    do i = 1, nk
       if (a(i) .ne. b(i)) then
          eq = .false.; return
       end if
    end do
  end function same_key

  !> Heap sort of a permutation of the records by the given key columns.
  subroutine sort_perm(n, rec, keys, perm)
    integer, intent(in) :: n
    integer(kind=i8), intent(in) :: rec(:,:)
    integer, intent(in) :: keys(:)
    integer, intent(out) :: perm(:)
    integer :: i, m, t
    do i = 1, n
       perm(i) = i
    end do
    if (n .lt. 2) return
    do i = n / 2, 1, -1
       call sift(i, n)
    end do
    do m = n, 2, -1
       t = perm(1); perm(1) = perm(m); perm(m) = t
       call sift(1, m - 1)
    end do
  contains
    function less(a, b) result(l)
      integer, intent(in) :: a, b
      logical :: l
      integer :: k
      l = .false.
      do k = 1, size(keys)
         if (rec(keys(k), a) .lt. rec(keys(k), b)) then
            l = .true.; return
         else if (rec(keys(k), a) .gt. rec(keys(k), b)) then
            return
         end if
      end do
    end function less
    subroutine sift(start, last)
      integer, intent(in) :: start, last
      integer :: root, child, rv
      root = start
      rv = perm(root)
      do
         child = 2 * root
         if (child .gt. last) exit
         if (child .lt. last) then
            if (less(perm(child), perm(child + 1))) child = child + 1
         end if
         if (less(rv, perm(child))) then
            perm(root) = perm(child)
            root = child
         else
            exit
         end if
      end do
      perm(root) = rv
    end subroutine sift
  end subroutine sort_perm

  !> Walk one column from (e, L) through high faces; fill = .true. stores it.
  subroutine walk(this, nb_e, nb_o, nb_L, visited, col, e0, L0, nl, nelv, fill, o0, c)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: nb_e(:,:), nb_o(:,:), nb_L(:,:,:), e0, L0, nl, nelv, o0
    logical, intent(inout) :: visited(:,:)
    integer, intent(inout) :: col(:,:)
    logical, intent(in) :: fill
    integer, intent(out) :: c
    integer :: e, L, o, f, m, e2, L2

    this%ncol = this%ncol + 1
    c = this%ncol
    e = e0; L = L0; o = o0; m = 0
    do
       m = m + 1
       visited(L, e) = .true.
       col(L, e) = c
       if (fill) then
          this%cseg_e(m, c) = e; this%cseg_L(m, c) = L; this%cseg_o(m, c) = o
          this%nseg(c) = m
       end if
       ! exit face in the walk direction
       if (o .eq. 1) then
          f = 2
       else
          f = 1
       end if
       e2 = nb_e(f, e)
       if (e2 .eq. 0) exit
       L2 = nb_L(L, f, e)
       if (visited(L2, e2)) exit
       o = nb_o(f, e)
       e = e2; L = L2
       if (m .ge. nelv) exit
    end do
    if (.not. fill) this%maxseg = max(this%maxseg, m)
  end subroutine walk

  !> Mark one local copy per (column line, unique node) with 1.
  subroutine column_mark(this, cnt)
    class(rs_line_t), intent(in) :: this
    real(kind=dp), intent(inout) :: cnt(:)
    integer :: c, m, e, L, o, p, pl, idx, lx, pstart

    lx = this%lx
    do c = 1, this%ncol
       do m = 1, this%nseg(c)
          if (this%cg_id(c) .gt. 0) then
             ! global order: skip the entry node of every segment but the
             ! first, and that of the first if the chain is not the first
             call gseg(this, c, m, e, L, o)
             if (m .eq. 1 .and. this%cg_off(c) .eq. 1) then
                pstart = 1
             else
                pstart = 2
             end if
          else
          e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
          ! skip the shared entry node of segments m > 1
          if (m .eq. 1) then
             pstart = 1
          else
             pstart = 2
          end if
          end if
          do p = pstart, lx
             if (o .eq. 1) then
                pl = p
             else
                pl = lx + 1 - p
             end if
             idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx**3
             cnt(idx) = cnt(idx) + 1.0_dp
          end do
       end do
    end do
  end subroutine column_mark

  subroutine rs_line_solve(this, z, r, n)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    real(kind=dp), allocatable :: zd(:), rd(:)

    if (NEKO_BCKND_DEVICE .eq. 1) then
       call rs_line_solve_device(this, z, r, n)
       return
    end if
    ! host: the line data are double, z and r in Neko's precision
    allocate(zd(n), rd(n))
    rd = r
    call rs_line_solve_host(this, zd, rd, n)
    z = real(zd, rp)
    deallocate(zd, rd)
    call this%gs_h%op(z, n, GS_OP_ADD)
  end subroutine rs_line_solve

  !> Host line solve in double precision, without the final gather-scatter.
  subroutine rs_line_solve_host(this, z, r, n)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=dp), dimension(n), intent(inout) :: z
    real(kind=dp), dimension(n), intent(inout) :: r
    real(kind=dp) :: y(max(this%maxn, this%lx))
    integer :: e, L, p, d, idx, info, lx, nl, c, m, o, pl, q, lenc, kl, ku, ldab, pstart

    lx = this%lx
    nl = lx * lx
    do idx = 1, n
       z(idx) = 0.0_dp
    end do
    if (.not. this%columns) then
       do e = 1, this%nelv
          d = this%dir(e)
          do L = 1, nl
             do p = 1, lx
                idx = node_index(lx, d, L, p) + (e - 1) * lx**3
                y(p) = this%wsq(idx) * this%mask(idx) * r(idx)
             end do
             call dgetrs('N', lx, 1, this%lu(:,:,L,e), lx, this%ipiv(:,L,e), y, lx, info)
             do p = 1, lx
                idx = node_index(lx, d, L, p) + (e - 1) * lx**3
                z(idx) = this%wsq(idx) * this%mask(idx) * y(p)
             end do
          end do
       end do
    else
       kl = lx - 1; ku = lx - 1; ldab = kl + 1
       do c = 1, this%ncol
          if (this%cg_id(c) .gt. 0) cycle
          lenc = this%nseg(c) * (lx - 1) + 1
          do m = 1, this%nseg(c)
             e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
             do p = 1, lx
                if (o .eq. 1) then
                   pl = p
                else
                   pl = lx + 1 - p
                end if
                idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx**3
                q = (m - 1) * (lx - 1) + p
                y(q) = this%wsq(idx) * this%mask(idx) * r(idx)
             end do
          end do
          call dpbtrs('L', lenc, kl, 1, this%cab(1, this%cbo(c) + 1), ldab, y, this%maxn, info)
          do m = 1, this%nseg(c)
             e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
             if (m .eq. 1) then
                pstart = 1
             else
                pstart = 2
             end if
             do p = pstart, lx
                if (o .eq. 1) then
                   pl = p
                else
                   pl = lx + 1 - p
                end if
                idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx**3
                q = (m - 1) * (lx - 1) + p
                z(idx) = this%wsq(idx) * this%mask(idx) * y(q)
             end do
          end do
       end do
       if (this%nglob .gt. 0) call solve_global(this, z, r, n)
       if (this%nstrip .gt. 0) call strip_solve(this, z, r, n)
       do e = 1, this%nelv
          if (.not. this%iso(e)) cycle
          do idx = (e - 1) * lx**3 + 1, e * lx**3
             z(idx) = this%jmask(idx) * this%wsq(idx)**2 * this%mask(idx) * r(idx) / &
                  this%diag(idx)
          end do
       end do
    end if
  end subroutine rs_line_solve_host

  !> The distributed columns: pieces to the owners, banded solves, pieces back.
  subroutine solve_global(this, z, r, n)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=dp), dimension(n), intent(inout) :: z
    real(kind=dp), dimension(n), intent(in) :: r
    integer :: i, c, mg, e, L, o, p, idx, q, pos, lx, pstart

    lx = this%lx
    ! pack: y = wsq * mask * r along each chain in the global order
    do i = 1, this%nps
       c = this%ps_c(i)
       pos = this%ps_pos(i)
       do mg = 1, this%nseg(c)
          call gseg(this, c, mg, e, L, o)
          do p = 1, lx
             idx = node_index(lx, this%dir(e), L, pos_of(lx, o, p)) + (e - 1) * lx**3
             q = (mg - 1) * (lx - 1) + p
             this%sbuf(pos + q - 1) = this%wsq(idx) * this%mask(idx) * r(idx)
          end do
       end do
    end do
    call global_exchange_solve(this)
    ! unpack with the global skip rule (one write per unique node per column)
    do i = 1, this%nps
       c = this%ps_c(i)
       pos = this%ps_pos(i)
       do mg = 1, this%nseg(c)
          call gseg(this, c, mg, e, L, o)
          if (mg .eq. 1 .and. this%cg_off(c) .eq. 1) then
             pstart = 1
          else
             pstart = 2
          end if
          do p = pstart, lx
             idx = node_index(lx, this%dir(e), L, pos_of(lx, o, p)) + (e - 1) * lx**3
             q = (mg - 1) * (lx - 1) + p
             z(idx) = this%wsq(idx) * this%mask(idx) * this%sbuf(pos + q - 1)
          end do
       end do
    end do
  end subroutine solve_global

  !> The rank-to-owner exchange and the owners' band solves of the
  !! distributed columns, on the packed buffer sbuf (in: weighted
  !! residual pieces, out: solution pieces). Shared by the host and the
  !! device paths.
  subroutine global_exchange_solve(this)
    class(rs_line_t), intent(inout) :: this
    integer :: i, q, kl, ldab, info, ierr

    kl = this%lx - 1; ldab = kl + 1
    call MPI_Alltoallv(this%sbuf, this%sendcnt, this%senddsp, MPI_DOUBLE_PRECISION, &
         this%rbuf, this%recvcnt, this%recvdsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    do i = 1, this%npr
       do q = 1, this%pr_len(i)
          this%gy(this%hoff(this%pr_own(i)) + this%pr_q0(i) + q - 1) = this%rbuf(this%pr_pos(i) + q - 1)
       end do
    end do
    !$omp parallel do private(i, info) schedule(dynamic, 16)
    do i = 1, this%nown
       call dpbtrs('L', this%own_len(i), kl, 1, this%gcab(1, this%hoff(i) + 1), ldab, &
            this%gy(this%hoff(i) + 1), max(this%own_len(i), 1), info)
    end do
    !$omp end parallel do
    do i = 1, this%npr
       do q = 1, this%pr_len(i)
          this%rbuf(this%pr_pos(i) + q - 1) = this%gy(this%hoff(this%pr_own(i)) + this%pr_q0(i) + q - 1)
       end do
    end do
    call MPI_Alltoallv(this%rbuf, this%recvcnt, this%recvdsp, MPI_DOUBLE_PRECISION, &
         this%sbuf, this%sendcnt, this%senddsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
  end subroutine global_exchange_solve

  !> Schur-complement setup of the cut columns. For every send piece
  !! (chain c, global order): the interior band matrix (junction nodes
  !! dropped) is assembled from the line blocks and Cholesky-factored
  !! into cab(:,:,c) (the host path does not use cab of global chains);
  !! b and a are the couplings of the start / end junction to the first /
  !! last lx-1 interior entries, v = A_II^-1 b and w = A_II^-1 a the spike
  !! vectors; sbb = b.v, saa = a.w, sab = b.w, the direct coupling of the
  !! two junctions (one-segment pieces) and the end junction's assembled
  !! diagonal go to the owner once, who builds the tridiagonal junction
  !! system of each column.
  subroutine schur_setup(this)
    class(rs_line_t), intent(inout) :: this
    integer :: lx, kl, ldab, lx3, ip, c, nsg, len, lenI, s0, mg, e, L, og, p, q, qp, qi, qpi, pl, pl2
    integer :: idx, info, ntotp, k, ierr, j, mk, p0, jo, i
    real(kind=dp), allocatable :: ab(:,:), bf(:), af(:), v(:), w(:), sc(:), rc(:)
    integer, allocatable :: nodes(:)
    logical :: has_s, has_e
    real(kind=dp) :: sbb, saa, sab, dirc, ajj

    lx = this%lx; kl = lx - 1; ldab = kl + 1; lx3 = lx**3
    allocate(this%c2p(this%ncol)); this%c2p = 0
    allocate(this%pn_off(max(this%nps,1)), this%pn_len(max(this%nps,1)), this%pc_cl(max(this%nps,1)), &
         this%pc_jend(max(this%nps,1)), this%pc_b(max(kl * this%nps, 1)), this%pc_a(max(kl * this%nps, 1)))
    this%pc_b = 0.0_dp; this%pc_a = 0.0_dp; this%pc_jend = -1; this%pc_cl = 0
    ntotp = 0
    do ip = 1, this%nps
       c = this%ps_c(ip); this%c2p(c) = ip
       len = this%nseg(c) * (lx - 1) + 1
       s0 = 0; if (this%cg_off(c) .gt. 1) s0 = 1
       lenI = len - s0; if (.not. this%cg_last(c)) lenI = lenI - 1
       this%pn_off(ip) = ntotp; this%pn_len(ip) = lenI
       ntotp = ntotp + lenI
    end do
    this%npcor = ntotp
    allocate(this%pn_idx(max(ntotp,1)), this%pcor_idx(max(ntotp,1)), this%pcor_p(max(ntotp,1)), &
         this%pcor_v(max(ntotp,1)), this%pcor_w(max(ntotp,1)), sc(5 * max(this%nps,1)), &
         ab(ldab, this%maxn), bf(this%maxn), af(this%maxn), v(this%maxn), w(this%maxn), nodes(this%maxn))
    do ip = 1, this%nps
       c = this%ps_c(ip); nsg = this%nseg(c)
       len = nsg * (lx - 1) + 1
       has_s = this%cg_off(c) .gt. 1; has_e = .not. this%cg_last(c)
       s0 = 0; if (has_s) s0 = 1
       lenI = this%pn_len(ip)
       ab(:, 1:len) = 0.0_dp; bf(1:len) = 0.0_dp; af(1:len) = 0.0_dp
       dirc = 0.0_dp; ajj = 0.0_dp
       do mg = 1, nsg
          call gseg(this, c, mg, e, L, og)
          do p = 1, lx
             q = (mg - 1) * (lx - 1) + p
             pl = pos_of(lx, og, p)
             idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx3
             nodes(q) = idx
             qi = q - s0
             if (qi .ge. 1 .and. qi .le. lenI) then
                do i = 1, lx
                   qp = (mg - 1) * (lx - 1) + i
                   qpi = qp - s0
                   pl2 = pos_of(lx, og, i)
                   if (qpi .lt. 1 .or. qpi .gt. lenI .or. q .lt. qp) cycle
                   if (p .eq. i) then
                      ab(1, qpi) = this%blk(pl, pl, L, e)       ! assembled diagonal, set once
                   else
                      ab(1 + qi - qpi, qpi) = ab(1 + qi - qpi, qpi) + this%blk(pl, pl2, L, e)
                   end if
                end do
                if (has_s .and. mg .eq. 1) bf(qi) = this%blk(pos_of(lx, og, 1), pl, L, e)
                if (has_e .and. mg .eq. nsg) af(qi) = this%blk(pos_of(lx, og, lx), pl, L, e)
             end if
          end do
          if (has_s .and. has_e .and. nsg .eq. 1) dirc = this%blk(pos_of(lx, og, 1), pos_of(lx, og, lx), L, e)
          if (has_e .and. mg .eq. nsg) then
             ajj = this%blk(pos_of(lx, og, lx), pos_of(lx, og, lx), L, e)
             this%pc_jend(ip) = nodes(len) - 1
          end if
       end do
       do qi = 1, lenI
          this%pn_idx(this%pn_off(ip) + qi) = nodes(qi + s0) - 1
       end do
       if (lenI .gt. 0) then
          call dpbtrf('L', lenI, kl, ab, ldab, info)
          if (info .ne. 0) call neko_error('rs_line: dpbtrf failed (piece interior not SPD)')
          this%cab(:, this%cbo(c) + 1 : this%cbo(c) + lenI) = ab(:, 1:lenI)
          v(1:lenI) = bf(1:lenI); w(1:lenI) = af(1:lenI)
          call dpbtrs('L', lenI, kl, 1, ab, ldab, v, this%maxn, info)
          call dpbtrs('L', lenI, kl, 1, ab, ldab, w, this%maxn, info)
          sbb = dot_product(bf(1:lenI), v(1:lenI)); saa = dot_product(af(1:lenI), w(1:lenI))
          sab = dot_product(bf(1:lenI), w(1:lenI))
          do qi = 1, lenI
             idx = this%pn_idx(this%pn_off(ip) + qi) + 1
             this%pcor_idx(this%pn_off(ip) + qi) = idx - 1
             this%pcor_p(this%pn_off(ip) + qi) = ip - 1
             this%pcor_v(this%pn_off(ip) + qi) = this%wsq(idx) * this%mask(idx) * v(qi)
             this%pcor_w(this%pn_off(ip) + qi) = this%wsq(idx) * this%mask(idx) * w(qi)
          end do
          do p = 1, kl
             if (p .le. lenI) this%pc_b(kl * (ip - 1) + p) = bf(p)
             if (lenI - kl + p .ge. 1) this%pc_a(kl * (ip - 1) + p) = af(lenI - kl + p)
          end do
       else
          sbb = 0.0_dp; saa = 0.0_dp; sab = 0.0_dp
       end if
       sc(5 * (ip - 1) + 1) = sbb; sc(5 * (ip - 1) + 2) = saa; sc(5 * (ip - 1) + 3) = sab
       sc(5 * (ip - 1) + 4) = ajj; sc(5 * (ip - 1) + 5) = dirc
    end do
    allocate(rc(5 * max(this%npr, 1)), this%pbuf(3 * max(this%nps,1)), this%xjbuf(2 * max(this%nps,1)), &
         this%rsbuf(3 * max(this%npr,1)), this%xsbuf(2 * max(this%npr,1)))
    block
      real(kind=dp), allocatable :: ss(:)
      allocate(ss(5 * max(this%nps,1)))
      do ip = 1, this%nps
         ss(5 * this%ps_slot(ip) + 1 : 5 * this%ps_slot(ip) + 5) = sc(5 * (ip - 1) + 1 : 5 * (ip - 1) + 5)
      end do
      call MPI_Alltoallv(ss, 5 * this%pscnt, 5 * this%psdsp, MPI_DOUBLE_PRECISION, rc, 5 * this%prcnt, &
           5 * this%prdsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
      deallocate(ss)
    end block
    allocate(this%jun_off(max(this%nown,1) + 1))
    this%jun_off(1) = 0
    do j = 1, this%nown
       this%jun_off(j + 1) = this%jun_off(j) + max(this%own_np(j) - 1, 0)
    end do
    allocate(this%sd(max(this%jun_off(this%nown + 1), 1)), this%se(max(this%jun_off(this%nown + 1), 1)))
    do j = 1, this%nown
       mk = this%own_np(j); p0 = this%own_p0(j); jo = this%jun_off(j)
       do k = 1, mk - 1
          this%sd(jo + k) = rc(5 * this%pr_slot(p0 + k - 1) + 4) - rc(5 * this%pr_slot(p0 + k - 1) + 2) &
               - rc(5 * this%pr_slot(p0 + k) + 1)
          this%se(jo + k) = rc(5 * this%pr_slot(p0 + k) + 5) - rc(5 * this%pr_slot(p0 + k) + 3)
       end do
    end do
    deallocate(ab, bf, af, v, w, sc, rc, nodes)
  end subroutine schur_setup

  !> Per application: junction scalars (s_start, s_end, r_Jend) of every
  !! sent piece in pbuf (slot order) -> owners; each owned column's
  !! tridiagonal junction system; the junction values (start, end) of
  !! every piece back into xjbuf.
  subroutine schur_exchange_solve(this)
    class(rs_line_t), intent(inout), target :: this
    integer :: j, mk, p0, jo, k, s
    real(kind=dp), allocatable :: rhs(:), dd(:), xj(:)
    real(kind=dp) :: m

    call schur_p2p(this, c_loc(this%pbuf), c_loc(this%rsbuf), 3, 1)
    allocate(rhs(max(this%gmaxn, 1)), dd(max(this%gmaxn, 1)), xj(max(this%gmaxn, 1)))
    this%xsbuf = 0.0_dp
    do j = 1, this%nown
       mk = this%own_np(j); p0 = this%own_p0(j); jo = this%jun_off(j)
       do k = 1, mk - 1
          s = this%pr_slot(p0 + k - 1)
          rhs(k) = this%rsbuf(3 * s + 3) - this%rsbuf(3 * s + 2) - this%rsbuf(3 * this%pr_slot(p0 + k) + 1)
          dd(k) = this%sd(jo + k)
       end do
       do k = 2, mk - 1
          m = this%se(jo + k - 1) / dd(k - 1)
          dd(k) = dd(k) - m * this%se(jo + k - 1)
          rhs(k) = rhs(k) - m * rhs(k - 1)
       end do
       if (mk .gt. 1) xj(mk - 1) = rhs(mk - 1) / dd(mk - 1)
       do k = mk - 2, 1, -1
          xj(k) = (rhs(k) - this%se(jo + k) * xj(k + 1)) / dd(k)
       end do
       do k = 1, mk
          s = this%pr_slot(p0 + k - 1)
          if (k .gt. 1) this%xsbuf(2 * s + 1) = xj(k - 1)
          if (k .lt. mk) this%xsbuf(2 * s + 2) = xj(k)
       end do
    end do
    deallocate(rhs, dd, xj)
    call schur_p2p(this, c_loc(this%xsbuf), c_loc(this%xjbuf), 2, 2)
  end subroutine schur_exchange_solve

  !> Point-to-point exchange of m scalars per piece with the neighbours
  !! only (no collective): dir 1 pieces -> owners (sb in send-slot order,
  !! rb in receive-slot order), dir 2 owners -> pieces. sb / rb are host
  !! or, with schur_dmpi, device addresses.
  subroutine schur_p2p(this, sb, rb, m, dir)
    class(rs_line_t), intent(inout) :: this
    type(c_ptr), intent(in) :: sb, rb
    integer, intent(in) :: m, dir
    integer :: i, r, nq, b, tag

    b = m * (storage_size(1.0_dp) / 8)
    tag = 7100 + dir
    nq = 0
    if (dir .eq. 1) then
       do i = 1, this%nrnb
          r = this%rnb(i); nq = nq + 1
          call rs_mpi_irecv(rb, int(b * this%prdsp(r + 1), c_int), int(b * this%prcnt(r + 1), c_int), &
               int(r, c_int), int(tag, c_int), this%schur_reqs, int(nq, c_int))
       end do
       do i = 1, this%nsnb
          r = this%snb(i); nq = nq + 1
          call rs_mpi_isend(sb, int(b * this%psdsp(r + 1), c_int), int(b * this%pscnt(r + 1), c_int), &
               int(r, c_int), int(tag, c_int), this%schur_reqs, int(nq, c_int))
       end do
    else
       do i = 1, this%nsnb
          r = this%snb(i); nq = nq + 1
          call rs_mpi_irecv(rb, int(b * this%psdsp(r + 1), c_int), int(b * this%pscnt(r + 1), c_int), &
               int(r, c_int), int(tag, c_int), this%schur_reqs, int(nq, c_int))
       end do
       do i = 1, this%nrnb
          r = this%rnb(i); nq = nq + 1
          call rs_mpi_isend(sb, int(b * this%prdsp(r + 1), c_int), int(b * this%prcnt(r + 1), c_int), &
               int(r, c_int), int(tag, c_int), this%schur_reqs, int(nq, c_int))
       end do
    end if
    if (nq .gt. 0) call rs_mpi_waitall(int(nq, c_int), this%schur_reqs)
  end subroutine schur_p2p

  !> Device variant of global_exchange_solve: the owners' band solves run
  !! on the device (fill / solve / unfill kernels), the host only holds the
  !! packed MPI buffers.
  subroutine global_exchange_solve_device(this, td)
    class(rs_line_t), intent(inout) :: this
    real(kind=dp), intent(inout) :: td(0:9)   ! td(3..6) set here when the timers are on
    integer :: ierr

    call MPI_Alltoallv(this%sbuf, this%sendcnt, this%senddsp, MPI_DOUBLE_PRECISION, &
         this%rbuf, this%recvcnt, this%recvdsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    if (this%timers) td(3) = MPI_Wtime()
    if (this%gown_dev) then
       call device_memcpy(this%rbuf, this%rbuf_d, this%nrb, HOST_TO_DEVICE, sync = .true.)
       if (this%timers) td(4) = MPI_Wtime()
       call device_sync()
       call rs_line_cuda_gfill(this%gyw_d, this%rbuf_d, this%rb_map_d, int(this%nrb, c_int))
       if (this%group_kernel) then
          call rs_line_cuda_gsolve_group(this%gyw_d, this%gcabf_d, this%gcabb_d, this%goff_d, &
               this%glen_d, int(this%nown, c_int), int(this%lx, c_int), &
               int(merge(1, 0, this%fac_sp), c_int))
       else
          call rs_line_cuda_gsolve(this%gyw_d, this%gcabd_d, this%glen_d, int(this%nown, c_int), &
               int(this%lx, c_int))
       end if
       call rs_line_cuda_gunfill(this%rbuf_d, this%gyw_d, this%rb_map_d, int(this%nrb, c_int))
       call rs_line_cuda_sync()
       if (this%timers) td(5) = MPI_Wtime()
       call device_memcpy(this%rbuf, this%rbuf_d, this%nrb, DEVICE_TO_HOST, sync = .true.)
    end if
    if (this%timers) then
       if (.not. this%gown_dev) then
          td(4) = td(3); td(5) = td(3)
       end if
       td(6) = MPI_Wtime()
    end if
    call MPI_Alltoallv(this%rbuf, this%recvcnt, this%recvdsp, MPI_DOUBLE_PRECISION, &
         this%sbuf, this%sendcnt, this%senddsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
  end subroutine global_exchange_solve_device

  !> Device copy of a group-kernel factor array, double or (single-precision
  !! factor storage) float, allocated and copied by bytes: not associated
  !! with the host array, which release_host can then free.
  subroutine upload_fac(a, a_d, fsp)
    real(kind=dp), intent(inout), target :: a(:)
    type(c_ptr), intent(inout) :: a_d
    logical, intent(in) :: fsp
    real(kind=sp), allocatable, target :: as(:)
    integer(c_size_t) :: nbyt
    a_d = c_null_ptr
    if (fsp) then
       allocate(as(size(a))); as = real(a, sp)
       nbyt = int(size(as, kind=i8), c_size_t) * 4_c_size_t
       call device_alloc(a_d, nbyt)
       call rs_line_cuda_copy(a_d, c_loc(as), nbyt, 1_c_int)
       deallocate(as)
    else
       nbyt = int(size(a, kind=i8), c_size_t) * 8_c_size_t
       call device_alloc(a_d, nbyt)
       call rs_line_cuda_copy(a_d, c_loc(a), nbyt, 1_c_int)
    end if
  end subroutine upload_fac

  subroutine free_fac(a_d)
    type(c_ptr), intent(inout) :: a_d
    if (c_associated(a_d)) call device_free(a_d)
    a_d = c_null_ptr
  end subroutine free_fac

  !> GPU runs: free the host data that only the setup and the one-time
  !! device check (RS_LINE_CHECK) read -- band factors (cab, gcab), their
  !! group-kernel layouts (cabf, cabb, gcabf, gcabb, uploaded by bytes) and
  !! the line blocks (blk; read by rs_colagg's setup, so the caller frees
  !! after the whole hierarchy is built). Several GB per rank on the airfoil.
  subroutine rs_line_release_host(this)
    class(rs_line_t), intent(inout) :: this
    if (NEKO_BCKND_DEVICE .ne. 1 .or. this%check_device) return
    if (allocated(this%cab)) deallocate(this%cab)
    if (allocated(this%gcab)) deallocate(this%gcab)
    if (allocated(this%cabf)) deallocate(this%cabf)
    if (allocated(this%cabb)) deallocate(this%cabb)
    if (allocated(this%gcabf)) deallocate(this%gcabf)
    if (allocated(this%gcabb)) deallocate(this%gcabb)
    if (allocated(this%blk)) deallocate(this%blk)
    if (allocated(this%lu)) deallocate(this%lu)
  end subroutine rs_line_release_host

  !> Host memory report (collective): current (VmRSS) and peak (VmHWM)
  !! resident memory of this process, maxima over the ranks, printed by
  !! rank 0 as 'RS_MEM <tag>'. Only where every rank passes.
  subroutine rs_mem_report(tag)
    character(len=*), intent(in) :: tag
    integer(kind=i8) :: kb(2)
    integer :: u, ios, ierr
    character(len=256) :: line
    kb = 0_i8
    open(newunit=u, file='/proc/self/status', action='read', iostat=ios)
    if (ios .eq. 0) then
       do
          read(u, '(A)', iostat=ios) line
          if (ios .ne. 0) exit
          if (line(1:6) .eq. 'VmRSS:') read(line(7:), *, iostat=ios) kb(1)
          if (line(1:6) .eq. 'VmHWM:') read(line(7:), *, iostat=ios) kb(2)
       end do
       close(u)
    end if
    call MPI_Allreduce(MPI_IN_PLACE, kb, 2, MPI_INTEGER8, MPI_MAX, NEKO_COMM, ierr)
    if (pe_rank .eq. 0) write(*, '(A,A,A,F8.2,A,F8.2,A)') 'RS_MEM ', tag, ': rss ', &
         real(kb(1), dp) / 1048576.0_dp, ' GB, peak ', real(kb(2), dp) / 1048576.0_dp, &
         ' GB (max over ranks)'
  end subroutine rs_mem_report

  !> Gather-scatter / projector on a host array, also in device mode
  !! (temporary device mirror).
  subroutine hgs(gs_h, a, n, op)
    type(gs_t), intent(inout) :: gs_h
    real(kind=dp), intent(inout) :: a(:)
    integer, intent(in) :: n, op
    real(kind=rp), allocatable, target :: ar(:)
    type(c_ptr) :: a_d
    ! through a copy in Neko's precision (setup and checks only)
    allocate(ar(n)); ar = real(a(1:n), rp)
    if (NEKO_BCKND_DEVICE .eq. 1) then
       a_d = c_null_ptr
       call device_map(ar, a_d, n)
       call device_memcpy(ar, a_d, n, HOST_TO_DEVICE, sync = .true.)
       call gs_h%op(ar, n, op)
       call device_memcpy(ar, a_d, n, DEVICE_TO_HOST, sync = .true.)
       call device_deassociate(ar)
       call device_free(a_d)
    else
       call gs_h%op(ar, n, op)
    end if
    a(1:n) = ar; deallocate(ar)
  end subroutine hgs

  subroutine hbcp(bcp, a, n)
    class(scalar_bc_projector_t), intent(inout) :: bcp
    real(kind=dp), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(kind=rp), allocatable, target :: ar(:)
    type(c_ptr) :: a_d
    allocate(ar(n)); ar = real(a(1:n), rp)
    if (NEKO_BCKND_DEVICE .eq. 1) then
       a_d = c_null_ptr
       call device_map(ar, a_d, n)
       call device_memcpy(ar, a_d, n, HOST_TO_DEVICE, sync = .true.)
       call bcp%apply(ar, n)
       call device_memcpy(ar, a_d, n, DEVICE_TO_HOST, sync = .true.)
       call device_deassociate(ar)
       call device_free(a_d)
    else
       call bcp%apply(ar, n)
    end if
    a(1:n) = ar; deallocate(ar)
  end subroutine hbcp

  !> Pack the rank-local columns for the device kernel and copy the data.
  !> Pin / unpin a host array for the device copies (target dummies, so
  !! that c_loc is legal on allocatable components of a non-target object)
  !> Release the device copy of a 2D host array mapped through a 1D view.
  subroutine unmap_2d(a, a_d)
    real(kind=dp), intent(inout), target :: a(:,:)
    type(c_ptr), intent(inout) :: a_d
    real(kind=dp), pointer :: af(:)
    call c_f_pointer(c_loc(a), af, [size(a)])
    call device_deassociate(af); call device_free(a_d)
  end subroutine unmap_2d

  !> Byte offset of a device address
  function ptr_add(p, off) result(q)
    type(c_ptr), intent(in) :: p
    integer(c_size_t), intent(in) :: off
    type(c_ptr) :: q
    q = transfer(transfer(p, 0_c_intptr_t) + int(off, c_intptr_t), q)
  end function ptr_add

  subroutine pin_arr(a, n)
    real(kind=dp), intent(in), target :: a(:)
    integer, intent(in) :: n
    character(len=8) :: env
    integer :: istat
    call get_environment_variable('RS_LINE_NOPIN', env, status = istat)
    if (istat .eq. 0) return   ! diagnostics: leave the buffers pageable
    call rs_line_cuda_pin(c_loc(a), int(n, c_size_t) * int(storage_size(1.0_dp) / 8, c_size_t))
  end subroutine pin_arr

  subroutine unpin_arr(a)
    real(kind=dp), intent(in), target :: a(:)
    call rs_line_cuda_unpin(c_loc(a))
  end subroutine unpin_arr

  subroutine rs_line_device_setup(this)
    class(rs_line_t), intent(inout), target :: this
    integer :: lx, c, cl, m, e, L, o, p, pl, q, idx, k, j, ntot, pstart, n, i, cb, ce
    integer, allocatable :: loccol(:)

    real(kind=dp) :: td(0:3)

    lx = this%lx; n = this%n
    td(0) = MPI_Wtime()
    if (.not. this%columns) call neko_error('rs_line: the device port needs line_columns')
    if (lx .gt. 8) call neko_error('rs_line: the CUDA kernels assume lx <= 8')
    if (rs_line_cuda_real_bytes() .ne. storage_size(1.0_rp) / 8) call neko_error( &
         'rs_line: the GPU kernels were compiled for another real kind than Neko''s rp ' // &
         '(compile rs_line_cuda.cu with -DRS_REAL_SP for a single-precision Neko)')
    if (rp .ne. dp) then
       call device_alloc(this%rw_d, int(this%n, c_size_t) * 8_c_size_t)
       call device_alloc(this%zw_d, int(this%n, c_size_t) * 8_c_size_t)
    end if
    block
      character(len=8) :: envg
      integer :: istg
      call get_environment_variable('RS_LINE_GROUP', envg, status = istg)
      this%group_kernel = .true.
      if (istg .eq. 0 .and. envg(1:1) .eq. '0') this%group_kernel = .false.
    end block
    ! Schur-complement exchange of the cut columns: the pieces become
    ! local columns (interiors) and the exchange carries scalars only
    this%schur = .false.
    if (this%group_kernel .and. this%global_columns .and. this%nglob .gt. 0 .and. pe_size .gt. 1) then
       block
         character(len=8) :: envs
         integer :: ists
         this%schur = .true.
         call get_environment_variable('RS_LINE_SCHUR', envs, status = ists)
         if (ists .eq. 0 .and. envs(1:1) .eq. '0') this%schur = .false.
       end block
    end if
    if (this%schur) call schur_setup(this)
    this%ncol_loc = 0; ntot = 0
    do c = 1, this%ncol
       if (this%cg_id(c) .ne. 0 .and. .not. this%schur) cycle
       this%ncol_loc = this%ncol_loc + 1
       if (this%schur .and. this%cg_id(c) .ne. 0) then
          ntot = ntot + this%pn_len(this%c2p(c))
       else
          ntot = ntot + this%nseg(c) * (lx - 1) + 1
       end if
    end do
    allocate(this%cnode(max(ntot, 1)), this%cwrite(max(ntot, 1)), &
         this%coff(max(this%ncol_loc, 1)), this%clen(max(this%ncol_loc, 1)))
    if (this%group_kernel) then
       ! group kernel: factors in the blocked layout (cabf / cabb below),
       ! work vector of 8 x (block rows) entries; cabd stays a dummy
       ! (ywork is sized once the blocks are known, below)
       allocate(this%cabd(1), this%ywork(1))
    else
       if (int(this%ncol_loc, i8) * int(lx, i8) * int(this%maxn, i8) .ge. 2_i8**31) &
            call neko_error('rs_line: packed factors exceed 2^31 entries (split the columns)')
       allocate(this%cabd(max(this%ncol_loc * lx * this%maxn, 1)), &
            this%ywork(max(this%ncol_loc * this%maxn, 1)))
    end if
    allocate(this%isocoef(n), this%zh(n), loccol(max(this%ncol_loc, 1)))
    ! no zero-fill: the kernel reads only the entries the packing writes
    cl = 0; ntot = 0
    do c = 1, this%ncol
       if (this%cg_id(c) .ne. 0 .and. .not. this%schur) cycle
       cl = cl + 1
       this%coff(cl) = ntot
       if (this%schur .and. this%cg_id(c) .ne. 0) then
          this%clen(cl) = this%pn_len(this%c2p(c))
       else
          this%clen(cl) = this%nseg(c) * (lx - 1) + 1
       end if
       loccol(cl) = c
       ntot = ntot + this%clen(cl)
    end do
    if (this%group_kernel .and. this%ncol_loc .gt. 0) then
       ! group kernel: columns sorted by length (descending) so that the
       ! blocks of eight (one wavefront) carry little padding; block row
       ! offsets cboff
       block
         integer(kind=i8), allocatable :: keyl(:)
         integer, allocatable :: perml(:), loc2(:), len2(:)
         integer :: nblk, b
         allocate(keyl(this%ncol_loc), perml(this%ncol_loc), loc2(this%ncol_loc), len2(this%ncol_loc))
         do cl = 1, this%ncol_loc
            keyl(cl) = -int(this%clen(cl), i8)
         end do
         call sort_ki(this%ncol_loc, keyl, perml)
         do cl = 1, this%ncol_loc
            loc2(cl) = loccol(perml(cl)); len2(cl) = this%clen(perml(cl))
         end do
         ntot = 0
         do cl = 1, this%ncol_loc
            loccol(cl) = loc2(cl); this%clen(cl) = len2(cl)
            this%coff(cl) = ntot
            ntot = ntot + this%clen(cl)
         end do
         nblk = (this%ncol_loc + 7) / 8
         allocate(this%cboff(nblk + 1))
         this%cboff(1) = 0
         do b = 1, nblk
            this%cboff(b + 1) = this%cboff(b) + this%clen(8 * (b - 1) + 1)   ! the block's longest
         end do
         if (int(this%cboff(nblk + 1), i8) * 8_i8 * int(lx, i8) .ge. 2_i8**31) &
              call neko_error('rs_line: blocked group factors exceed 2^31 entries')
         deallocate(this%ywork)
         allocate(this%ywork(max(8 * this%cboff(nblk + 1), 1)))
         deallocate(keyl, perml, loc2, len2)
       end block
    end if
    !$omp parallel do private(cl, c, m, e, L, o, pstart, p, q, pl, idx) schedule(static)
    do cl = 1, this%ncol_loc
       c = loccol(cl)
       if (this%schur .and. this%cg_id(c) .ne. 0) then
          ! a piece interior: its node list is ready (global order, junctions excluded)
          do q = 1, this%clen(cl)
             this%cnode(this%coff(cl) + q) = this%pn_idx(this%pn_off(this%c2p(c)) + q)
             this%cwrite(this%coff(cl) + q) = 1
          end do
          this%pc_cl(this%c2p(c)) = cl - 1
          cycle
       end if
       do m = 1, this%nseg(c)
          e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
          if (m .eq. 1) then
             pstart = 1
          else
             pstart = 2
          end if
          ! the junction entry q of consecutive segments is one unique
          ! node with two local copies: keep the copy of the earlier
          ! segment (p = 1 of segment m > 1 is skipped) and write every
          ! entry exactly once
          do p = pstart, lx
             q = (m - 1) * (lx - 1) + p
             pl = pos_of(lx, o, p)
             idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx**3
             this%cnode(this%coff(cl) + q) = idx - 1
             this%cwrite(this%coff(cl) + q) = 1
          end do
       end do
    end do
    !$omp end parallel do
    td(1) = MPI_Wtime()
    ! band factors: cabd[(j*lx + k)*ncol + c] (0-based) = cab(k+1, j+1, c),
    ! written in blocks of columns so that both sides stay cache-friendly
    if (.not. this%group_kernel) then
    !$omp parallel private(cb, ce, cl, j, k)
    block
      real(kind=dp), allocatable :: tmp(:,:,:)
      allocate(tmp(lx, this%maxn, 64))
      !$omp do schedule(static)
      do cb = 1, this%ncol_loc, 64
         ce = min(cb + 63, this%ncol_loc)
         do cl = cb, ce
            tmp(:, :, cl - cb + 1) = 0.0_dp
            tmp(:, 1:this%clen(cl), cl - cb + 1) = &
                 this%cab(:, this%cbo(loccol(cl)) + 1 : this%cbo(loccol(cl)) + this%clen(cl))
         end do
         do j = 1, this%maxn
            do k = 1, lx
               do cl = cb, ce
                  this%cabd(((j - 1) * lx + (k - 1)) * this%ncol_loc + cl) = tmp(k, j, cl - cb + 1)
               end do
            end do
         end do
      end do
      !$omp end do
      deallocate(tmp)
    end block
    !$omp end parallel
    end if
    ! group kernel layouts (no padding): cabf(off*lx + j*lx + t), t < kd:
    ! L(j, j-kd+t) = cab(1+kd-t, j-kd+t+1), t = kd: L(j,j) = cab(1, j+1);
    ! cabb: t < kd: L(j+1+t, j) = cab(2+t, j+1); zero outside the column
    block
      integer :: jj, tt, ii, nn, yo
      if (this%group_kernel) then
         allocate(this%cabf(size(this%ywork) * lx), this%cabb(size(this%ywork) * lx))
         !$omp parallel do private(cl, c, nn, jj, tt, ii, yo) schedule(static)
         do cl = 1, this%ncol_loc
            c = loccol(cl); nn = this%clen(cl)
            yo = this%cboff((cl - 1) / 8 + 1) * 8 + mod(cl - 1, 8)   ! work entry of row 0 (0-based)
            do jj = 0, nn - 1
               do tt = 0, lx - 1
                  ii = jj - (lx - 1) + tt
                  if (tt .eq. lx - 1) then
                     ! the kernel multiplies by the inverse diagonal
                     this%cabf((yo + jj * 8) * lx + tt + 1) = 1.0_dp / this%cab(1, this%cbo(c) + jj + 1)
                     this%cabb((yo + jj * 8) * lx + tt + 1) = 1.0_dp / this%cab(1, this%cbo(c) + jj + 1)
                  else
                     if (ii .ge. 0) then
                        this%cabf((yo + jj * 8) * lx + tt + 1) = this%cab(1 + (lx - 1) - tt, this%cbo(c) + ii + 1)
                     else
                        this%cabf((yo + jj * 8) * lx + tt + 1) = 0.0_dp
                     end if
                     if (jj + 1 + tt .lt. nn) then
                        this%cabb((yo + jj * 8) * lx + tt + 1) = this%cab(2 + tt, this%cbo(c) + jj + 1)
                     else
                        this%cabb((yo + jj * 8) * lx + tt + 1) = 0.0_dp
                     end if
                  end if
               end do
            end do
         end do
         !$omp end parallel do
      end if
    end block
    td(2) = MPI_Wtime()
    do i = 1, n
       this%isocoef(i) = this%jmask(i) * this%wsq(i)**2 * this%mask(i) / this%diag(i)
    end do
    this%zh = 0.0_dp
    call device_map(this%cnode, this%cnode_d, size(this%cnode))
    call device_memcpy(this%cnode, this%cnode_d, size(this%cnode), HOST_TO_DEVICE, sync = .true.)
    call device_map(this%cwrite, this%cwrite_d, size(this%cwrite))
    call device_memcpy(this%cwrite, this%cwrite_d, size(this%cwrite), HOST_TO_DEVICE, sync = .true.)
    call device_map(this%coff, this%coff_d, size(this%coff))
    call device_memcpy(this%coff, this%coff_d, size(this%coff), HOST_TO_DEVICE, sync = .true.)
    call device_map(this%clen, this%clen_d, size(this%clen))
    call device_memcpy(this%clen, this%clen_d, size(this%clen), HOST_TO_DEVICE, sync = .true.)
    call device_map(this%cabd, this%cabd_d, size(this%cabd))
    call device_memcpy(this%cabd, this%cabd_d, size(this%cabd), HOST_TO_DEVICE, sync = .true.)
    call device_map(this%ywork, this%ywork_d, size(this%ywork))
    if (this%group_kernel) then
       call upload_fac(this%cabf, this%cabf_d, this%fac_sp)
       call upload_fac(this%cabb, this%cabb_d, this%fac_sp)
       if (allocated(this%cboff)) then
          call device_map(this%cboff, this%cboff_d, size(this%cboff))
          call device_memcpy(this%cboff, this%cboff_d, size(this%cboff), HOST_TO_DEVICE, sync = .true.)
       end if
    end if
    call device_map(this%isocoef, this%isocoef_d, n)
    call device_memcpy(this%isocoef, this%isocoef_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%wsq, this%wsq_d, n)
    call device_memcpy(this%wsq, this%wsq_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%mask, this%mask_d, n)
    call device_memcpy(this%mask, this%mask_d, n, HOST_TO_DEVICE, sync = .true.)
    allocate(this%wm(n)); this%wm = this%wsq * this%mask
    call device_map(this%wm, this%wm_d, n)
    call device_memcpy(this%wm, this%wm_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%zh, this%zh_d, n)
    call device_memcpy(this%zh, this%zh_d, n, HOST_TO_DEVICE, sync = .true.)
    deallocate(loccol)
    ! distributed columns: the sbuf slots' nodes and write flags (the same
    ! loops as solve_global's pack / unpack; a slot shared by two segments,
    ! the junction, is represented by the earlier segment's end node)
    this%ngp = 0
    if (allocated(this%sendcnt) .and. .not. this%schur) this%ngp = sum(this%sendcnt)
    allocate(this%gp_idx(max(this%ngp, 1)), this%gp_wr(max(this%ngp, 1)))
    this%gp_idx = 0; this%gp_wr = 0
    if (this%ngp .gt. 0) then
    block
      integer :: ii, mg, og, pos, pstart_g
      do ii = 1, this%nps
         c = this%ps_c(ii)
         pos = this%ps_pos(ii)
         do mg = 1, this%nseg(c)
            call gseg(this, c, mg, e, L, og)
            if (mg .eq. 1 .and. this%cg_off(c) .eq. 1) then
               pstart_g = 1
            else
               pstart_g = 2
            end if
            do p = 1, lx
               if (mg .gt. 1 .and. p .eq. 1) cycle
               idx = node_index(lx, this%dir(e), L, pos_of(lx, og, p)) + (e - 1) * lx**3
               q = (mg - 1) * (lx - 1) + p
               this%gp_idx(pos + q - 1) = idx - 1
               if (p .ge. pstart_g) this%gp_wr(pos + q - 1) = 1
            end do
         end do
      end do
    end block
    end if
    if (this%ngp .gt. 0) then
       call device_map(this%gp_idx, this%gp_idx_d, this%ngp)
       call device_memcpy(this%gp_idx, this%gp_idx_d, this%ngp, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%gp_wr, this%gp_wr_d, this%ngp)
       call device_memcpy(this%gp_wr, this%gp_wr_d, this%ngp, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%sbuf, this%sbuf_d, this%ngp)
       call pin_arr(this%sbuf, this%ngp)
       this%sbuf_pinned = .true.
    end if
    ! owner side of the distributed columns on the device: the received
    ! pieces go rbuf -> packed gy (rb_map), one thread per owned column
    ! solves its band system (factors packed like cabd), and back
    this%nrb = 0; this%gown_dev = .false.
    if (allocated(this%recvcnt)) this%nrb = sum(this%recvcnt)
    if (this%nown .gt. 0 .and. this%nrb .gt. 0 .and. .not. this%schur) then
       allocate(this%glen(this%nown), this%goff(this%nown), this%rb_map(this%nrb))
       this%glen = this%own_len(1:this%nown)
       block
         integer :: ii, qq, j, kk, gtot, jj, tt, nn
         gtot = 0
         do c = 1, this%nown
            this%goff(c) = gtot
            gtot = gtot + this%own_len(c)
         end do
         if (this%group_kernel) then
            ! rb_map -> goff + q; factors in the cabf / cabb layouts
            do ii = 1, this%npr
               do qq = 1, this%pr_len(ii)
                  this%rb_map(this%pr_pos(ii) + qq - 1) = &
                       this%goff(this%pr_own(ii)) + (this%pr_q0(ii) + qq - 2)
               end do
            end do
            if (int(gtot, i8) * int(lx, i8) .ge. 2_i8**31) &
                 call neko_error('rs_line: packed factors of the owned global columns exceed 2^31 entries')
            allocate(this%gcabd(1), this%gyw(max(gtot, 1)), this%gcabf(max(gtot * lx, 1)), &
                 this%gcabb(max(gtot * lx, 1)))
            !$omp parallel do private(c, nn, jj, tt, ii) schedule(static)
            do c = 1, this%nown
               nn = this%own_len(c)
               do jj = 0, nn - 1
                  do tt = 0, lx - 1
                     ii = jj - (lx - 1) + tt
                     if (tt .eq. lx - 1) then
                        this%gcabf(this%goff(c) * lx + jj * lx + tt + 1) = 1.0_dp / this%gcab(1, this%hoff(c) + jj + 1)
                        this%gcabb(this%goff(c) * lx + jj * lx + tt + 1) = 1.0_dp / this%gcab(1, this%hoff(c) + jj + 1)
                     else
                        if (ii .ge. 0) then
                           this%gcabf(this%goff(c) * lx + jj * lx + tt + 1) = this%gcab(1 + (lx - 1) - tt, this%hoff(c) + ii + 1)
                        else
                           this%gcabf(this%goff(c) * lx + jj * lx + tt + 1) = 0.0_dp
                        end if
                        if (jj + 1 + tt .lt. nn) then
                           this%gcabb(this%goff(c) * lx + jj * lx + tt + 1) = this%gcab(2 + tt, this%hoff(c) + jj + 1)
                        else
                           this%gcabb(this%goff(c) * lx + jj * lx + tt + 1) = 0.0_dp
                        end if
                     end if
                  end do
               end do
            end do
            !$omp end parallel do
         else
            if (int(this%nown, i8) * int(lx, i8) * int(this%gmaxn, i8) .ge. 2_i8**31) &
                 call neko_error('rs_line: packed factors of the owned global columns exceed 2^31 entries')
            allocate(this%gcabd(this%nown * lx * this%gmaxn), this%gyw(this%nown * this%gmaxn), &
                 this%gcabf(1), this%gcabb(1))
            do ii = 1, this%npr
               do qq = 1, this%pr_len(ii)
                  this%rb_map(this%pr_pos(ii) + qq - 1) = &
                       (this%pr_q0(ii) + qq - 2) * this%nown + (this%pr_own(ii) - 1)
               end do
            end do
            !$omp parallel do private(c, j, kk) schedule(static)
            do c = 1, this%nown
               do j = 1, this%own_len(c)
                  do kk = 1, lx
                     this%gcabd(((j - 1) * lx + (kk - 1)) * this%nown + c) = this%gcab(kk, this%hoff(c) + j)
                  end do
               end do
            end do
            !$omp end parallel do
         end if
       end block
       this%gyw = 0.0_dp
       call device_map(this%rb_map, this%rb_map_d, this%nrb)
       call device_memcpy(this%rb_map, this%rb_map_d, this%nrb, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%glen, this%glen_d, this%nown)
       call device_memcpy(this%glen, this%glen_d, this%nown, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%gcabd, this%gcabd_d, size(this%gcabd))
       call device_memcpy(this%gcabd, this%gcabd_d, size(this%gcabd), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%goff, this%goff_d, this%nown)
       call device_memcpy(this%goff, this%goff_d, this%nown, HOST_TO_DEVICE, sync = .true.)
       call upload_fac(this%gcabf, this%gcabf_d, this%fac_sp)
       call upload_fac(this%gcabb, this%gcabb_d, this%fac_sp)
       call device_map(this%gyw, this%gyw_d, size(this%gyw))
       call device_memcpy(this%gyw, this%gyw_d, size(this%gyw), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%rbuf, this%rbuf_d, this%nrb)
       call pin_arr(this%rbuf, this%nrb)
       this%rbuf_pinned = .true.
       this%gown_dev = .true.
    end if
    if (this%schur) then
       call device_map(this%pc_cl, this%pc_cl_d, this%nps)
       call device_memcpy(this%pc_cl, this%pc_cl_d, this%nps, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pc_jend, this%pc_jend_d, this%nps)
       call device_memcpy(this%pc_jend, this%pc_jend_d, this%nps, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%ps_slot, this%ps_slot_d, this%nps)
       call device_memcpy(this%ps_slot, this%ps_slot_d, this%nps, HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pc_b, this%pc_b_d, size(this%pc_b))
       call device_memcpy(this%pc_b, this%pc_b_d, size(this%pc_b), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pc_a, this%pc_a_d, size(this%pc_a))
       call device_memcpy(this%pc_a, this%pc_a_d, size(this%pc_a), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pcor_idx, this%pcor_idx_d, size(this%pcor_idx))
       call device_memcpy(this%pcor_idx, this%pcor_idx_d, size(this%pcor_idx), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pcor_p, this%pcor_p_d, size(this%pcor_p))
       call device_memcpy(this%pcor_p, this%pcor_p_d, size(this%pcor_p), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pcor_v, this%pcor_v_d, size(this%pcor_v))
       call device_memcpy(this%pcor_v, this%pcor_v_d, size(this%pcor_v), HOST_TO_DEVICE, sync = .true.)
       call device_map(this%pcor_w, this%pcor_w_d, size(this%pcor_w))
       call device_memcpy(this%pcor_w, this%pcor_w_d, size(this%pcor_w), HOST_TO_DEVICE, sync = .true.)
       ! neighbours of the per-call exchange, and where it runs
       block
         character(len=8) :: envm
         integer :: istm, r, ierr, nj
         logical :: dm
         this%nsnb = count(this%pscnt .gt. 0); this%nrnb = count(this%prcnt .gt. 0)
         allocate(this%snb(max(this%nsnb, 1)), this%rnb(max(this%nrnb, 1)))
         this%nsnb = 0; this%nrnb = 0
         do r = 1, pe_size
            if (this%pscnt(r) .gt. 0) then
               this%nsnb = this%nsnb + 1; this%snb(this%nsnb) = r - 1
            end if
            if (this%prcnt(r) .gt. 0) then
               this%nrnb = this%nrnb + 1; this%rnb(this%nrnb) = r - 1
            end if
         end do
         call rs_mpi_init_reqs(int(max(this%nsnb + this%nrnb, 1), c_int), this%schur_reqs)
         dm = .false.
         select type (cm => this%gs_h%comm)
         type is (gs_device_mpi_t)
            dm = .true.
         end select
         call get_environment_variable('RS_LINE_DMPI', envm, status = istm)
         if (istm .eq. 0) dm = (envm(1:1) .eq. '1')
         call MPI_Allreduce(MPI_IN_PLACE, dm, 1, MPI_LOGICAL, MPI_LAND, NEKO_COMM, ierr)
         this%schur_dmpi = dm
         if (dm) then
            nj = max(this%jun_off(this%nown + 1), 1)
            allocate(this%own_p0z(max(this%nown, 1)), this%sdd(nj), this%sxj(nj))
            this%own_p0z = 0
            if (this%nown .gt. 0) this%own_p0z(1:this%nown) = this%own_p0(1:this%nown) - 1
            this%sdd = 0.0_dp; this%sxj = 0.0_dp
            ! the four exchange buffers in one device allocation padded to
            ! whole 2 MiB pages: UCX registers 64 KiB-aligned ranges and
            ! merges neighbouring ones, and a dmabuf registration that
            ! spans several allocations fails (ibv_reg_dmabuf_mr EINVAL,
            ! Helma job 963866)
            block
              integer(c_size_t) :: nbyt, b8
              b8 = int(storage_size(1.0_dp) / 8, c_size_t)
              nbyt = b8 * int(size(this%pbuf) + size(this%xjbuf) + size(this%rsbuf) + size(this%xsbuf), c_size_t)
              nbyt = ((nbyt + 2097151_c_size_t) / 2097152_c_size_t) * 2097152_c_size_t
              call device_alloc(this%xblk_d, nbyt)
              this%pbuf_d = ptr_add(this%xblk_d, 0_c_size_t)
              this%xjbuf_d = ptr_add(this%pbuf_d, b8 * size(this%pbuf))
              this%rsbuf_d = ptr_add(this%xjbuf_d, b8 * size(this%xjbuf))
              this%xsbuf_d = ptr_add(this%rsbuf_d, b8 * size(this%rsbuf))
            end block
            call device_map(this%pr_slot, this%pr_slot_d, size(this%pr_slot))
            call device_memcpy(this%pr_slot, this%pr_slot_d, size(this%pr_slot), HOST_TO_DEVICE, sync = .true.)
            call device_map(this%own_p0z, this%own_p0z_d, size(this%own_p0z))
            call device_memcpy(this%own_p0z, this%own_p0z_d, size(this%own_p0z), HOST_TO_DEVICE, sync = .true.)
            call device_map(this%own_np, this%own_np_d, size(this%own_np))
            call device_memcpy(this%own_np, this%own_np_d, size(this%own_np), HOST_TO_DEVICE, sync = .true.)
            call device_map(this%jun_off, this%jun_off_d, size(this%jun_off))
            call device_memcpy(this%jun_off, this%jun_off_d, size(this%jun_off), HOST_TO_DEVICE, sync = .true.)
            call device_map(this%sd, this%sd_d, size(this%sd))
            call device_memcpy(this%sd, this%sd_d, size(this%sd), HOST_TO_DEVICE, sync = .true.)
            call device_map(this%se, this%se_d, size(this%se))
            call device_memcpy(this%se, this%se_d, size(this%se), HOST_TO_DEVICE, sync = .true.)
            call device_map(this%sdd, this%sdd_d, nj)
            call device_map(this%sxj, this%sxj_d, nj)
         else
            call device_map(this%pbuf, this%pbuf_d, size(this%pbuf))
            call device_map(this%xjbuf, this%xjbuf_d, size(this%xjbuf))
            call pin_arr(this%pbuf, size(this%pbuf))
            call pin_arr(this%xjbuf, size(this%xjbuf))
            this%schur_pinned = .true.
         end if
       end block
       if (pe_rank .eq. 0) write(*, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,L1)') 'RS_LINE schur: pieces sent ', this%nps, &
            ' received ', this%npr, ' owned columns ', this%nown, ' neighbours ', this%nsnb, '/', &
            this%nrnb, ' device MPI ', this%schur_dmpi
    end if
    if (this%nstrip .gt. 0) then
       ! strips: factors, offsets and the node of every row on the device;
       ! the kernel keeps a strip in shared memory (48 KB by default)
       block
         integer :: s, m, p, t, idx, irow
         allocate(this%snode(this%sbo(this%nstrip + 1)))
         do s = 1, this%nstrip
            do m = 1, this%snseg(s)
               do p = merge(1, 2, m .eq. 1), lx
                  do t = 1, lx
                     call strip_node(this, s, m, p, t, idx, irow)
                     this%snode(this%sbo(s) + irow) = idx - 1
                  end do
               end do
            end do
         end do
       end block
       this%strips_dev = this%smaxn * 8 .le. 48000
       if (this%strips_dev) then
          block
            integer(c_size_t) :: nbyt
            nbyt = int(size(this%sab, kind=i8), c_size_t) * int(storage_size(1.0_dp) / 8, c_size_t)
            call device_alloc(this%sab_d, nbyt)
            call rs_line_cuda_copy(this%sab_d, c_loc(this%sab), nbyt, 1_c_int)
          end block
          call device_map(this%sbo, this%sbo_d, size(this%sbo))
          call device_memcpy(this%sbo, this%sbo_d, size(this%sbo), HOST_TO_DEVICE, sync = .true.)
          call device_map(this%snode, this%snode_d, size(this%snode))
          call device_memcpy(this%snode, this%snode_d, size(this%snode), HOST_TO_DEVICE, sync = .true.)
       end if
       block
         integer :: istat2
         character(len=8) :: envc
         ! the host factors are only needed for the RS_LINE_CHECK reference
         call get_environment_variable('RS_LINE_CHECK', envc, status = istat2)
         if (this%strips_dev .and. istat2 .ne. 0) deallocate(this%sab)
       end block
    end if
    if (this%planes) then
       ! collective: every rank, also those without strips
       block
         integer :: nfb, ierr2
         nfb = merge(1, 0, this%nstrip .gt. 0 .and. .not. this%strips_dev)
         call MPI_Allreduce(MPI_IN_PLACE, nfb, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr2)
         if (pe_rank .eq. 0 .and. nfb .gt. 0) write(*, '(A,I0,A)') 'RS_LINE strips: ', nfb, &
              ' rank(s) with a strip longer than the kernel''s shared memory: strips solved on the host there'
       end block
    end if
    td(3) = MPI_Wtime()
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,I0,A,3F8.3)') 'RS_LINE device: local columns ', this%ncol_loc, &
            ' nodes ', ntot, ' lists/pack/copies ', td(1) - td(0), td(2) - td(1), td(3) - td(2)
    end if
  end subroutine rs_line_device_setup

  !> Host solve of the local columns and isotropic nodes without the
  !! final gather-scatter (for the device check).
  subroutine rs_line_solve_host_nogs(this, z, r, n)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=dp), dimension(n), intent(inout) :: z
    real(kind=dp), dimension(n), intent(inout) :: r
    real(kind=dp) :: y(max(this%maxn, this%lx))
    integer :: e, L, p, idx, info, lx, c, m, o, pl, q, lenc, kl, ldab, pstart
    lx = this%lx
    z = 0.0_dp
    kl = lx - 1; ldab = kl + 1
    do c = 1, this%ncol
       if (this%cg_id(c) .gt. 0) cycle
       lenc = this%nseg(c) * (lx - 1) + 1
       do m = 1, this%nseg(c)
          e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
          do p = 1, lx
             pl = pos_of(lx, o, p)
             idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx**3
             q = (m - 1) * (lx - 1) + p
             y(q) = this%wsq(idx) * this%mask(idx) * r(idx)
          end do
       end do
       call dpbtrs('L', lenc, kl, 1, this%cab(1, this%cbo(c) + 1), ldab, y, this%maxn, info)
       do m = 1, this%nseg(c)
          e = this%cseg_e(m, c); L = this%cseg_L(m, c); o = this%cseg_o(m, c)
          if (m .eq. 1) then
             pstart = 1
          else
             pstart = 2
          end if
          do p = pstart, lx
             pl = pos_of(lx, o, p)
             idx = node_index(lx, this%dir(e), L, pl) + (e - 1) * lx**3
             q = (m - 1) * (lx - 1) + p
             z(idx) = this%wsq(idx) * this%mask(idx) * y(q)
          end do
       end do
    end do
    if (this%nstrip .gt. 0) call strip_solve(this, z, r, n)
    do e = 1, this%nelv
       if (.not. this%iso(e)) cycle
       do idx = (e - 1) * lx**3 + 1, e * lx**3
          z(idx) = this%jmask(idx) * this%wsq(idx)**2 * this%mask(idx) * r(idx) / this%diag(idx)
       end do
    end do
  end subroutine rs_line_solve_host_nogs

  !> Device solve: local columns by the kernel, isotropic nodes by the
  !! diagonal, distributed columns through the host, then gather-scatter.
  subroutine rs_line_solve_device(this, z, r, n)
    class(rs_line_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    type(c_ptr) :: z_d, r_d, zr_d, rr_d
    real(kind=dp) :: tt(0:3), td(0:9)

    td = 0.0_dp
    ! z_d, r_d below are the double vectors the kernels work on: Neko's own
    ! when rp is dp, otherwise the work copies (converted at entry and exit)
    zr_d = device_get_ptr(z)
    rr_d = device_get_ptr(r)
    ! our kernels run on the default stream, Neko's on glb_cmd_queue (not
    ! ordered with it): finish the caller's pending work on z and r first
    call device_sync()
    if (rp .eq. dp) then
       z_d = zr_d; r_d = rr_d
    else
       z_d = this%zw_d; r_d = this%rw_d
       call rs_line_cuda_r2d(r_d, rr_d, int(n, c_int))
    end if
    call rs_line_cuda_dzero(z_d, int(n, c_int))
    call device_sync()
    if (this%timers) tt(0) = MPI_Wtime()
    if (this%ncol_loc .gt. 0) then
       if (this%group_kernel) then
          call rs_line_cuda_solve_group(z_d, r_d, this%cabf_d, this%cabb_d, this%cnode_d, &
               this%cwrite_d, this%coff_d, this%clen_d, this%cboff_d, this%wm_d, &
               this%ywork_d, int(this%ncol_loc, c_int), int(this%lx, c_int), &
               int(merge(1, 0, this%fac_sp), c_int))
       else
          call rs_line_cuda_solve(z_d, r_d, this%cabd_d, this%cnode_d, this%cwrite_d, &
               this%coff_d, this%clen_d, this%wsq_d, this%mask_d, this%ywork_d, &
               int(this%ncol_loc, c_int), int(this%lx, c_int))
       end if
       call rs_line_cuda_sync()
    end if
    if (count(this%iso) .gt. 0) call rs_line_cuda_addcol3(z_d, this%isocoef_d, r_d, int(n, c_int))
    if (this%timers) then
       call device_sync(); tt(1) = MPI_Wtime()
    end if
    if (this%nglob .gt. 0 .and. this%schur .and. this%schur_dmpi) then
       ! Schur-complement exchange on the device: junction scalars from the
       ! device buffers straight to the owners (CUDA/ROCm-aware MPI, point
       ! to point), the junction systems by a device kernel, the junction
       ! values back the same way. Sub-timers: "toHost" = scalar kernel,
       ! "a2a" = the two exchanges, "owners" = junction kernel
       if (this%timers) td(0) = MPI_Wtime()
       if (this%nps .gt. 0) then
          call device_sync()
          call rs_line_cuda_schur_s(this%ywork_d, this%cboff_d, this%clen_d, this%pc_cl_d, this%pc_b_d, &
               this%pc_a_d, this%pc_jend_d, r_d, this%wsq_d, this%mask_d, this%ps_slot_d, this%pbuf_d, &
               int(this%nps, c_int), int(this%lx, c_int))
          call rs_line_cuda_sync()
       end if
       if (this%timers) td(2) = MPI_Wtime()
       call schur_p2p(this, this%pbuf_d, this%rsbuf_d, 3, 1)
       if (this%timers) td(3) = MPI_Wtime()
       if (this%nown .gt. 0) then
          call rs_line_cuda_schur_own(this%rsbuf_d, this%xsbuf_d, this%pr_slot_d, this%own_p0z_d, &
               this%own_np_d, this%jun_off_d, this%sd_d, this%se_d, this%sdd_d, this%sxj_d, &
               int(this%nown, c_int))
          call rs_line_cuda_sync()
       end if
       if (this%timers) td(5) = MPI_Wtime()
       call schur_p2p(this, this%xsbuf_d, this%xjbuf_d, 2, 2)
       if (this%timers) td(7) = MPI_Wtime()
       if (this%nps .gt. 0) then
          call rs_line_cuda_schur_corr(z_d, this%pcor_idx_d, this%pcor_v_d, this%pcor_w_d, this%pcor_p_d, &
               this%ps_slot_d, this%xjbuf_d, int(this%npcor, c_int))
          call rs_line_cuda_schur_jun(z_d, this%wsq_d, this%mask_d, this%pc_jend_d, this%ps_slot_d, &
               this%xjbuf_d, int(this%nps, c_int))
          call rs_line_cuda_sync()
       end if
       if (this%timers) then
          td(9) = MPI_Wtime(); td(1) = td(0); td(4) = td(3); td(6) = td(5); td(8) = td(7)
          this%tdist = this%tdist + (td(1:9) - td(0:8))
       end if
    else if (this%nglob .gt. 0 .and. this%schur) then
       ! Schur-complement exchange through the host: junction scalars to the
       ! owners, the tridiagonal junction systems on the host, the junction
       ! values back
       if (this%timers) td(0) = MPI_Wtime()
       if (this%nps .gt. 0) then
          call device_sync()
          call rs_line_cuda_schur_s(this%ywork_d, this%cboff_d, this%clen_d, this%pc_cl_d, this%pc_b_d, &
               this%pc_a_d, this%pc_jend_d, r_d, this%wsq_d, this%mask_d, this%ps_slot_d, this%pbuf_d, &
               int(this%nps, c_int), int(this%lx, c_int))
          call rs_line_cuda_sync()
          call device_memcpy(this%pbuf, this%pbuf_d, 3 * this%nps, DEVICE_TO_HOST, sync = .true.)
       end if
       if (this%timers) td(2) = MPI_Wtime()
       call schur_exchange_solve(this)
       if (this%timers) td(7) = MPI_Wtime()
       if (this%nps .gt. 0) then
          call device_memcpy(this%xjbuf, this%xjbuf_d, 2 * this%nps, HOST_TO_DEVICE, sync = .true.)
          call device_sync()
          call rs_line_cuda_schur_corr(z_d, this%pcor_idx_d, this%pcor_v_d, this%pcor_w_d, this%pcor_p_d, &
               this%ps_slot_d, this%xjbuf_d, int(this%npcor, c_int))
          call rs_line_cuda_schur_jun(z_d, this%wsq_d, this%mask_d, this%pc_jend_d, this%ps_slot_d, &
               this%xjbuf_d, int(this%nps, c_int))
          call rs_line_cuda_sync()
       end if
       if (this%timers) then
          td(9) = MPI_Wtime(); td(1) = td(0); td(3:6) = td(2); td(8) = td(7)
          this%tdist = this%tdist + (td(1:9) - td(0:8))
       end if
    else if (this%nglob .gt. 0) then
       ! distributed columns: only the packed pieces cross to the host
       ! (every rank takes part in the exchange, even with no pieces)
       if (this%timers) td(0) = MPI_Wtime()
       if (this%ngp .gt. 0) then
          call device_sync()
          call rs_line_cuda_gpack(r_d, this%wsq_d, this%mask_d, this%gp_idx_d, this%sbuf_d, &
               int(this%ngp, c_int))
          call rs_line_cuda_sync()
          if (this%timers) td(1) = MPI_Wtime()
          call device_memcpy(this%sbuf, this%sbuf_d, this%ngp, DEVICE_TO_HOST, sync = .true.)
       end if
       if (this%timers) then
          if (this%ngp .le. 0) td(1) = td(0)
          td(2) = MPI_Wtime()
       end if
       call global_exchange_solve_device(this, td)
       if (this%timers) td(7) = MPI_Wtime()
       if (this%ngp .gt. 0) then
          call device_memcpy(this%sbuf, this%sbuf_d, this%ngp, HOST_TO_DEVICE, sync = .true.)
          if (this%timers) td(8) = MPI_Wtime()
          call rs_line_cuda_gunpack(z_d, this%wsq_d, this%mask_d, this%gp_idx_d, this%gp_wr_d, &
               this%sbuf_d, int(this%ngp, c_int))
          call rs_line_cuda_sync()
       end if
       if (this%timers) then
          if (this%ngp .le. 0) td(8) = td(7)
          td(9) = MPI_Wtime()
          this%tdist = this%tdist + (td(1:9) - td(0:8))
       end if
    end if
    if (this%nstrip .gt. 0 .and. this%timers) then
       call device_sync(); this%tstrip = this%tstrip - MPI_Wtime()
    end if
    if (this%nstrip .gt. 0 .and. this%strips_dev) then
       call device_sync()
       call rs_line_cuda_strip_solve(z_d, r_d, this%wsq_d, this%mask_d, this%sab_d, this%sbo_d, &
            this%snode_d, int(this%nstrip, c_int), int(this%lx**2 - 1, c_int), int(this%smaxn, c_int))
       call rs_line_cuda_sync()
    else if (this%nstrip .gt. 0) then
       ! plane strips on the host (r to the host, strip solves, the strip
       ! part added on the device)
       block
         real(kind=dp), allocatable, target :: rh(:)
         allocate(rh(n))
         call rs_line_cuda_copy(c_loc(rh), r_d, int(n, c_size_t) * 8_c_size_t, 2_c_int)
         this%zh = 0.0_dp
         call strip_solve(this, this%zh, rh, n)
         deallocate(rh)
       end block
       call device_memcpy(this%zh, this%zh_d, n, HOST_TO_DEVICE, sync = .true.)
       call rs_line_cuda_dadd2(z_d, this%zh_d, int(n, c_int))
    end if
    if (this%nstrip .gt. 0 .and. this%timers) then
       call device_sync(); this%tstrip = this%tstrip + MPI_Wtime()
    end if
    if (this%timers) then
       call device_sync(); tt(2) = MPI_Wtime()
    end if
    if (rp .ne. dp) call rs_line_cuda_d2r(zr_d, z_d, int(n, c_int))
    call rs_line_cuda_sync()   ! before Neko's gs on its own stream
    call this%gs_h%op(z, n, GS_OP_ADD)
    if (this%timers) then
       call device_sync(); tt(3) = MPI_Wtime()
       this%tsol = this%tsol + (tt(1:3) - tt(0:2))
       this%nsol = this%nsol + 1
    end if
    if (this%check_device) then
       ! one-time comparison (env RS_LINE_CHECK) of the complete device
       ! solve (local columns, isotropic part, distributed columns, gs)
       ! with the complete host solve, per rank; the error is also given
       ! at the nodes of the distributed pieces alone (review 5)
       block
         real(kind=dp), allocatable :: zc(:), zd(:)
         real(kind=dp), allocatable, target :: rh(:)
         logical, allocatable :: gmark(:)
         real(kind=dp) :: dmax, zmax, dglob, dloc
         integer :: i
         allocate(zc(n), rh(n), zd(n), gmark(n))
         ! r as the kernels saw it (double), z after the gather-scatter
         call rs_line_cuda_copy(c_loc(rh), r_d, int(n, c_size_t) * 8_c_size_t, 2_c_int)
         call device_memcpy(z, zr_d, n, DEVICE_TO_HOST, sync = .true.)
         zd = z
         this%zh = 0.0_dp
         call rs_line_solve_host_nogs(this, zc, rh, n)
         if (this%nglob .gt. 0) call solve_global(this, zc, rh, n)
         call hgs(this%gs_h, zc, n, GS_OP_ADD)
         gmark = .false.
         do i = 1, this%ngp
            gmark(this%gp_idx(i) + 1) = .true.
         end do
         if (this%schur) then
            ! the pieces' interior nodes and their end junctions
            do i = 1, this%npcor
               gmark(this%pcor_idx(i) + 1) = .true.
            end do
            do i = 1, this%nps
               if (this%pc_jend(i) .ge. 0) gmark(this%pc_jend(i) + 1) = .true.
            end do
         end if
         dmax = maxval(abs(zc - zd)); zmax = maxval(abs(zc))
         dglob = 0.0_dp; dloc = 0.0_dp
         if (any(gmark)) dglob = maxval(abs(zc - zd), mask = gmark)
         if (.not. all(gmark)) dloc = maxval(abs(zc - zd), mask = .not. gmark)
         write(*, '(A,I0,A,I0,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,I0,A,I0,A,I0,A,I0)') &
              'RS_LINE device check (rank ', pe_rank, ', lx ', this%lx, '): max|z_gpu - z_cpu| ', dmax, &
              ' at distributed nodes ', dglob, ' elsewhere ', dloc, ' max|z_cpu| ', zmax, &
              ' local columns ', this%ncol_loc, ' pieces sent ', this%ngp, ' received ', this%nrb, &
              ' owned ', this%nown
         deallocate(zc, rh, zd, gmark)
       end block
       this%check_device = .false.
    end if
  end subroutine rs_line_solve_device

  subroutine rs_line_update(this)
    class(rs_line_t), intent(inout) :: this
  end subroutine rs_line_update

  subroutine rs_line_free(this)
    class(rs_line_t), intent(inout) :: this

    if (pe_rank .eq. 0 .and. this%nsol .gt. 0) then
       write(*, '(A,I0,A,I0,A,3F9.4)') 'RS_LINE device solve (rank 0, lx ', this%lx, ', ', this%nsol, &
            ' calls) local/distributed/gs ', this%tsol
       if (this%nstrip .gt. 0) write(*, '(A,I0,A,F9.4)') 'RS_LINE device solve (rank 0, lx ', this%lx, &
            ') of which plane strips ', this%tstrip
       if (this%nglob .gt. 0) then
          write(*, '(A,I0,A,9F8.4)') 'RS_LINE distributed (rank 0, lx ', this%lx, &
               ') pack/toHost/a2a/toDev/owners/toHost/a2a/toDev/unpack ', this%tdist
       end if
    end if
    this%tsol = 0.0_dp; this%nsol = 0; this%tdist = 0.0_dp; this%tstrip = 0.0_dp
    if (c_associated(this%rw_d)) call device_free(this%rw_d)
    if (c_associated(this%zw_d)) call device_free(this%zw_d)
    this%rw_d = c_null_ptr; this%zw_d = c_null_ptr
    if (allocated(this%dir)) deallocate(this%dir)
    if (allocated(this%blk)) deallocate(this%blk)
    if (allocated(this%lu)) deallocate(this%lu)
    if (allocated(this%ipiv)) deallocate(this%ipiv)
    if (c_associated(this%cnode_d)) then
       call device_deassociate(this%wsq); call device_free(this%wsq_d)
       call device_deassociate(this%mask); call device_free(this%mask_d)
       this%wsq_d = c_null_ptr; this%mask_d = c_null_ptr
       if (c_associated(this%wm_d)) then
          call device_deassociate(this%wm); call device_free(this%wm_d)
       end if
       this%wm_d = c_null_ptr
    end if
    if (allocated(this%wsq)) deallocate(this%wsq)
    if (allocated(this%mask)) deallocate(this%mask)
    if (allocated(this%wm)) deallocate(this%wm)
    if (allocated(this%lat)) deallocate(this%lat)
    if (allocated(this%iso)) deallocate(this%iso)
    if (allocated(this%diag)) deallocate(this%diag)
    if (allocated(this%jmask)) deallocate(this%jmask)
    if (c_associated(this%cnode_d)) then
       call device_deassociate(this%cnode); call device_free(this%cnode_d)
       call device_deassociate(this%cwrite); call device_free(this%cwrite_d)
       call device_deassociate(this%coff); call device_free(this%coff_d)
       call device_deassociate(this%clen); call device_free(this%clen_d)
       call device_deassociate(this%cabd); call device_free(this%cabd_d)
       call device_deassociate(this%ywork); call device_free(this%ywork_d)
       if (c_associated(this%cabf_d)) then
          call free_fac(this%cabf_d)
          call free_fac(this%cabb_d)
          this%cabf_d = c_null_ptr; this%cabb_d = c_null_ptr
       end if
       if (c_associated(this%cboff_d)) then
          call device_deassociate(this%cboff); call device_free(this%cboff_d)
          this%cboff_d = c_null_ptr
       end if
       call device_deassociate(this%isocoef); call device_free(this%isocoef_d)
       call device_deassociate(this%zh); call device_free(this%zh_d)
       this%cnode_d = c_null_ptr
       if (c_associated(this%gp_idx_d)) then
          if (this%sbuf_pinned) call unpin_arr(this%sbuf)
          this%sbuf_pinned = .false.
          call device_deassociate(this%gp_idx); call device_free(this%gp_idx_d)
          call device_deassociate(this%gp_wr); call device_free(this%gp_wr_d)
          call device_deassociate(this%sbuf); call device_free(this%sbuf_d)
          this%gp_idx_d = c_null_ptr; this%gp_wr_d = c_null_ptr; this%sbuf_d = c_null_ptr
       end if
       if (this%gown_dev) then
          if (this%rbuf_pinned) call unpin_arr(this%rbuf)
          this%rbuf_pinned = .false.
          call device_deassociate(this%rb_map); call device_free(this%rb_map_d)
          call device_deassociate(this%glen); call device_free(this%glen_d)
          call device_deassociate(this%gcabd); call device_free(this%gcabd_d)
          call device_deassociate(this%gyw); call device_free(this%gyw_d)
          call device_deassociate(this%rbuf); call device_free(this%rbuf_d)
          call device_deassociate(this%goff); call device_free(this%goff_d)
          call free_fac(this%gcabf_d)
          call free_fac(this%gcabb_d)
          this%rb_map_d = c_null_ptr; this%glen_d = c_null_ptr; this%gcabd_d = c_null_ptr
          this%gyw_d = c_null_ptr; this%rbuf_d = c_null_ptr
          this%goff_d = c_null_ptr; this%gcabf_d = c_null_ptr; this%gcabb_d = c_null_ptr
       end if
    end if
    if (allocated(this%rb_map)) deallocate(this%rb_map, this%glen, this%gcabd, this%gyw, &
         this%goff, this%gcabf, this%gcabb)
    if (c_associated(this%sab_d)) then
       call device_free(this%sab_d)
       call device_deassociate(this%sbo); call device_free(this%sbo_d)
       call device_deassociate(this%snode); call device_free(this%snode_d)
       this%sab_d = c_null_ptr; this%sbo_d = c_null_ptr; this%snode_d = c_null_ptr
    end if
    this%strips_dev = .false.
    if (allocated(this%snode)) deallocate(this%snode)
    if (c_associated(this%pc_cl_d)) then
       call device_deassociate(this%pc_cl); call device_free(this%pc_cl_d)
       call device_deassociate(this%pc_jend); call device_free(this%pc_jend_d)
       call device_deassociate(this%ps_slot); call device_free(this%ps_slot_d)
       call device_deassociate(this%pc_b); call device_free(this%pc_b_d)
       call device_deassociate(this%pc_a); call device_free(this%pc_a_d)
       call device_deassociate(this%pcor_idx); call device_free(this%pcor_idx_d)
       call device_deassociate(this%pcor_p); call device_free(this%pcor_p_d)
       call device_deassociate(this%pcor_v); call device_free(this%pcor_v_d)
       call device_deassociate(this%pcor_w); call device_free(this%pcor_w_d)
       if (this%schur_pinned) then
          call unpin_arr(this%pbuf); call unpin_arr(this%xjbuf)
       end if
       this%schur_pinned = .false.
       if (.not. this%schur_dmpi) then
          call device_deassociate(this%pbuf); call device_free(this%pbuf_d)
          call device_deassociate(this%xjbuf); call device_free(this%xjbuf_d)
       end if
       if (this%schur_dmpi) then
          call device_free(this%xblk_d)
          this%xblk_d = c_null_ptr; this%pbuf_d = c_null_ptr; this%xjbuf_d = c_null_ptr
          call device_deassociate(this%pr_slot); call device_free(this%pr_slot_d)
          call device_deassociate(this%own_p0z); call device_free(this%own_p0z_d)
          call device_deassociate(this%own_np); call device_free(this%own_np_d)
          call device_deassociate(this%jun_off); call device_free(this%jun_off_d)
          call device_deassociate(this%sd); call device_free(this%sd_d)
          call device_deassociate(this%se); call device_free(this%se_d)
          call device_deassociate(this%sdd); call device_free(this%sdd_d)
          call device_deassociate(this%sxj); call device_free(this%sxj_d)
          this%rsbuf_d = c_null_ptr; this%xsbuf_d = c_null_ptr; this%pr_slot_d = c_null_ptr
          this%own_p0z_d = c_null_ptr; this%own_np_d = c_null_ptr; this%jun_off_d = c_null_ptr
          this%sd_d = c_null_ptr; this%se_d = c_null_ptr; this%sdd_d = c_null_ptr; this%sxj_d = c_null_ptr
       end if
       this%schur_dmpi = .false.
       if (c_associated(this%schur_reqs)) call rs_mpi_free_reqs(this%schur_reqs)
       this%schur_reqs = c_null_ptr
       if (allocated(this%snb)) deallocate(this%snb, this%rnb)
       if (allocated(this%own_p0z)) deallocate(this%own_p0z, this%sdd, this%sxj)
       this%nsnb = 0; this%nrnb = 0
       this%pc_cl_d = c_null_ptr; this%pc_jend_d = c_null_ptr; this%pc_b_d = c_null_ptr; this%pc_a_d = c_null_ptr
       this%pcor_idx_d = c_null_ptr; this%pcor_p_d = c_null_ptr; this%pcor_v_d = c_null_ptr
       this%pcor_w_d = c_null_ptr; this%pbuf_d = c_null_ptr; this%xjbuf_d = c_null_ptr; this%ps_slot_d = c_null_ptr
    end if
    if (allocated(this%c2p)) deallocate(this%c2p, this%pn_off, this%pn_len, this%pn_idx)
    if (allocated(this%pc_cl)) deallocate(this%pc_cl, this%pc_jend, this%pc_b, this%pc_a)
    if (allocated(this%pcor_idx)) deallocate(this%pcor_idx, this%pcor_p, this%pcor_v, this%pcor_w)
    if (allocated(this%pbuf)) deallocate(this%pbuf, this%xjbuf)
    if (allocated(this%rsbuf)) deallocate(this%rsbuf, this%xsbuf)
    if (allocated(this%sd)) deallocate(this%sd, this%se, this%jun_off)
    if (allocated(this%ps_slot)) deallocate(this%ps_slot, this%pr_slot, this%pscnt, this%psdsp, &
         this%prcnt, this%prdsp, this%own_p0, this%own_np)
    if (allocated(this%cg_last)) deallocate(this%cg_last)
    this%schur = .false.; this%npcor = 0
    this%gown_dev = .false.; this%nrb = 0
    if (allocated(this%gp_idx)) deallocate(this%gp_idx, this%gp_wr)
    this%ngp = 0
    if (allocated(this%cnode)) deallocate(this%cnode, this%cwrite, this%coff, this%clen)
    if (allocated(this%cabd)) deallocate(this%cabd, this%ywork, this%isocoef, this%zh)
    if (allocated(this%cabf)) deallocate(this%cabf, this%cabb)
    if (allocated(this%cboff)) deallocate(this%cboff)
    this%ncol_loc = 0
    if (allocated(this%nseg)) deallocate(this%nseg)
    if (allocated(this%cseg_e)) deallocate(this%cseg_e)
    if (allocated(this%cseg_L)) deallocate(this%cseg_L)
    if (allocated(this%cseg_o)) deallocate(this%cseg_o)
    if (allocated(this%cab)) deallocate(this%cab)
    if (allocated(this%cbo)) deallocate(this%cbo)
    if (allocated(this%pel)) deallocate(this%pel, this%pnorm, this%pid)
    if (allocated(this%pblk)) deallocate(this%pblk)
    if (allocated(this%snseg)) deallocate(this%snseg, this%sseg_e, this%sseg_k, this%sseg_o, this%sseg_x)
    if (allocated(this%sbo)) deallocate(this%sbo)
    if (allocated(this%sab)) deallocate(this%sab)
    this%npel = 0; this%nstrip = 0; this%smaxseg = 0; this%smaxn = 0
    if (allocated(this%cg_id)) deallocate(this%cg_id)
    if (allocated(this%cg_rev)) deallocate(this%cg_rev)
    if (allocated(this%cg_off)) deallocate(this%cg_off)
    if (allocated(this%cg_own)) deallocate(this%cg_own)
    if (allocated(this%own_len)) deallocate(this%own_len)
    if (allocated(this%gcab)) deallocate(this%gcab)
    if (allocated(this%hoff)) deallocate(this%hoff)
    if (allocated(this%gy)) deallocate(this%gy)
    if (allocated(this%ps_c)) deallocate(this%ps_c, this%ps_pos, this%ps_len)
    if (allocated(this%pr_own)) deallocate(this%pr_own, this%pr_q0, this%pr_pos, this%pr_len)
    if (allocated(this%sendcnt)) deallocate(this%sendcnt, this%senddsp, this%recvcnt, this%recvdsp)
    if (allocated(this%sbuf)) deallocate(this%sbuf)
    if (allocated(this%rbuf)) deallocate(this%rbuf)
    this%nglob = 0; this%nown = 0; this%gmaxn = 0; this%nps = 0; this%npr = 0
    nullify(this%gs_h)
    this%lx = 0; this%nelv = 0; this%n = 0; this%ncol = 0; this%maxseg = 0; this%maxn = 0
  end subroutine rs_line_free

end module rs_line
