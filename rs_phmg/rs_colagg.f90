!> Two-level line-aware aggregation solver for the N = 1 level of
!! `rs_phmg` (coarse_solver = "colagg").
!!
!! Aggregates: consecutive nodes along every column line of the N = 1
!! `rs_line` (columns = true) are merged when the assembled line coupling
!! between them dominates the lateral couplings of both,
!! |a_q,q+1| >= theta * max(lat_q, lat_q+1), where lat is the sum over the
!! elements holding the node of the element-local |couplings| to nodes off
!! the element's stiff line (a heuristic bound of the lateral coupling, not
!! the assembled lateral row sum); isotropic regions therefore
!! stay unaggregated (the coarse problem is exact there) and columns
!! collapse only where the anisotropy is real. The aggregate ids are made
!! consistent over duplicate lines, element copies and ranks by label
!! propagation (min dof id over the merge set, gather-scatter MIN).
!! The coarse operator A_c = P^T A P (P = piecewise constant on the
!! aggregates, Dirichlet nodes masked) is assembled from the N = 1 element
!! matrices (lx**3 local comb products) into a replicated sparse matrix;
!! it is LU-factored when small (dense_max) and solved redundantly by
!! CG with a symmetric Gauss-Seidel preconditioner otherwise. One application is a symmetric V(1,1) cycle:
!! column-line Chebyshev pre-smoothing from zero, exact coarse
!! correction, Chebyshev post-smoothing. Replicating A_c is the prototype
!! path; the production path is to hand A_c to a parallel AMG. CPU only.
module rs_colagg
  use num_types, only : rp, dp, i8
  use precon, only : pc_t
  use coefs, only : coef_t
  use bc_list, only : bc_list_t
  use json_module, only : json_file
  use ax_product, only : ax_t
  use gather_scatter, only : gs_t, GS_OP_ADD, GS_OP_MIN
  use scalar_bc_projector, only : scalar_bc_projector_t
  use cheby, only : cheby_t
  use cheby_device, only : cheby_device_t
  use, intrinsic :: iso_c_binding, only : c_ptr, c_null_ptr, c_associated, c_loc, c_f_pointer
  use device, only : device_map, device_memcpy, device_get_ptr, device_free, &
       device_deassociate, device_sync, HOST_TO_DEVICE, DEVICE_TO_HOST
  use device_math, only : device_rzero, device_add2, device_sub3, device_copy, &
       device_glsc2, device_add2s2, device_add2s1, device_col3, device_cmult, device_vlsc3
  use, intrinsic :: iso_c_binding, only : c_int, c_int64_t, c_double, c_char, c_funptr, &
       c_null_char, c_f_procpointer
  use rs_line, only : rs_line_cuda_sync
  use rs_spchol, only : spchol_t
  use neko_config, only : NEKO_BCKND_DEVICE
  use rs_line, only : hgs, hbcp, rs_mem_report
  use field, only : field_t
  use dofmap, only : dofmap_t
  use mesh, only : mesh_t
  use space, only : space_t
  use krylov, only : ksp_monitor_t
  use rs_line, only : rs_line_t
  use math, only : glsc3
  use utils, only : neko_error
  use logger, only : neko_log, LOG_SIZE
  use comm, only : pe_rank, pe_size, NEKO_COMM, MPI_REAL_PRECISION
  use mpi_f08, only : MPI_Allreduce, MPI_Allgather, MPI_Allgatherv, MPI_IN_PLACE, MPI_Bcast, &
       MPI_SUM, MPI_MAX, MPI_INTEGER, MPI_INTEGER8, MPI_DOUBLE_PRECISION, MPI_Wtime
  implicit none
  private

  !> One level of the aggregation hierarchy built on the replicated A_c
  type :: agg_lvl_t
     integer :: n = 0
     integer :: nnz = 0
     integer, allocatable :: ia(:), ja(:)
     integer, allocatable :: agg(:)        ! row -> aggregate of the next level
     real(kind=rp), allocatable :: av(:), dinv(:)
     real(kind=rp), allocatable :: x(:), r(:), b(:)
  end type agg_lvl_t

  !> MKL PARDISO (Fortran calling convention, all arguments by reference),
  !! looked up at run time with dlsym so that builds without MKL still link
  abstract interface
     subroutine pardiso_if(pt, maxfct, mnum, mtype, phase, n, a, ia, ja, perm, nrhs, &
          iparm, msglvl, b, x, error) bind(c)
       import c_int64_t, c_int, c_double
       integer(c_int64_t) :: pt(64)
       integer(c_int) :: maxfct, mnum, mtype, phase, n, ia(*), ja(*), perm(*), nrhs, iparm(64), &
            msglvl, error
       real(c_double) :: a(*), b(*), x(*)
     end subroutine pardiso_if
     subroutine pardisoinit_if(pt, mtype, iparm) bind(c)
       import c_int64_t, c_int
       integer(c_int64_t) :: pt(64)
       integer(c_int) :: mtype, iparm(64)
     end subroutine pardisoinit_if
  end interface
  procedure(pardiso_if), pointer :: pds_fn => null()
  procedure(pardisoinit_if), pointer :: pds_init_fn => null()

  interface
     function rs_dlsym(handle, name) bind(c, name = 'dlsym') result(f)
       import c_ptr, c_funptr, c_char
       type(c_ptr), value :: handle
       character(kind=c_char), dimension(*) :: name
       type(c_funptr) :: f
     end function rs_dlsym
  end interface

  interface
     function rs_colagg_cuda_potrf(a, n) bind(c, name = 'rs_colagg_cuda_potrf') result(info)
       import c_ptr, c_int
       type(c_ptr), value :: a
       integer(c_int), value :: n
       integer(c_int) :: info
     end function rs_colagg_cuda_potrf
     subroutine rs_colagg_cuda_trsv2(a, x, n) bind(c, name = 'rs_colagg_cuda_trsv2')
       import c_ptr, c_int
       type(c_ptr), value :: a, x
       integer(c_int), value :: n
     end subroutine rs_colagg_cuda_trsv2
     subroutine rs_colagg_cuda_restrict(w, mult, agg, rc, n) bind(c, name = 'rs_colagg_cuda_restrict')
       import c_ptr, c_int
       type(c_ptr), value :: w, mult, agg, rc
       integer(c_int), value :: n
     end subroutine rs_colagg_cuda_restrict
     subroutine rs_colagg_cuda_prolong(z, xc, agg, n) bind(c, name = 'rs_colagg_cuda_prolong')
       import c_ptr, c_int
       type(c_ptr), value :: z, xc, agg
       integer(c_int), value :: n
     end subroutine rs_colagg_cuda_prolong
     subroutine rs_colagg_cuda_spmv(ia, ja, av, p, q, m) bind(c, name = 'rs_colagg_cuda_spmv')
       import c_ptr, c_int
       type(c_ptr), value :: ia, ja, av, p, q
       integer(c_int), value :: m
     end subroutine rs_colagg_cuda_spmv
  end interface

  type, public, extends(pc_t) :: rs_colagg_t
     class(ax_t), pointer :: ax => null()
     type(coef_t), pointer :: coef => null()
     type(gs_t), pointer :: gs_h => null()
     class(scalar_bc_projector_t), pointer :: bcp => null()
     type(cheby_t), pointer :: cheby => null()
     type(cheby_device_t), pointer :: cheby_dev => null()
     type(c_ptr) :: w_d = c_null_ptr, res_d = c_null_ptr, zbuf_d = c_null_ptr
     !> device coarse level: aggregate map (0-based), CSR, live mask, vectors
     integer, allocatable :: agg0(:), ia0(:), ja0(:)
     real(kind=rp), allocatable :: live(:), rcd(:), xcd(:), pcd(:), qcd(:), zcd(:)
     type(c_ptr) :: agg0_d = c_null_ptr, ia0_d = c_null_ptr, ja0_d = c_null_ptr
     type(c_ptr) :: av_d = c_null_ptr, dinv_d = c_null_ptr, live_d = c_null_ptr
     type(c_ptr) :: rcd_d = c_null_ptr, xcd_d = c_null_ptr, pcd_d = c_null_ptr
     !> dense coarse solve on the device: Cholesky factor of A_c (lower,
     !! column-major) on the device, two triangular solves per call
     logical :: dense_dev = .false.
     type(c_ptr) :: ac_d = c_null_ptr
     !> ones(nagg) on the device: rank-local dot products (device_vlsc3)
     !! for the replicated coarse CG, which then needs no communication
     real(kind=rp), allocatable :: onesc(:)
     type(c_ptr) :: onesc_d = c_null_ptr
     type(c_ptr) :: qcd_d = c_null_ptr, zcd_d = c_null_ptr
     logical :: crs_on_device = .false.
     !> device cycle: solve the (replicated, host-resident after the
     !! allreduce) coarse system on the host with the preconditioned CG of
     !! the CPU path instead of the device Jacobi-CG
     logical :: crs_host = .false.
     !> sparse direct coarse solve (option colagg_direct): A_c factorized
     !! once, replicated on every rank, by rs_spchol (default) or MKL PARDISO
     !! (real SPD, METIS ordering; colagg_direct_solver "pardiso"); two
     !! triangular solves on the host per call
     logical :: direct = .false.
     !> 1: own sparse Cholesky (rs_spchol, default), 2: MKL PARDISO
     integer :: direct_kind = 1
     type(spchol_t) :: spc
     integer(c_int64_t) :: pds_pt(64) = 0
     integer(c_int) :: pds_iparm(64) = 0
     integer, allocatable :: iau(:), jau(:)
     real(kind=dp), allocatable :: avu(:)          ! PARDISO: double
     !> device cycle timers (env RS_TIMERS): pre-smoother, residual,
     !! restriction (+ allreduce), coarse solve, prolongation, post-smoother
     logical :: timers = .false.
     real(kind=dp) :: tcyc(6) = 0.0_dp
     integer :: ncyc = 0
     type(mesh_t), pointer :: msh => null()
     type(space_t), pointer :: Xh => null()
     type(field_t) :: zf
     integer :: n = 0
     integer :: nagg = 0
     integer :: nnz = 0
     integer :: nsmooth = 3
     integer :: dense_max = 2000
     real(kind=rp) :: theta = 1.0_rp
     !> length cap of the aggregation: run length n allowed iff n^2 <= kappa * ratio
     real(kind=rp) :: kappa = 1.0e30_rp
     real(kind=rp) :: crs_tol = 1.0e-10_rp
     integer :: crs_max_iter = 10000
     logical :: dense = .false.
     logical :: pinned = .false.
     !> distributed coarse CG: this rank owns rows r0..r1 of the replicated A_c
     logical :: parallel_crs = .true.
     integer :: r0 = 1, r1 = 0
     integer, allocatable :: rcnt(:), rdsp(:)
     !> multilevel preconditioner of the redundant CG on the sparse A_c
     integer :: nlv = 0
     type(agg_lvl_t), allocatable :: lv(:)
     integer :: ncrs = 0
     real(kind=dp), allocatable :: acc(:,:)        ! LAPACK: double
     integer, allocatable :: ipivc(:)
     integer, allocatable :: aggid(:)
     logical, allocatable :: dead(:)
     integer, allocatable :: ia(:), ja(:)
     !> A_c in double (direct factorization); avr: its copy in Neko's
     !! precision for the device matvec
     real(kind=dp), allocatable :: av(:)
     real(kind=rp), allocatable :: avr(:), dinv(:)
     real(kind=dp), allocatable :: ac(:,:)         ! LAPACK / cuSOLVER: double
     integer, allocatable :: ipiv(:)
     real(kind=rp), allocatable :: w(:), res(:), zbuf(:)
     real(kind=rp), allocatable :: rc(:), xc(:), pc(:), qc(:), zc(:)
     integer :: crs_iters = 0
     integer :: crs_calls = 0
     integer :: crs_fails = 0
     !> 1 - (aggregates / unique N = 1 dofs): 0 = nothing merged
     real(kind=rp) :: merged_fraction = 0.0_rp
   contains
     procedure, pass(this) :: init => rs_colagg_init
     procedure, pass(this) :: init_from_components => rs_colagg_init_from_components
     procedure, pass(this) :: solve => rs_colagg_solve
     procedure, pass(this) :: cycles => rs_colagg_cycles
     procedure, pass(this) :: update => rs_colagg_update
     procedure, pass(this) :: free => rs_colagg_free
  end type rs_colagg_t

contains

  subroutine rs_colagg_init(this, coef, bclst, json)
    class(rs_colagg_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json
    call neko_error('rs_colagg: use init_from_components')
  end subroutine rs_colagg_init

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

  subroutine rs_colagg_init_from_components(this, ax, coef, gs_h, bcp, cheby, &
       msh, Xh, dm, line, nsmooth, theta, dense_max, hierarchy, kappa, cheby_device, crs_host, direct, &
       direct_solver)
    class(rs_colagg_t), intent(inout), target :: this
    class(ax_t), intent(in), target :: ax
    type(coef_t), intent(inout), target :: coef
    type(gs_t), intent(inout), target :: gs_h
    class(scalar_bc_projector_t), intent(inout), target :: bcp
    type(cheby_t), intent(inout), target :: cheby
    type(mesh_t), intent(in), target :: msh
    type(space_t), intent(in), target :: Xh
    type(dofmap_t), intent(in), target :: dm
    type(rs_line_t), intent(in) :: line
    integer, intent(in) :: nsmooth
    real(kind=rp), intent(in) :: theta
    integer, intent(in) :: dense_max
    logical, intent(in), optional :: hierarchy
    real(kind=rp), intent(in), optional :: kappa
    type(cheby_device_t), intent(inout), target, optional :: cheby_device
    logical, intent(in), optional :: crs_host
    logical, intent(in), optional :: direct
    character(len=*), intent(in), optional :: direct_solver
    type(c_ptr) :: x_d, w_d, k_d, r1_d, r2_d, z1_d, z2_d
    integer :: lx, nelv, n, lx3, nl, c, m, e, e0, L, L0, o, o0, d, p, pl, pl2
    integer :: idx, idx2, i, j, k, ierr, info, it, nedge, nedge_max, nmerge
    integer :: off, nloc, ntot, nt, nt_loc, ndead, maxsz, nsingle, ii, jj, runlen
    real(kind=rp) :: ratio
    integer(kind=i8) :: id
    real(kind=rp) :: chg, chg_glb, dmax, dmin, asym
    real(kind=dp) :: a
    real(kind=dp), allocatable :: key(:), key_old(:), mask(:)
    real(kind=rp), allocatable :: x(:), w(:), ae(:,:,:)
    real(kind=dp), allocatable :: tv(:), gv(:)      ! A_c assembled in double
    integer(kind=i8), allocatable :: tk(:), gk(:), lk(:), allk(:), uniq(:)
    integer, allocatable :: ed_i(:), ed_j(:), counts(:), displs(:), aggsz(:)
    character(len=LOG_SIZE) :: log_buf
    real(kind=dp) :: tm(0:6)

    call this%free()
    tm(0) = MPI_Wtime()
    this%ax => ax; this%coef => coef; this%gs_h => gs_h; this%bcp => bcp
    this%cheby => cheby; this%msh => msh; this%Xh => Xh
    if (present(cheby_device)) this%cheby_dev => cheby_device
    this%nsmooth = nsmooth; this%theta = theta; this%dense_max = dense_max
    this%crs_host = .false.
    if (present(crs_host)) this%crs_host = crs_host
    this%direct = .false.
    if (present(direct)) this%direct = direct
    this%direct_kind = 1
    if (present(direct_solver)) then
       select case (trim(direct_solver))
       case ('spchol')
          this%direct_kind = 1
       case ('pardiso')
          this%direct_kind = 2
       case default
          call neko_error('rs_colagg: colagg_direct_solver must be spchol or pardiso')
       end select
    end if
    this%kappa = 1.0e30_rp
    if (present(kappa)) then
       if (kappa .gt. 0.0_rp) this%kappa = kappa
    end if
    lx = Xh%lx; nelv = msh%nelv; lx3 = lx**3; nl = lx * lx; n = lx3 * nelv
    this%n = n
    if (.not. line%columns .or. .not. allocated(line%lat)) then
       call neko_error('rs_colagg: needs the line smoother with columns at N = 1')
    end if
    call this%zf%init(dm, 'colagg_z')
    allocate(this%aggid(n), this%w(n), this%res(n), this%zbuf(n), mask(n))
    this%w = 0.0_rp; this%res = 0.0_rp; this%zbuf = 0.0_rp
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call device_map(this%w, this%w_d, n)
       call device_map(this%res, this%res_d, n)
       call device_map(this%zbuf, this%zbuf_d, n)
    end if
    do i = 1, n
       mask(i) = 1.0_rp
    end do
    call hbcp(bcp, mask, n)

    ! ---- merge edges along the columns ------------------------------------
    nedge_max = max(1, line%ncol * line%maxseg * lx)
    allocate(ed_i(nedge_max), ed_j(nedge_max))
    nedge = 0; nmerge = 0
    do c = 1, line%ncol
       runlen = 1
       do m = 1, line%nseg(c)
          e = line%cseg_e(m, c); L = line%cseg_L(m, c); o = line%cseg_o(m, c)
          d = line%dir(e)
          do p = 1, lx - 1
             if (o .eq. 1) then
                pl = p; pl2 = p + 1
             else
                pl = lx + 1 - p; pl2 = lx - p
             end if
             idx = node_index(lx, d, L, pl) + (e - 1) * lx3
             idx2 = node_index(lx, d, L, pl2) + (e - 1) * lx3
             a = abs(line%blk(pl, pl2, L, e))
             ratio = a / max(theta * max(line%lat(idx), line%lat(idx2)), tiny(1.0_rp))
             ! strength, and a length cap: a run of n merged nodes needs a
             ! coupling ratio of at least n**2 / kappa (review 2, finding 4)
             if (a .gt. 0.0_rp .and. ratio .ge. 1.0_rp .and. &
                  real(runlen + 1, rp)**2 .le. this%kappa * ratio) then
                nedge = nedge + 1; ed_i(nedge) = idx; ed_j(nedge) = idx2
                nmerge = nmerge + 1
                runlen = runlen + 1
             else
                runlen = 1
             end if
          end do
          if (m .gt. 1) then
             ! junction: the last node of segment m-1 and the first node of
             ! segment m are copies of the same dof
             e0 = line%cseg_e(m-1, c); L0 = line%cseg_L(m-1, c); o0 = line%cseg_o(m-1, c)
             if (o0 .eq. 1) then
                pl = lx
             else
                pl = 1
             end if
             idx = node_index(lx, line%dir(e0), L0, pl) + (e0 - 1) * lx3
             if (o .eq. 1) then
                pl2 = 1
             else
                pl2 = lx
             end if
             idx2 = node_index(lx, d, L, pl2) + (e - 1) * lx3
             nedge = nedge + 1; ed_i(nedge) = idx; ed_j(nedge) = idx2
          end if
       end do
    end do

    ! ---- aggregate keys by label propagation -------------------------------
    allocate(key(n), key_old(n))
    do e = 1, nelv
       do idx = 1, lx3
          id = coef%dof%dof(mod(idx-1,lx)+1, mod((idx-1)/lx,lx)+1, (idx-1)/nl+1, e)
          key(idx + (e - 1) * lx3) = real(id, dp) + 1.0_dp
       end do
    end do
    do it = 1, 10000
       key_old = key
       do k = 1, nedge
          a = min(key(ed_i(k)), key(ed_j(k)))
          key(ed_i(k)) = a; key(ed_j(k)) = a
       end do
       do k = nedge, 1, -1
          a = min(key(ed_i(k)), key(ed_j(k)))
          key(ed_i(k)) = a; key(ed_j(k)) = a
       end do
       call gs_min_key(gs_h, key, n)
       chg = 0.0_rp
       do i = 1, n
          if (key(i) .ne. key_old(i)) chg = 1.0_rp
       end do
       call MPI_Allreduce(chg, chg_glb, 1, MPI_REAL_PRECISION, MPI_MAX, NEKO_COMM, ierr)
       if (chg_glb .eq. 0.0_rp) exit
    end do
    deallocate(ed_i, ed_j, key_old)
    tm(1) = MPI_Wtime()

    ! local distinct keys -> global sorted unique list
    allocate(lk(n))
    do i = 1, n
       lk(i) = nint(key(i), i8)
    end do
    call sort_i8(lk, n)
    nloc = 0
    do i = 1, n
       if (i .eq. 1) then
          nloc = 1
       else if (lk(i) .ne. lk(i-1)) then
          nloc = nloc + 1; lk(nloc) = lk(i)
       end if
    end do
    allocate(counts(pe_size), displs(pe_size))
    call MPI_Allgather(nloc, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    displs(1) = 0
    do i = 2, pe_size
       displs(i) = displs(i-1) + counts(i-1)
    end do
    ntot = sum(counts)
    allocate(allk(ntot))
    call MPI_Allgatherv(lk, nloc, MPI_INTEGER8, allk, counts, displs, &
         MPI_INTEGER8, NEKO_COMM, ierr)
    call sort_i8(allk, ntot)
    allocate(uniq(ntot))
    k = 0
    do i = 1, ntot
       if (i .eq. 1) then
          k = 1; uniq(1) = allk(1)
       else if (allk(i) .ne. allk(i-1)) then
          k = k + 1; uniq(k) = allk(i)
       end if
    end do
    this%nagg = k
    do i = 1, n
       this%aggid(i) = bsearch_i8(uniq, k, nint(key(i), i8))
       if (this%aggid(i) .lt. 1) call neko_error('rs_colagg: aggregate key not found')
    end do
    deallocate(lk, allk, uniq, key)
    tm(2) = MPI_Wtime()

    ! ---- element matrices (local comb products, no gather-scatter) ---------
    allocate(ae(lx3, lx3, nelv), x(n), w(n))
    if (NEKO_BCKND_DEVICE .eq. 1) then
       x_d = c_null_ptr; w_d = c_null_ptr
       call device_map(x, x_d, n)
       call device_map(w, w_d, n)
    end if
    do off = 1, lx3
       do i = 1, n
          x(i) = 0.0_rp
       end do
       do e = 1, nelv
          x(off + (e - 1) * lx3) = 1.0_rp
       end do
       if (NEKO_BCKND_DEVICE .eq. 1) then
          call device_memcpy(x, x_d, n, HOST_TO_DEVICE, sync = .true.)
          call ax%compute(w, x, coef, msh, Xh)
          call device_memcpy(w, w_d, n, DEVICE_TO_HOST, sync = .true.)
       else
          call ax%compute(w, x, coef, msh, Xh)
       end if
       do e = 1, nelv
          do i = 1, lx3
             ae(i, off, e) = w(i + (e - 1) * lx3)
          end do
       end do
    end do
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call device_deassociate(x); call device_free(x_d)
       call device_deassociate(w); call device_free(w_d)
    end if
    deallocate(x, w)

    ! ---- triplets of P^T A P, local merge, global merge --------------------
    nt_loc = nelv * lx3 * lx3
    allocate(tk(max(nt_loc, 1)), tv(max(nt_loc, 1)))
    nt = 0
    do e = 1, nelv
       do j = 1, lx3
          do i = 1, lx3
             a = mask(i + (e - 1) * lx3) * mask(j + (e - 1) * lx3) * ae(i, j, e)
             if (a .ne. 0.0_rp) then
                nt = nt + 1
                tk(nt) = int(this%aggid(i + (e - 1) * lx3) - 1, i8) * int(this%nagg, i8) + &
                     int(this%aggid(j + (e - 1) * lx3) - 1, i8)
                tv(nt) = a
             end if
          end do
       end do
    end do
    deallocate(ae)
    call sort_kv(nt, tk, tv)
    call merge_kv(nt, tk, tv)
    call MPI_Allgather(nt, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    displs(1) = 0
    do i = 2, pe_size
       displs(i) = displs(i-1) + counts(i-1)
    end do
    ntot = sum(counts)
    allocate(gk(ntot), gv(ntot))
    call MPI_Allgatherv(tk, nt, MPI_INTEGER8, gk, counts, displs, MPI_INTEGER8, &
         NEKO_COMM, ierr)
    call MPI_Allgatherv(tv, nt, MPI_DOUBLE_PRECISION, gv, counts, displs, &
         MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    deallocate(tk, tv)
    call sort_kv(ntot, gk, gv)
    call merge_kv(ntot, gk, gv)
    tm(3) = MPI_Wtime()

    ! ---- pure Neumann (no Dirichlet node anywhere): A_c is singular with
    ! the aggregated constant in its null space; pin aggregate 1 (a large
    ! diagonal entry), which keeps A_c SPD and the cycle symmetric --------
    block
      real(kind=rp) :: msum(2)
      msum(1) = 0.0_rp; msum(2) = 0.0_rp
      do i = 1, n
         msum(1) = msum(1) + coef%mult(i,1,1,1) * (1.0_rp - mask(i))
         msum(2) = msum(2) + coef%mult(i,1,1,1)
      end do
      call MPI_Allreduce(MPI_IN_PLACE, msum, 2, MPI_REAL_PRECISION, MPI_SUM, &
           NEKO_COMM, ierr)
      this%pinned = msum(1) .lt. 0.5_rp
    end block
    ! ---- dead aggregates (zero diagonal: all nodes Dirichlet) --------------
    allocate(this%dead(this%nagg), this%dinv(this%nagg))
    this%dead = .true.
    do k = 1, ntot
       i = int(gk(k) / int(this%nagg, i8)) + 1
       j = int(mod(gk(k), int(this%nagg, i8))) + 1
       if (i .eq. j .and. gv(k) .ne. 0.0_rp) this%dead(i) = .false.
    end do
    ! CSR with dead rows/columns removed and identity on dead rows
    allocate(this%ia(this%nagg + 1))
    this%ia = 0
    do k = 1, ntot
       i = int(gk(k) / int(this%nagg, i8)) + 1
       j = int(mod(gk(k), int(this%nagg, i8))) + 1
       if (this%dead(i) .or. this%dead(j)) cycle
       this%ia(i + 1) = this%ia(i + 1) + 1
    end do
    do i = 1, this%nagg
       if (this%dead(i)) this%ia(i + 1) = 1
    end do
    this%ia(1) = 1
    do i = 1, this%nagg
       this%ia(i + 1) = this%ia(i + 1) + this%ia(i)
    end do
    this%nnz = this%ia(this%nagg + 1) - 1
    allocate(this%ja(this%nnz), this%av(this%nnz))
    block
      integer, allocatable :: fill(:)
      allocate(fill(this%nagg))
      fill = this%ia(1:this%nagg)
      do k = 1, ntot
         i = int(gk(k) / int(this%nagg, i8)) + 1
         j = int(mod(gk(k), int(this%nagg, i8))) + 1
         if (this%dead(i) .or. this%dead(j)) cycle
         this%ja(fill(i)) = j; this%av(fill(i)) = gv(k); fill(i) = fill(i) + 1
      end do
      do i = 1, this%nagg
         if (this%dead(i)) then
            this%ja(fill(i)) = i; this%av(fill(i)) = 1.0_rp
         end if
      end do
      deallocate(fill)
    end block
    deallocate(gk, gv)
    if (this%pinned) then
       do k = this%ia(1), this%ia(2) - 1
          if (this%ja(k) .eq. 1) this%av(k) = this%av(k) * 1.0e6_rp
       end do
       if (pe_rank .eq. 0) write(*, '(A)') 'RS_COLAGG pure Neumann problem: aggregate 1 pinned'
    end if
    dmin = huge(1.0_rp); dmax = 0.0_rp; asym = 0.0_rp
    do i = 1, this%nagg
       this%dinv(i) = 0.0_rp
       do k = this%ia(i), this%ia(i + 1) - 1
          if (this%ja(k) .eq. i) this%dinv(i) = this%av(k)
       end do
       if (this%dinv(i) .le. 0.0_rp) call neko_error('rs_colagg: nonpositive coarse diagonal')
       if (.not. this%dead(i)) then
          dmin = min(dmin, this%dinv(i)); dmax = max(dmax, this%dinv(i))
       end if
       this%dinv(i) = 1.0_rp / this%dinv(i)
    end do
    ! symmetry check of the sparse operator
    do i = 1, this%nagg
       do k = this%ia(i), this%ia(i + 1) - 1
          j = this%ja(k)
          a = 0.0_rp
          do ii = this%ia(j), this%ia(j + 1) - 1
             if (this%ja(ii) .eq. i) a = this%av(ii)
          end do
          asym = max(asym, abs(a - this%av(k)))
       end do
    end do

    ! env RS_COLAGG_DUMP=<prefix>: rank 0 writes A_c (full CSR, 1-based) to
    ! <prefix>_<nagg>.csr as a stream: int32 n, nnz, ia(n+1), ja(nnz), real64 av(nnz)
    block
      character(len=512) :: pfx, fname
      integer :: istat, u
      call get_environment_variable('RS_COLAGG_DUMP', pfx, status = istat)
      if (istat .eq. 0 .and. pe_rank .eq. 0) then
         write(fname, '(A,A,I0,A)') trim(pfx), '_', this%nagg, '.csr'
         open(newunit = u, file = trim(fname), access = 'stream', form = 'unformatted', &
              status = 'replace')
         write(u) int(this%nagg, 4), int(this%nnz, 4), int(this%ia, 4), int(this%ja, 4), &
              real(this%av, 8)
         close(u)
         write(*, '(A,A)') 'RS_COLAGG A_c written to ', trim(fname)
      end if
    end block

    ! ---- coarse solver ------------------------------------------------------
    allocate(this%rc(this%nagg), this%xc(this%nagg), this%pc(this%nagg), &
         this%qc(this%nagg), this%zc(this%nagg))
    this%dense = this%nagg .le. this%dense_max
    if (this%dense) then
       allocate(this%ac(this%nagg, this%nagg), this%ipiv(this%nagg))
       this%ac = 0.0_rp
       do i = 1, this%nagg
          do k = this%ia(i), this%ia(i + 1) - 1
             this%ac(i, this%ja(k)) = this%av(k)
          end do
       end do
       if (NEKO_BCKND_DEVICE .ne. 1) then
          ! definiteness check (review 2, finding 11): Cholesky on a copy
          block
            real(kind=dp), allocatable :: chol(:,:)
            allocate(chol(this%nagg, this%nagg))
            chol = this%ac
            call dpotrf('L', this%nagg, chol, this%nagg, info)
            if (info .ne. 0) then
               write(*, '(A,I0)') 'RS_COLAGG dpotrf info ', info
               call neko_error('rs_colagg: A_c is not positive definite')
            end if
            deallocate(chol)
          end block
          call dgetrf(this%nagg, this%nagg, this%ac, this%nagg, this%ipiv, info)
          if (info .ne. 0) then
             write(*, '(A,I0)') 'RS_COLAGG dgetrf info ', info
             call neko_error('rs_colagg: dgetrf failed')
          end if
       end if
       ! on the device the factorization (Cholesky) is done there in
       ! rs_colagg_device_setup; ac stays the unfactored operator here
    end if
    if (this%dense) this%direct = .false.
    if (this%direct) call direct_setup(this)

    tm(4) = MPI_Wtime()
    ! row distribution of the coarse CG (matvec and SGS block per rank)
    allocate(this%rcnt(pe_size), this%rdsp(pe_size))
    do i = 1, pe_size
       this%rcnt(i) = (this%nagg * i) / pe_size - (this%nagg * (i - 1)) / pe_size
       this%rdsp(i) = (this%nagg * (i - 1)) / pe_size
    end do
    this%r0 = this%rdsp(pe_rank + 1) + 1
    this%r1 = this%rdsp(pe_rank + 1) + this%rcnt(pe_rank + 1)
    ! the multilevel preconditioner of the coarse CG is opt-in: as
    ! implemented (unsmoothed pairwise aggregation, GS V-cycle) it cuts
    ! the inner iterations 3-5x but is not faster than SGS-CG (2026-10-08)
    if (.not. this%dense .and. .not. this%direct .and. present(hierarchy)) then
       if (hierarchy) call build_hierarchy(this)
    end if

    ! ---- statistics ----------------------------------------------------------
    allocate(aggsz(this%nagg))
    aggsz = 0
    do i = 1, n
       aggsz(this%aggid(i)) = aggsz(this%aggid(i)) + nint(coef%mult(i,1,1,1) * 1.0_rp)
    end do
    ! mult is 1/multiplicity; count unique nodes per aggregate as a sum of mult
    block
      real(kind=rp), allocatable :: sz(:)
      allocate(sz(this%nagg))
      sz = 0.0_rp
      do i = 1, n
         sz(this%aggid(i)) = sz(this%aggid(i)) + coef%mult(i,1,1,1)
      end do
      call MPI_Allreduce(MPI_IN_PLACE, sz, this%nagg, MPI_REAL_PRECISION, MPI_SUM, &
           NEKO_COMM, ierr)
      maxsz = 0; nsingle = 0
      do i = 1, this%nagg
         maxsz = max(maxsz, nint(sz(i)))
         if (nint(sz(i)) .le. 1) nsingle = nsingle + 1
      end do
      this%merged_fraction = 1.0_rp - real(this%nagg, rp) / max(sum(sz), 1.0_rp)
      deallocate(sz)
    end block
    deallocate(aggsz)
    ndead = count(this%dead)
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0,A,ES9.2,A,F6.3)') 'RS_COLAGG aggregates ', &
            this%nagg, ' dead ', ndead, ' singletons ', nsingle, ' max size ', maxsz, &
            ' nnz ', this%nnz, ' merge edges(rank0) ', nmerge, ' theta ', theta, &
            ' merged fraction ', this%merged_fraction
       write(*, '(A,ES10.3,A,ES10.3,A,ES10.3,A,L1,A,I0)') 'RS_COLAGG A_c diag min ', &
            dmin, ' max ', dmax, ' asym ', asym, ' dense ', this%dense, ' propagation its ', it
    end if

    if (NEKO_BCKND_DEVICE .eq. 1) call rs_colagg_device_setup(this)

    ! ---- symmetry self-test of the cycle --------------------------------------
    block
      real(kind=rp), allocatable :: r1(:), r2(:), z1(:), z2(:)
      real(kind=rp) :: s12, s21, s11, s22
      integer(kind=i8) :: seed
      allocate(r1(n), r2(n), z1(n), z2(n))
      seed = 12345_i8 + int(pe_rank, i8)
      do i = 1, n
         seed = mod(seed * 6364136223846793005_i8 + 1442695040888963407_i8, 2_i8**62)
         r1(i) = real(mod(seed, 1000000_i8), rp) / 1.0e6_rp - 0.5_rp
         seed = mod(seed * 6364136223846793005_i8 + 1442695040888963407_i8, 2_i8**62)
         r2(i) = real(mod(seed, 1000000_i8), rp) / 1.0e6_rp - 0.5_rp
      end do
      ! make them consistent on shared nodes (sum of copies) and masked
      if (NEKO_BCKND_DEVICE .eq. 1) then
         r1_d = c_null_ptr; r2_d = c_null_ptr; z1_d = c_null_ptr; z2_d = c_null_ptr
         call device_map(r1, r1_d, n); call device_map(r2, r2_d, n)
         call device_map(z1, z1_d, n); call device_map(z2, z2_d, n)
         call device_memcpy(r1, r1_d, n, HOST_TO_DEVICE, sync = .true.)
         call device_memcpy(r2, r2_d, n, HOST_TO_DEVICE, sync = .true.)
      end if
      call gs_h%op(r1, n, GS_OP_ADD)
      call gs_h%op(r2, n, GS_OP_ADD)
      call bcp%apply(r1, n)
      call bcp%apply(r2, n)
      call this%solve(z1, r1, n)
      call this%solve(z2, r2, n)
      if (NEKO_BCKND_DEVICE .eq. 1) then
         call device_memcpy(r1, r1_d, n, DEVICE_TO_HOST, sync = .true.)
         call device_memcpy(r2, r2_d, n, DEVICE_TO_HOST, sync = .true.)
         call device_memcpy(z1, z1_d, n, DEVICE_TO_HOST, sync = .true.)
         call device_memcpy(z2, z2_d, n, DEVICE_TO_HOST, sync = .true.)
         call device_deassociate(r1); call device_free(r1_d)
         call device_deassociate(r2); call device_free(r2_d)
         call device_deassociate(z1); call device_free(z1_d)
         call device_deassociate(z2); call device_free(z2_d)
      end if
      s12 = glsc3(z1, coef%mult, r2, n)
      s21 = glsc3(r1, coef%mult, z2, n)
      s11 = glsc3(z1, coef%mult, r1, n)
      s22 = glsc3(z2, coef%mult, r2, n)
      if (pe_rank .eq. 0) then
         write(*, '(A,ES12.5,A,ES12.5,A,ES9.2,A,ES10.3,A,ES10.3)') &
              'RS_COLAGG symmetry (Br1,r2) ', s12, ' (r1,Br2) ', s21, ' rel ', &
              abs(s12 - s21) / max(abs(s12), tiny(1.0_rp)), ' (Br1,r1) ', s11, &
              ' (Br2,r2) ', s22
      end if
      deallocate(r1, r2, z1, z2)
    end block
    this%crs_iters = 0; this%crs_calls = 0; this%crs_fails = 0
    block
      character(len=8) :: env
      integer :: istat
      call get_environment_variable('RS_TIMERS', env, status = istat)
      this%timers = (istat .eq. 0)
    end block
    this%tcyc = 0.0_dp; this%ncyc = 0
    tm(5) = MPI_Wtime()
    call rs_mem_report('rs_colagg after setup')
    if (pe_rank .eq. 0) then
       write(*, '(A,5F8.3)') 'RS_COLAGG setup propagation/keys/triplets/csr+factor/stats+selftest ', &
            tm(1) - tm(0), tm(2) - tm(1), tm(3) - tm(2), tm(4) - tm(3), tm(5) - tm(4)
    end if
    deallocate(mask, counts, displs)
  end subroutine rs_colagg_init_from_components

  !> Heap sort of (key, value) pairs by key.
  subroutine sort_kv(n, k, v)
    integer, intent(in) :: n
    integer(kind=i8), intent(inout) :: k(:)
    real(kind=dp), intent(inout) :: v(:)
    integer :: i, j, c, m
    integer(kind=i8) :: tk
    real(kind=dp) :: tv
    if (n .lt. 2) return
    do i = n / 2, 1, -1
       call sift(i, n)
    end do
    do m = n, 2, -1
       tk = k(1); k(1) = k(m); k(m) = tk
       tv = v(1); v(1) = v(m); v(m) = tv
       call sift(1, m - 1)
    end do
  contains
    subroutine sift(start, last)
      integer, intent(in) :: start, last
      integer :: root, child
      integer(kind=i8) :: rk
      real(kind=rp) :: rv
      root = start
      rk = k(root); rv = v(root)
      do
         child = 2 * root
         if (child .gt. last) exit
         if (child .lt. last) then
            if (k(child + 1) .gt. k(child)) child = child + 1
         end if
         if (k(child) .gt. rk) then
            k(root) = k(child); v(root) = v(child)
            root = child
         else
            exit
         end if
      end do
      k(root) = rk; v(root) = rv
    end subroutine sift
  end subroutine sort_kv

  !> Sum the values of equal consecutive keys; n is updated.
  subroutine merge_kv(n, k, v)
    integer, intent(inout) :: n
    integer(kind=i8), intent(inout) :: k(:)
    real(kind=dp), intent(inout) :: v(:)
    integer :: i, m
    if (n .lt. 1) return
    m = 1
    do i = 2, n
       if (k(i) .eq. k(m)) then
          v(m) = v(m) + v(i)
       else
          m = m + 1
          k(m) = k(i); v(m) = v(i)
       end if
    end do
    n = m
  end subroutine merge_kv

  subroutine sort_i8(a, n)
    integer, intent(in) :: n
    integer(kind=i8), intent(inout) :: a(:)
    integer :: i, m, root, child
    integer(kind=i8) :: t, r
    if (n .lt. 2) return
    do i = n / 2, 1, -1
       call sift_i8(a, i, n)
    end do
    do m = n, 2, -1
       t = a(1); a(1) = a(m); a(m) = t
       call sift_i8(a, 1, m - 1)
    end do
  end subroutine sort_i8

  subroutine sift_i8(a, start, last)
    integer(kind=i8), intent(inout) :: a(:)
    integer, intent(in) :: start, last
    integer :: root, child
    integer(kind=i8) :: r
    root = start
    r = a(root)
    do
       child = 2 * root
       if (child .gt. last) exit
       if (child .lt. last) then
          if (a(child + 1) .gt. a(child)) child = child + 1
       end if
       if (a(child) .gt. r) then
          a(root) = a(child)
          root = child
       else
          exit
       end if
    end do
    a(root) = r
  end subroutine sift_i8

  pure function bsearch_i8(a, n, key) result(pos)
    integer, intent(in) :: n
    integer(kind=i8), intent(in) :: a(:), key
    integer :: pos, lo, hi, mid
    pos = -1; lo = 1; hi = n
    do while (lo .le. hi)
       mid = (lo + hi) / 2
       if (a(mid) .eq. key) then
          pos = mid; return
       else if (a(mid) .lt. key) then
          lo = mid + 1
       else
          hi = mid - 1
       end if
    end do
  end function bsearch_i8

  !> Sparse direct factorization of A_c with MKL PARDISO (upper triangle,
  !! real SPD, METIS nested dissection), the same on every rank.
  subroutine direct_setup(this)
    class(rs_colagg_t), intent(inout) :: this
    type(c_funptr) :: f1, f2
    integer :: i, k, nu
    integer(c_int) :: ph, err, idum(1), mtype
    real(c_double) :: ddum(1)
    real(kind=dp) :: t0, t1
    integer :: info

    if (this%direct_kind .eq. 1) then
       call this%spc%init(this%nagg, this%ia, this%ja, this%av, info)
       if (info .ne. 0) then
          write(*, '(A,I0)') 'RS_COLAGG rs_spchol: nonpositive pivot in supernode ', info
          call neko_error('rs_colagg: sparse Cholesky of A_c failed (not SPD?)')
       end if
       if (pe_rank .eq. 0) then
          write(*, '(A,I0,A,I0,A,A,A,I0,A,3F8.3,A,ES9.2)') 'RS_COLAGG direct coarse solve (rs_spchol): ', &
               this%nagg, ' unknowns, nnz(L) ', this%spc%nnz_l, ', ordering ', &
               trim(merge('nested dissection', 'AMD              ', this%spc%ordering_used .eq. 1)), &
               ', supernodes ', this%spc%nsuper, ', order/symbolic/numeric [s] ', &
               this%spc%t_order, this%spc%t_symbolic, this%spc%t_numeric, ', flops ', this%spc%flops
       end if
       return
    end if
    if (.not. associated(pds_fn)) then
       f1 = rs_dlsym(c_null_ptr, 'pardiso' // c_null_char)
       f2 = rs_dlsym(c_null_ptr, 'pardisoinit' // c_null_char)
       if (.not. c_associated(f1) .or. .not. c_associated(f2)) then
          call neko_error('rs_colagg: colagg_direct needs MKL PARDISO (pardiso not found at run time)')
       end if
       call c_f_procpointer(f1, pds_fn)
       call c_f_procpointer(f2, pds_init_fn)
    end if
    ! upper triangle in CSR, 1-based, columns increasing
    nu = 0
    do i = 1, this%nagg
       do k = this%ia(i), this%ia(i + 1) - 1
          if (this%ja(k) .ge. i) nu = nu + 1
       end do
    end do
    allocate(this%iau(this%nagg + 1), this%jau(nu), this%avu(nu))
    nu = 0
    this%iau(1) = 1
    do i = 1, this%nagg
       do k = this%ia(i), this%ia(i + 1) - 1
          if (this%ja(k) .ge. i) then
             nu = nu + 1
             this%jau(nu) = this%ja(k); this%avu(nu) = this%av(k)
          end if
       end do
       call sort_row(this%jau(this%iau(i):nu), this%avu(this%iau(i):nu))
       this%iau(i + 1) = nu + 1
    end do
    mtype = 2
    this%pds_pt = 0
    call pds_init_fn(this%pds_pt, mtype, this%pds_iparm)
    this%pds_iparm(18) = -1          ! report the nonzeros of the factor
    this%pds_iparm(27) = 0           ! no matrix checker
    t0 = MPI_Wtime()
    ph = 12; err = 0
    call pds_fn(this%pds_pt, 1, 1, mtype, ph, this%nagg, this%avu, this%iau, this%jau, idum, 1, &
         this%pds_iparm, 0, ddum, ddum, err)
    t1 = MPI_Wtime()
    if (err .ne. 0) then
       write(*, '(A,I0)') 'RS_COLAGG PARDISO analysis/factorization error ', err
       call neko_error('rs_colagg: PARDISO factorization of A_c failed (not SPD?)')
    end if
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,I0,A,I0,A,F8.3,A,I0,A)') 'RS_COLAGG direct coarse solve (PARDISO): ', &
            this%nagg, ' unknowns, nnz(upper) ', nu, ', nnz(factor) ', this%pds_iparm(18), &
            ', analysis+factorization ', t1 - t0, ' s, peak memory ', &
            max(this%pds_iparm(15), this%pds_iparm(16) + this%pds_iparm(17)), ' kB'
    end if
  contains
    subroutine sort_row(jj, vv)
      integer, intent(inout) :: jj(:)
      real(kind=dp), intent(inout) :: vv(:)
      integer :: a, b, tj
      real(kind=dp) :: tv
      do a = 2, size(jj)
         tj = jj(a); tv = vv(a); b = a - 1
         do while (b .ge. 1)
            if (jj(b) .le. tj) exit
            jj(b + 1) = jj(b); vv(b + 1) = vv(b); b = b - 1
         end do
         jj(b + 1) = tj; vv(b + 1) = tv
      end do
    end subroutine sort_row
  end subroutine direct_setup

  subroutine direct_solve(this, b, x)
    class(rs_colagg_t), intent(inout) :: this
    real(kind=rp), intent(in) :: b(:)
    real(kind=rp), intent(out) :: x(:)
    integer(c_int) :: ph, err, idum(1)
    real(c_double), allocatable :: bb(:), xx(:)

    ! the factors are double; b and x are in Neko's precision
    allocate(bb(this%nagg), xx(this%nagg))
    bb = b(1:this%nagg)
    if (this%direct_kind .eq. 1) then
       call this%spc%solve(bb, xx)
    else
       ph = 33; err = 0
       call pds_fn(this%pds_pt, 1, 1, 2, ph, this%nagg, this%avu, this%iau, this%jau, idum, 1, &
            this%pds_iparm, 0, bb, xx, err)
       if (err .ne. 0) call neko_error('rs_colagg: PARDISO solve failed')
    end if
    x(1:this%nagg) = real(xx, rp)
  end subroutine direct_solve

  !> Redundant coarse solve x = A_c^{-1} b (dense LU or Jacobi-CG).
  subroutine coarse_solve(this, b, x)
    class(rs_colagg_t), intent(inout) :: this
    real(kind=rp), intent(in) :: b(:)
    real(kind=rp), intent(out) :: x(:)
    integer :: i, k, it, info, m
    real(kind=rp) :: rz, rz_new, alpha, beta, pq, rr, bb

    m = this%nagg
    this%crs_calls = this%crs_calls + 1
    if (this%dense) then
       block
         real(kind=dp), allocatable :: xd(:)
         allocate(xd(m)); xd = b(1:m)
         call dgetrs('N', m, 1, this%ac, m, this%ipiv, xd, m, info)
         x(1:m) = real(xd, rp)
       end block
       return
    end if
    if (this%direct) then
       call direct_solve(this, b, x)
       return
    end if
    if (this%parallel_crs .and. pe_size .gt. 1 .and. this%nlv .le. 0) then
       call coarse_solve_dist(this, b, x)
       return
    end if
    bb = 0.0_rp
    do i = 1, m
       x(i) = 0.0_rp
       this%rc(i) = b(i)
       bb = bb + b(i) * b(i)
    end do
    if (bb .le. 0.0_rp) return
    call vcycle_apply(this, this%rc, this%zc)
    do i = 1, m
       this%pc(i) = this%zc(i)
    end do
    rz = dot_product(this%rc(1:m), this%zc(1:m))
    do it = 1, this%crs_max_iter
       do i = 1, m
          this%qc(i) = 0.0_rp
          do k = this%ia(i), this%ia(i + 1) - 1
             this%qc(i) = this%qc(i) + this%av(k) * this%pc(this%ja(k))
          end do
       end do
       pq = dot_product(this%pc(1:m), this%qc(1:m))
       alpha = rz / pq
       rr = 0.0_rp
       do i = 1, m
          x(i) = x(i) + alpha * this%pc(i)
          this%rc(i) = this%rc(i) - alpha * this%qc(i)
          rr = rr + this%rc(i) * this%rc(i)
       end do
       this%crs_iters = this%crs_iters + 1
       if (sqrt(rr) .le. this%crs_tol * sqrt(bb)) return
       call vcycle_apply(this, this%rc, this%zc)
       rz_new = dot_product(this%rc(1:m), this%zc(1:m))
       beta = rz_new / rz
       rz = rz_new
       do i = 1, m
          this%pc(i) = this%zc(i) + beta * this%pc(i)
       end do
    end do
    this%crs_fails = this%crs_fails + 1
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,ES9.2)') 'RS_COLAGG WARNING: coarse SGS-CG not converged in ', &
            this%crs_max_iter, ' iterations, residual ', sqrt(rr) / sqrt(bb)
    end if
  end subroutine coarse_solve

  !> Distributed CG on the replicated A_c: every rank updates its own rows
  !! (matvec, block symmetric Gauss-Seidel preconditioner on its block),
  !! the search direction is gathered (Allgatherv) once per iteration and
  !! the two inner products reduced together. x is replicated on return.
  subroutine coarse_solve_dist(this, b, x)
    class(rs_colagg_t), intent(inout) :: this
    real(kind=rp), intent(in) :: b(:)
    real(kind=rp), intent(out) :: x(:)
    integer :: i, k, it, m, ierr, j
    real(kind=rp) :: rz, rz_new, alpha, beta, acc, loc(2), glb(2), bb
    real(kind=rp), allocatable :: rl(:), zl(:), pl(:), ql(:), xl(:)

    m = this%nagg
    allocate(rl(this%r0:this%r1), zl(this%r0:this%r1), pl(this%r0:this%r1), &
         ql(this%r0:this%r1), xl(this%r0:this%r1))
    loc(1) = 0.0_rp
    do i = this%r0, this%r1
       xl(i) = 0.0_rp; rl(i) = b(i); loc(1) = loc(1) + b(i) * b(i)
    end do
    call MPI_Allreduce(loc(1), bb, 1, MPI_REAL_PRECISION, MPI_SUM, NEKO_COMM, ierr)
    if (bb .le. 0.0_rp) then
       x(1:m) = 0.0_rp
       return
    end if
    call block_sgs(this, rl, zl)
    loc(1) = 0.0_rp
    do i = this%r0, this%r1
       pl(i) = zl(i); loc(1) = loc(1) + rl(i) * zl(i)
    end do
    call MPI_Allreduce(loc(1), rz, 1, MPI_REAL_PRECISION, MPI_SUM, NEKO_COMM, ierr)
    do it = 1, this%crs_max_iter
       ! gather the search direction into the replicated work vector
       call MPI_Allgatherv(pl, this%r1 - this%r0 + 1, MPI_REAL_PRECISION, this%pc, &
            this%rcnt, this%rdsp, MPI_REAL_PRECISION, NEKO_COMM, ierr)
       loc = 0.0_rp
       do i = this%r0, this%r1
          acc = 0.0_rp
          do k = this%ia(i), this%ia(i + 1) - 1
             acc = acc + this%av(k) * this%pc(this%ja(k))
          end do
          ql(i) = acc
          loc(1) = loc(1) + pl(i) * acc
       end do
       call MPI_Allreduce(loc(1), glb(1), 1, MPI_REAL_PRECISION, MPI_SUM, NEKO_COMM, ierr)
       alpha = rz / glb(1)
       loc(1) = 0.0_rp
       do i = this%r0, this%r1
          xl(i) = xl(i) + alpha * pl(i)
          rl(i) = rl(i) - alpha * ql(i)
          loc(1) = loc(1) + rl(i) * rl(i)
       end do
       call block_sgs(this, rl, zl)
       loc(2) = 0.0_rp
       do i = this%r0, this%r1
          loc(2) = loc(2) + rl(i) * zl(i)
       end do
       call MPI_Allreduce(loc, glb, 2, MPI_REAL_PRECISION, MPI_SUM, NEKO_COMM, ierr)
       this%crs_iters = this%crs_iters + 1
       if (sqrt(glb(1)) .le. this%crs_tol * sqrt(bb)) exit
       rz_new = glb(2)
       beta = rz_new / rz
       rz = rz_new
       do i = this%r0, this%r1
          pl(i) = zl(i) + beta * pl(i)
       end do
       if (it .eq. this%crs_max_iter) then
          this%crs_fails = this%crs_fails + 1
          if (pe_rank .eq. 0) write(*, '(A,I0)') 'RS_COLAGG WARNING: distributed coarse CG not converged in ', it
       end if
    end do
    call MPI_Allgatherv(xl, this%r1 - this%r0 + 1, MPI_REAL_PRECISION, x, &
         this%rcnt, this%rdsp, MPI_REAL_PRECISION, NEKO_COMM, ierr)
    deallocate(rl, zl, pl, ql, xl)
  end subroutine coarse_solve_dist

  !> Symmetric Gauss-Seidel on this rank's diagonal block of A_c.
  subroutine block_sgs(this, r, z)
    class(rs_colagg_t), intent(in) :: this
    real(kind=rp), intent(in) :: r(this%r0:)
    real(kind=rp), intent(out) :: z(this%r0:)
    integer :: i, k, j
    real(kind=rp) :: acc
    do i = this%r0, this%r1
       acc = r(i)
       do k = this%ia(i), this%ia(i + 1) - 1
          j = this%ja(k)
          if (j .ge. this%r0 .and. j .lt. i) acc = acc - this%av(k) * z(j)
       end do
       z(i) = acc * this%dinv(i)
    end do
    do i = this%r1, this%r0, -1
       acc = z(i) / this%dinv(i)
       do k = this%ia(i), this%ia(i + 1) - 1
          j = this%ja(k)
          if (j .gt. i .and. j .le. this%r1) acc = acc - this%av(k) * z(j)
       end do
       z(i) = acc * this%dinv(i)
    end do
  end subroutine block_sgs

  !> Greedy pairwise aggregation of one CSR level by the strongest
  !! (most negative) coupling; unmatched nodes join their strongest
  !! neighbour's aggregate (up to 8 members); decoupled rows form one
  !! aggregate. agg(i) in 1..nc.
  subroutine pairwise(lv, agg, nc)
    type(agg_lvl_t), intent(in) :: lv
    integer, intent(out) :: agg(:)
    integer, intent(out) :: nc
    integer :: i, j, k, best, bestany, ndead
    real(kind=rp) :: bestv, bestvany
    integer, allocatable :: sz(:)
    allocate(sz(lv%n))
    agg(1:lv%n) = 0; nc = 0; sz = 0; ndead = 0
    do i = 1, lv%n
       if (agg(i) .ne. 0) cycle
       if (lv%ia(i + 1) - lv%ia(i) .le. 1) then
          if (ndead .eq. 0) then
             nc = nc + 1; ndead = nc
          end if
          agg(i) = ndead
          cycle
       end if
       best = 0; bestv = 0.0_rp; bestany = 0; bestvany = 0.0_rp
       do k = lv%ia(i), lv%ia(i + 1) - 1
          j = lv%ja(k)
          if (j .eq. i) cycle
          if (lv%av(k) .lt. bestvany .and. agg(j) .ne. 0) then
             if (agg(j) .ne. ndead .and. sz(agg(j)) .lt. 8) then
                bestvany = lv%av(k); bestany = j
             end if
          end if
          if (agg(j) .ne. 0) cycle
          if (lv%av(k) .lt. bestv) then
             bestv = lv%av(k); best = j
          end if
       end do
       if (best .gt. 0) then
          nc = nc + 1; agg(i) = nc; agg(best) = nc; sz(nc) = 2
       else if (bestany .gt. 0) then
          agg(i) = agg(bestany); sz(agg(i)) = sz(agg(i)) + 1
       else
          nc = nc + 1; agg(i) = nc; sz(nc) = 1
       end if
    end do
    deallocate(sz)
  end subroutine pairwise

  !> Aggregation hierarchy on the replicated A_c: two passes of pairwise
  !! aggregation by the strongest (most negative) coupling per level
  !! (aggregates of up to four), Galerkin coarse operators, down to
  !! dense_max unknowns (dense LU). Redundant on every rank.
  subroutine build_hierarchy(this)
    class(rs_colagg_t), intent(inout) :: this
    integer, parameter :: MAXLV = 12
    integer :: l, i, j, k, n, nc, nt, info, pass, best
    real(kind=rp) :: bestv
    integer, allocatable :: a1(:), a2(:)
    integer(kind=i8), allocatable :: tk(:)
    real(kind=dp), allocatable :: tv(:)
    type(agg_lvl_t), allocatable :: tmp(:)

    allocate(this%lv(MAXLV))
    ! level 1 = A_c
    this%lv(1)%n = this%nagg; this%lv(1)%nnz = this%nnz
    allocate(this%lv(1)%ia(this%nagg + 1), this%lv(1)%ja(this%nnz), this%lv(1)%av(this%nnz), &
         this%lv(1)%dinv(this%nagg))
    this%lv(1)%ia = this%ia; this%lv(1)%ja = this%ja; this%lv(1)%av = this%av
    this%lv(1)%dinv = this%dinv
    this%nlv = 1
    do l = 1, MAXLV - 1
       n = this%lv(l)%n
       if (n .le. 300) exit              ! coarsest level: dense LU of <= ~300 unknowns
       allocate(a1(n), a2(n))
       ! two greedy pairwise passes with the join rule (an unmatched node
       ! joins the aggregate of its strongest neighbour, size <= 8);
       ! decoupled (dead, identity) rows all go into one aggregate
       call pairwise(this%lv(l), a1, nc)
       block
         integer :: np, i2
         integer, allocatable :: pia(:), pja(:), pagg(:)
         real(kind=rp), allocatable :: pav(:)
         integer(kind=i8), allocatable :: pk(:)
         real(kind=dp), allocatable :: pv(:)
         type(agg_lvl_t) :: pl
         ! the pair graph as a CSR (Galerkin of the pair map)
         np = nc
         allocate(pk(this%lv(l)%nnz), pv(this%lv(l)%nnz))
         do i = 1, n
            do k = this%lv(l)%ia(i), this%lv(l)%ia(i + 1) - 1
               pk(k) = int(a1(i) - 1, i8) * int(np, i8) + int(a1(this%lv(l)%ja(k)) - 1, i8)
               pv(k) = this%lv(l)%av(k)
            end do
         end do
         i2 = this%lv(l)%nnz
         call sort_kv(i2, pk, pv)
         call merge_kv(i2, pk, pv)
         pl%n = np; pl%nnz = i2
         allocate(pl%ia(np + 1), pl%ja(i2), pl%av(i2), pl%dinv(np))
         pl%ia = 0
         do k = 1, i2
            i = int(pk(k) / int(np, i8)) + 1
            pl%ia(i + 1) = pl%ia(i + 1) + 1
         end do
         pl%ia(1) = 1
         do i = 1, np
            pl%ia(i + 1) = pl%ia(i + 1) + pl%ia(i)
         end do
         do k = 1, i2
            pl%ja(k) = int(mod(pk(k), int(np, i8))) + 1
            pl%av(k) = pv(k)
         end do
         pl%dinv = 1.0_rp
         allocate(pagg(np))
         call pairwise(pl, pagg, nc)
         do i = 1, n
            a2(i) = pagg(a1(i))
         end do
         deallocate(pk, pv, pagg, pl%ia, pl%ja, pl%av, pl%dinv)
       end block
       if (nc .gt. 0.7_rp * n) then
          deallocate(a1, a2)
          exit                     ! no useful coarsening left
       end if
       allocate(this%lv(l)%agg(n))
       this%lv(l)%agg = a2
       ! Galerkin coarse operator via triplets
       nt = this%lv(l)%nnz
       allocate(tk(nt), tv(nt))
       do i = 1, n
          do k = this%lv(l)%ia(i), this%lv(l)%ia(i + 1) - 1
             tk(k) = int(a2(i) - 1, i8) * int(nc, i8) + int(a2(this%lv(l)%ja(k)) - 1, i8)
             tv(k) = this%lv(l)%av(k)
          end do
       end do
       call sort_kv(nt, tk, tv)
       call merge_kv(nt, tk, tv)
       this%lv(l+1)%n = nc; this%lv(l+1)%nnz = nt
       allocate(this%lv(l+1)%ia(nc + 1), this%lv(l+1)%ja(nt), this%lv(l+1)%av(nt), &
            this%lv(l+1)%dinv(nc))
       this%lv(l+1)%ia = 0
       do k = 1, nt
          i = int(tk(k) / int(nc, i8)) + 1
          this%lv(l+1)%ia(i + 1) = this%lv(l+1)%ia(i + 1) + 1
       end do
       this%lv(l+1)%ia(1) = 1
       do i = 1, nc
          this%lv(l+1)%ia(i + 1) = this%lv(l+1)%ia(i + 1) + this%lv(l+1)%ia(i)
       end do
       do k = 1, nt
          this%lv(l+1)%ja(k) = int(mod(tk(k), int(nc, i8))) + 1
          this%lv(l+1)%av(k) = tv(k)
       end do
       do i = 1, nc
          this%lv(l+1)%dinv(i) = 0.0_rp
          do k = this%lv(l+1)%ia(i), this%lv(l+1)%ia(i + 1) - 1
             if (this%lv(l+1)%ja(k) .eq. i) this%lv(l+1)%dinv(i) = this%lv(l+1)%av(k)
          end do
          if (this%lv(l+1)%dinv(i) .le. 0.0_rp) call neko_error('rs_colagg: hierarchy diagonal')
          this%lv(l+1)%dinv(i) = 1.0_rp / this%lv(l+1)%dinv(i)
       end do
       deallocate(tk, tv, a1, a2)
       this%nlv = l + 1
    end do
    ! work vectors and the dense coarsest solve
    do l = 1, this%nlv
       allocate(this%lv(l)%x(this%lv(l)%n), this%lv(l)%r(this%lv(l)%n), this%lv(l)%b(this%lv(l)%n))
    end do
    this%ncrs = this%lv(this%nlv)%n
    allocate(this%acc(this%ncrs, this%ncrs), this%ipivc(this%ncrs))
    this%acc = 0.0_rp
    do i = 1, this%ncrs
       do k = this%lv(this%nlv)%ia(i), this%lv(this%nlv)%ia(i + 1) - 1
          this%acc(i, this%lv(this%nlv)%ja(k)) = this%lv(this%nlv)%av(k)
       end do
    end do
    call dgetrf(this%ncrs, this%ncrs, this%acc, this%ncrs, this%ipivc, info)
    if (info .ne. 0) call neko_error('rs_colagg: dgetrf failed on the coarsest level')
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A)', advance = 'no') 'RS_COLAGG hierarchy levels ', this%nlv, ' sizes'
       do l = 1, this%nlv
          write(*, '(A,I0)', advance = 'no') ' ', this%lv(l)%n
       end do
       write(*, '(A)') ''
    end if
  end subroutine build_hierarchy

  !> One symmetric V-cycle of the hierarchy: z = V r (forward Gauss-Seidel
  !! from zero, coarse correction, backward Gauss-Seidel), SPD.
  subroutine vcycle_apply(this, r, z)
    class(rs_colagg_t), intent(inout) :: this
    real(kind=rp), intent(in) :: r(:)
    real(kind=rp), intent(out) :: z(:)
    integer :: l, i, k, n, info
    real(kind=rp) :: acc

    if (this%nlv .le. 0) then
       call sgs_apply(this, r, z)
       return
    end if
    this%lv(1)%b(1:this%lv(1)%n) = r(1:this%lv(1)%n)
    ! down: forward GS from zero, residual, restrict
    do l = 1, this%nlv - 1
       n = this%lv(l)%n
       do i = 1, n
          acc = this%lv(l)%b(i)
          do k = this%lv(l)%ia(i), this%lv(l)%ia(i + 1) - 1
             if (this%lv(l)%ja(k) .lt. i) acc = acc - this%lv(l)%av(k) * this%lv(l)%x(this%lv(l)%ja(k))
          end do
          this%lv(l)%x(i) = acc * this%lv(l)%dinv(i)
       end do
       do i = 1, n
          acc = this%lv(l)%b(i)
          do k = this%lv(l)%ia(i), this%lv(l)%ia(i + 1) - 1
             acc = acc - this%lv(l)%av(k) * this%lv(l)%x(this%lv(l)%ja(k))
          end do
          this%lv(l)%r(i) = acc
       end do
       this%lv(l+1)%b(1:this%lv(l+1)%n) = 0.0_rp
       do i = 1, n
          this%lv(l+1)%b(this%lv(l)%agg(i)) = this%lv(l+1)%b(this%lv(l)%agg(i)) + this%lv(l)%r(i)
       end do
    end do
    ! coarsest: dense
    l = this%nlv
    block
      real(kind=dp), allocatable :: xd(:)
      allocate(xd(this%ncrs)); xd = this%lv(l)%b(1:this%ncrs)
      call dgetrs('N', this%ncrs, 1, this%acc, this%ncrs, this%ipivc, xd, this%ncrs, info)
      this%lv(l)%x(1:this%ncrs) = real(xd, rp)
    end block
    ! up: prolongate, backward GS
    do l = this%nlv - 1, 1, -1
       n = this%lv(l)%n
       do i = 1, n
          this%lv(l)%x(i) = this%lv(l)%x(i) + this%lv(l+1)%x(this%lv(l)%agg(i))
       end do
       do i = n, 1, -1
          acc = this%lv(l)%b(i)
          do k = this%lv(l)%ia(i), this%lv(l)%ia(i + 1) - 1
             if (this%lv(l)%ja(k) .ne. i) acc = acc - this%lv(l)%av(k) * this%lv(l)%x(this%lv(l)%ja(k))
          end do
          this%lv(l)%x(i) = acc * this%lv(l)%dinv(i)
       end do
    end do
    z(1:this%lv(1)%n) = this%lv(1)%x(1:this%lv(1)%n)
  end subroutine vcycle_apply

  !> z = (symmetric Gauss-Seidel)^{-1} r on the replicated CSR A_c:
  !! forward sweep, then backward sweep (an SPD preconditioner).
  subroutine sgs_apply(this, r, z)
    class(rs_colagg_t), intent(in) :: this
    real(kind=rp), intent(in) :: r(:)
    real(kind=rp), intent(out) :: z(:)
    integer :: i, k, m
    real(kind=rp) :: acc
    m = this%nagg
    ! forward: (D + L) y = r
    do i = 1, m
       acc = r(i)
       do k = this%ia(i), this%ia(i + 1) - 1
          if (this%ja(k) .lt. i) acc = acc - this%av(k) * z(this%ja(k))
       end do
       z(i) = acc * this%dinv(i)
    end do
    ! backward: (D + U) z = D y
    do i = m, 1, -1
       acc = z(i) / this%dinv(i)
       do k = this%ia(i), this%ia(i + 1) - 1
          if (this%ja(k) .gt. i) acc = acc - this%av(k) * z(this%ja(k))
       end do
       z(i) = acc * this%dinv(i)
    end do
  end subroutine sgs_apply

  !> z = B r: symmetric V(1,1) cycle from zero.
  subroutine rs_colagg_solve(this, z, r, n)
    class(rs_colagg_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    type(ksp_monitor_t) :: res
    integer :: i, ierr

    if (NEKO_BCKND_DEVICE .eq. 1) then
       call rs_colagg_solve_device(this, z, r, n)
       return
    end if
    ! pre-smoothing from zero (cheby's zero_initial_guess only skips the
    ! residual; the iterate must be zeroed here)
    do i = 1, n
       this%zf%x(i,1,1,1) = 0.0_rp
    end do
    this%cheby%zero_initial_guess = .true.
    res = this%cheby%solve(this%ax, this%zf, r, n, this%coef, this%bcp, &
         this%gs_h, niter = this%nsmooth)
    ! residual
    call this%ax%compute(this%w, this%zf%x, this%coef, this%msh, this%Xh)
    call this%gs_h%op(this%w, n, GS_OP_ADD)
    call this%bcp%apply(this%w, n)
    do i = 1, n
       this%w(i) = r(i) - this%w(i)
    end do
    ! coarse correction: rc = P^T w (sum over unique nodes)
    this%zc = 0.0_rp
    do i = 1, n
       this%zc(this%aggid(i)) = this%zc(this%aggid(i)) + this%coef%mult(i,1,1,1) * this%w(i)
    end do
    call MPI_Allreduce(MPI_IN_PLACE, this%zc, this%nagg, MPI_REAL_PRECISION, &
         MPI_SUM, NEKO_COMM, ierr)
    where (this%dead) this%zc = 0.0_rp
    call coarse_solve(this, this%zc, this%xc)
    do i = 1, n
       this%zf%x(i,1,1,1) = this%zf%x(i,1,1,1) + this%xc(this%aggid(i))
    end do
    call this%bcp%apply(this%zf%x, n)
    ! post-smoothing from the current iterate
    this%cheby%zero_initial_guess = .false.
    res = this%cheby%solve(this%ax, this%zf, r, n, this%coef, this%bcp, &
         this%gs_h, niter = this%nsmooth)
    do i = 1, n
       z(i) = this%zf%x(i,1,1,1)
    end do
  end subroutine rs_colagg_solve

  !> Device copies of the coarse level (aggregate map, CSR, live mask) and
  !! the work vectors.
  !> map the dense coarse operator (2-D, column-major) to the device as a
  !! flat array and copy it (target dummy so that c_loc is legal)
  subroutine map_dense(a, a_d, n2)
    real(kind=dp), intent(inout), target :: a(:,:)
    type(c_ptr), intent(inout) :: a_d
    integer, intent(in) :: n2
    real(kind=dp), pointer :: af(:)
    call c_f_pointer(c_loc(a), af, [n2])
    call device_map(af, a_d, n2)
    call device_memcpy(af, a_d, n2, HOST_TO_DEVICE, sync = .true.)
  end subroutine map_dense

  subroutine unmap_dense(a, a_d, n2)
    real(kind=dp), intent(inout), target :: a(:,:)
    type(c_ptr), intent(inout) :: a_d
    integer, intent(in) :: n2
    real(kind=dp), pointer :: af(:)
    call c_f_pointer(c_loc(a), af, [n2])
    call device_deassociate(af)
    call device_free(a_d)
    a_d = c_null_ptr
  end subroutine unmap_dense

  subroutine rs_colagg_device_setup(this)
    class(rs_colagg_t), intent(inout) :: this
    integer :: i, n, m
    n = this%n; m = this%nagg
    allocate(this%agg0(n), this%ia0(m + 1), this%ja0(this%nnz), this%live(m), &
         this%rcd(m), this%xcd(m), this%pcd(m), this%qcd(m), this%zcd(m))
    do i = 1, n
       this%agg0(i) = this%aggid(i) - 1
    end do
    this%ia0 = this%ia - 1
    this%ja0 = this%ja - 1
    do i = 1, m
       if (this%dead(i)) then
          this%live(i) = 0.0_rp
       else
          this%live(i) = 1.0_rp
       end if
    end do
    this%rcd = 0.0_rp; this%xcd = 0.0_rp; this%pcd = 0.0_rp; this%qcd = 0.0_rp; this%zcd = 0.0_rp
    this%dense_dev = .false.
    if (this%dense) then
       block
         integer :: info
         if (int(m, i8) * int(m, i8) .ge. 2_i8**31) call neko_error('rs_colagg: dense coarse operator too large for the device')
         if (rp .ne. dp) call neko_error('rs_colagg: the dense device coarse solve needs a double-precision ' // &
              'Neko; use colagg_direct (colagg_dense_max 0)')
         call map_dense(this%ac, this%ac_d, m * m)
         info = rs_colagg_cuda_potrf(this%ac_d, int(m, c_int))
         call rs_line_cuda_sync()
         if (info .ne. 0) then
            write(*, '(A,I0)') 'RS_COLAGG device potrf info ', info
            call neko_error('rs_colagg: A_c is not positive definite (device Cholesky)')
         end if
         this%dense_dev = .true.
         if (pe_rank .eq. 0) write(*, '(A,I0,A)') 'RS_COLAGG dense coarse solve on the device: ', m, &
              ' unknowns, Cholesky factor resident'
       end block
    end if
    call device_map(this%agg0, this%agg0_d, n)
    call device_memcpy(this%agg0, this%agg0_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%ia0, this%ia0_d, m + 1)
    call device_memcpy(this%ia0, this%ia0_d, m + 1, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%ja0, this%ja0_d, this%nnz)
    call device_memcpy(this%ja0, this%ja0_d, this%nnz, HOST_TO_DEVICE, sync = .true.)
    allocate(this%avr(this%nnz)); this%avr = real(this%av, rp)
    call device_map(this%avr, this%av_d, this%nnz)
    call device_memcpy(this%avr, this%av_d, this%nnz, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%dinv, this%dinv_d, m)
    call device_memcpy(this%dinv, this%dinv_d, m, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%live, this%live_d, m)
    call device_memcpy(this%live, this%live_d, m, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%rcd, this%rcd_d, m)
    call device_memcpy(this%rcd, this%rcd_d, m, HOST_TO_DEVICE, sync = .true.)
    allocate(this%onesc(m))
    this%onesc = 1.0_rp
    call device_map(this%onesc, this%onesc_d, m)
    call device_memcpy(this%onesc, this%onesc_d, m, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%xcd, this%xcd_d, m)
    call device_memcpy(this%xcd, this%xcd_d, m, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%pcd, this%pcd_d, m)
    call device_memcpy(this%pcd, this%pcd_d, m, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%qcd, this%qcd_d, m)
    call device_memcpy(this%qcd, this%qcd_d, m, HOST_TO_DEVICE, sync = .true.)
    call device_map(this%zcd, this%zcd_d, m)
    call device_memcpy(this%zcd, this%zcd_d, m, HOST_TO_DEVICE, sync = .true.)
    this%crs_on_device = .true.
    if (pe_rank .eq. 0) write(*, '(A,I0,A,L1)') 'RS_COLAGG coarse level on the device: ', m, &
         ' aggregates, dense ', this%dense
  end subroutine rs_colagg_device_setup

  !> Jacobi-preconditioned CG on the device for the sparse A_c (every rank
  !! holds the full replicated matrix; the solve is redundant).
  subroutine coarse_solve_device(this)
    class(rs_colagg_t), intent(inout), target :: this
    integer :: m, it
    real(kind=rp) :: rz, rz_new, alpha, beta, pq, rr, bb
    ! the coarse system is replicated and its right-hand side identical on
    ! every rank (allreduce + broadcast), so the CG runs rank-locally
    m = this%nagg
    this%crs_calls = this%crs_calls + 1
    bb = cdot(this%rcd_d, this%rcd_d)
    call device_rzero(this%xcd_d, m)
    if (bb .le. 0.0_rp) return
    ! z = D^{-1} r; p = z
    call device_col3(this%zcd_d, this%dinv_d, this%rcd_d, m)
    call device_copy(this%pcd_d, this%zcd_d, m)
    rz = cdot(this%rcd_d, this%zcd_d)
    do it = 1, this%crs_max_iter
       call device_sync()
       call rs_colagg_cuda_spmv(this%ia0_d, this%ja0_d, this%av_d, this%pcd_d, this%qcd_d, int(m, c_int))
       call rs_line_cuda_sync()
       pq = cdot(this%pcd_d, this%qcd_d)
       alpha = rz / pq
       call device_add2s2(this%xcd_d, this%pcd_d, alpha, m)
       call device_add2s2(this%rcd_d, this%qcd_d, -alpha, m)
       rr = cdot(this%rcd_d, this%rcd_d)
       this%crs_iters = this%crs_iters + 1
       if (sqrt(rr) .le. this%crs_tol * sqrt(bb)) return
       call device_col3(this%zcd_d, this%dinv_d, this%rcd_d, m)
       rz_new = cdot(this%rcd_d, this%zcd_d)
       beta = rz_new / rz
       rz = rz_new
       call device_add2s1(this%pcd_d, this%zcd_d, beta, m)
    end do
    this%crs_fails = this%crs_fails + 1
    if (pe_rank .eq. 0) write(*, '(A,I0)') 'RS_COLAGG WARNING: device coarse CG not converged in ', &
         this%crs_max_iter
  contains
    function cdot(a_d, b_d) result(d)
      type(c_ptr), intent(in) :: a_d, b_d
      real(kind=rp) :: d
      type(c_ptr) :: ca, cb
      ca = a_d; cb = b_d
      d = device_vlsc3(ca, cb, this%onesc_d, m)
    end function cdot
  end subroutine coarse_solve_device

  !> Device version of the cycle: smoothing and residual on the device,
  !! restriction / coarse solve / prolongation on the host (N = 1 vectors
  !! are small), three copies per application.
  subroutine rs_colagg_solve_device(this, z, r, n)
    class(rs_colagg_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    type(ksp_monitor_t) :: res
    type(c_ptr) :: z_d, r_d
    integer :: i, ierr
    real(kind=dp) :: tt(0:6)

    z_d = device_get_ptr(z)
    r_d = device_get_ptr(r)
    if (this%timers) then
       call device_sync(); tt(0) = MPI_Wtime()
    end if
    call device_rzero(this%zf%x_d, n)
    res = this%cheby_dev%solve(this%ax, this%zf, r, n, this%coef, this%bcp, &
         this%gs_h, niter = this%nsmooth)
    if (this%timers) then
       call device_sync(); tt(1) = MPI_Wtime()
    end if
    call this%ax%compute(this%w, this%zf%x, this%coef, this%msh, this%Xh)
    call this%gs_h%op(this%w, n, GS_OP_ADD)
    call this%bcp%apply(this%w, n)
    call device_sub3(this%w_d, r_d, this%w_d, n)
    if (this%timers) then
       call device_sync(); tt(2) = MPI_Wtime()
    end if
    if (this%crs_on_device) then
       ! restriction, coarse solve and prolongation on the device
       call device_rzero(this%rcd_d, this%nagg)
       call device_sync()
       call rs_colagg_cuda_restrict(this%w_d, this%coef%mult_d, this%agg0_d, this%rcd_d, int(n, c_int))
       call rs_line_cuda_sync()
       if (pe_size .gt. 1) then
          call device_memcpy(this%rcd, this%rcd_d, this%nagg, DEVICE_TO_HOST, sync = .true.)
          call MPI_Allreduce(MPI_IN_PLACE, this%rcd, this%nagg, MPI_REAL_PRECISION, &
               MPI_SUM, NEKO_COMM, ierr)
          ! bitwise the same right-hand side everywhere (the reduction
          ! order may differ per rank): the replicated CG then stays in step
          call MPI_Bcast(this%rcd, this%nagg, MPI_REAL_PRECISION, 0, NEKO_COMM, ierr)
          call device_memcpy(this%rcd, this%rcd_d, this%nagg, HOST_TO_DEVICE, sync = .true.)
       end if
       call device_col3(this%rcd_d, this%rcd_d, this%live_d, this%nagg)   ! zero the dead
       if (this%timers) then
          call device_sync(); tt(3) = MPI_Wtime()
       end if
       if (this%dense .and. this%dense_dev) then
          ! two triangular solves on the device with the resident factor
          call device_copy(this%xcd_d, this%rcd_d, this%nagg)
          call device_sync()
          call rs_colagg_cuda_trsv2(this%ac_d, this%xcd_d, int(this%nagg, c_int))
          call rs_line_cuda_sync()
       else if (this%dense .or. this%crs_host .or. this%direct) then
          call device_memcpy(this%rcd, this%rcd_d, this%nagg, DEVICE_TO_HOST, sync = .true.)
          call coarse_solve(this, this%rcd, this%xcd)
          call device_memcpy(this%xcd, this%xcd_d, this%nagg, HOST_TO_DEVICE, sync = .true.)
       else
          call coarse_solve_device(this)
       end if
       call device_sync()
       if (this%timers) tt(4) = MPI_Wtime()
       call rs_colagg_cuda_prolong(this%zf%x_d, this%xcd_d, this%agg0_d, int(n, c_int))
       call rs_line_cuda_sync()
    else
       call device_memcpy(this%w, this%w_d, n, DEVICE_TO_HOST, sync = .true.)
       this%zc = 0.0_rp
       do i = 1, n
          this%zc(this%aggid(i)) = this%zc(this%aggid(i)) + this%coef%mult(i,1,1,1) * this%w(i)
       end do
       call MPI_Allreduce(MPI_IN_PLACE, this%zc, this%nagg, MPI_REAL_PRECISION, &
            MPI_SUM, NEKO_COMM, ierr)
       where (this%dead) this%zc = 0.0_rp
       call coarse_solve(this, this%zc, this%xc)
       call device_memcpy(this%zf%x, this%zf%x_d, n, DEVICE_TO_HOST, sync = .true.)
       do i = 1, n
          this%zf%x(i,1,1,1) = this%zf%x(i,1,1,1) + this%xc(this%aggid(i))
       end do
       call device_memcpy(this%zf%x, this%zf%x_d, n, HOST_TO_DEVICE, sync = .true.)
    end if
    call this%bcp%apply(this%zf%x, n)
    if (this%timers) then
       call device_sync(); tt(5) = MPI_Wtime()
       if (.not. this%crs_on_device) then
          tt(3) = tt(2); tt(4) = tt(5)
       end if
    end if
    res = this%cheby_dev%solve(this%ax, this%zf, r, n, this%coef, this%bcp, &
         this%gs_h, niter = this%nsmooth)
    call device_copy(z_d, this%zf%x_d, n)
    if (this%timers) then
       call device_sync(); tt(6) = MPI_Wtime()
       do i = 1, 6
          this%tcyc(i) = this%tcyc(i) + tt(i) - tt(i-1)
       end do
       this%ncyc = this%ncyc + 1
    end if
  end subroutine rs_colagg_solve_device

  !> x = k cycles of the two-level iteration on A x = r from zero.
  subroutine rs_colagg_cycles(this, x, r, n, k)
    class(rs_colagg_t), intent(inout) :: this
    integer, intent(in) :: n, k
    real(kind=rp), dimension(n), intent(inout) :: x
    real(kind=rp), dimension(n), intent(inout) :: r
    integer :: i, it
    type(c_ptr) :: x_d, r_d

    if (NEKO_BCKND_DEVICE .eq. 1) then
       x_d = device_get_ptr(x)
       r_d = device_get_ptr(r)
       do it = 1, k
          if (it .eq. 1) then
             call this%solve(x, r, n)
          else
             call this%ax%compute(this%res, x, this%coef, this%msh, this%Xh)
             call this%gs_h%op(this%res, n, GS_OP_ADD)
             call this%bcp%apply(this%res, n)
             call device_sub3(this%res_d, r_d, this%res_d, n)
             call this%solve(this%zbuf, this%res, n)
             call device_add2(x_d, this%zbuf_d, n)
          end if
       end do
       return
    end if
    do it = 1, k
       if (it .eq. 1) then
          call this%solve(x, r, n)
       else
          call this%ax%compute(this%res, x, this%coef, this%msh, this%Xh)
          call this%gs_h%op(this%res, n, GS_OP_ADD)
          call this%bcp%apply(this%res, n)
          do i = 1, n
             this%res(i) = r(i) - this%res(i)
          end do
          call this%solve(this%zbuf, this%res, n)
          do i = 1, n
             x(i) = x(i) + this%zbuf(i)
          end do
       end if
    end do
  end subroutine rs_colagg_cycles

  subroutine rs_colagg_update(this)
    class(rs_colagg_t), intent(inout) :: this
  end subroutine rs_colagg_update

  subroutine rs_colagg_free(this)
    class(rs_colagg_t), intent(inout) :: this
    if (pe_rank .eq. 0 .and. this%nagg .gt. 0) then
       write(*, '(A,I0,A,I0,A,I0,A)') 'RS_COLAGG coarse solves: ', this%crs_iters, &
            ' inner CG iterations in ', this%crs_calls, ' calls, ', this%crs_fails, &
            ' not converged'
       if (this%ncyc .gt. 0) then
          write(*, '(A,I0,A,6F9.4)') 'RS_COLAGG device cycle (rank 0, ', this%ncyc, &
               ' cycles) presmooth/resid/restrict/coarse/prolong/postsmooth ', this%tcyc
       end if
    end if
    this%tcyc = 0.0_dp; this%ncyc = 0
    if (allocated(this%iau)) then
       block
         integer(c_int) :: ph, err, idum(1)
         real(c_double) :: ddum(1)
         ph = -1; err = 0
         call pds_fn(this%pds_pt, 1, 1, 2, ph, this%nagg, ddum, idum, idum, idum, 1, &
              this%pds_iparm, 0, ddum, ddum, err)
       end block
       deallocate(this%iau, this%jau, this%avu)
    end if
    call this%spc%free()
    this%pds_pt = 0; this%direct = .false.
    if (c_associated(this%onesc_d)) then
       call device_deassociate(this%onesc); call device_free(this%onesc_d)
       this%onesc_d = c_null_ptr
    end if
    if (c_associated(this%ac_d) .and. allocated(this%ac)) call unmap_dense(this%ac, this%ac_d, size(this%ac))
    this%dense_dev = .false.
    if (allocated(this%onesc)) deallocate(this%onesc)
    if (allocated(this%aggid)) deallocate(this%aggid)
    if (allocated(this%dead)) deallocate(this%dead)
    if (allocated(this%ia)) deallocate(this%ia)
    if (allocated(this%ja)) deallocate(this%ja)
    if (this%crs_on_device) then
       call device_deassociate(this%avr); call device_free(this%av_d)
       call device_deassociate(this%dinv); call device_free(this%dinv_d)
       this%av_d = c_null_ptr; this%dinv_d = c_null_ptr
    end if
    if (allocated(this%av)) deallocate(this%av)
    if (allocated(this%avr)) deallocate(this%avr)
    if (allocated(this%dinv)) deallocate(this%dinv)
    if (allocated(this%ac)) deallocate(this%ac)
    if (allocated(this%ipiv)) deallocate(this%ipiv)
    if (c_associated(this%w_d)) then
       call device_deassociate(this%w); call device_free(this%w_d)
       call device_deassociate(this%res); call device_free(this%res_d)
       call device_deassociate(this%zbuf); call device_free(this%zbuf_d)
       this%w_d = c_null_ptr; this%res_d = c_null_ptr; this%zbuf_d = c_null_ptr
    end if
    if (allocated(this%w)) deallocate(this%w)
    if (allocated(this%res)) deallocate(this%res)
    if (this%crs_on_device) then
       call device_deassociate(this%agg0); call device_free(this%agg0_d)
       call device_deassociate(this%ia0); call device_free(this%ia0_d)
       call device_deassociate(this%ja0); call device_free(this%ja0_d)
       call device_deassociate(this%live); call device_free(this%live_d)
       call device_deassociate(this%rcd); call device_free(this%rcd_d)
       call device_deassociate(this%xcd); call device_free(this%xcd_d)
       call device_deassociate(this%pcd); call device_free(this%pcd_d)
       call device_deassociate(this%qcd); call device_free(this%qcd_d)
       call device_deassociate(this%zcd); call device_free(this%zcd_d)
       deallocate(this%agg0, this%ia0, this%ja0, this%live, this%rcd, this%xcd, this%pcd, &
            this%qcd, this%zcd)
       this%crs_on_device = .false.
    end if
    if (allocated(this%zbuf)) deallocate(this%zbuf)
    if (allocated(this%rc)) deallocate(this%rc)
    if (allocated(this%xc)) deallocate(this%xc)
    if (allocated(this%pc)) deallocate(this%pc)
    if (allocated(this%qc)) deallocate(this%qc)
    if (allocated(this%zc)) deallocate(this%zc)
    if (allocated(this%lv)) deallocate(this%lv)
    if (allocated(this%rcnt)) deallocate(this%rcnt, this%rdsp)
    if (allocated(this%acc)) deallocate(this%acc)
    if (allocated(this%ipivc)) deallocate(this%ipivc)
    this%nlv = 0; this%ncrs = 0
    if (this%n .gt. 0) call this%zf%free()
    nullify(this%ax, this%coef, this%gs_h, this%bcp, this%cheby, this%msh, this%Xh)
    this%n = 0; this%nagg = 0; this%nnz = 0
  end subroutine rs_colagg_free

  !> Gather-scatter minimum of the aggregation keys (global node ids + 1,
  !! integers held in double), exact in any working precision: in single
  !! precision the ids are split into (id / 2^23, mod(id, 2^23)), both exact
  !! in a float, and minimised lexicographically in two passes.
  subroutine gs_min_key(gs_h, key, n)
    type(gs_t), intent(inout) :: gs_h
    real(kind=dp), intent(inout) :: key(:)
    integer, intent(in) :: n
    real(kind=dp), parameter :: b = 8388608.0_dp   ! 2^23
    real(kind=dp), allocatable :: hi(:), lo(:), hmin(:)
    integer :: i
    if (rp .eq. dp) then
       call hgs(gs_h, key, n, GS_OP_MIN)
       return
    end if
    allocate(hi(n), lo(n), hmin(n))
    do i = 1, n
       hi(i) = aint(key(i) / b); lo(i) = key(i) - hi(i) * b
    end do
    hmin = hi
    call hgs(gs_h, hmin, n, GS_OP_MIN)
    do i = 1, n
       if (hi(i) .ne. hmin(i)) lo(i) = b      ! larger than any low part
    end do
    call hgs(gs_h, lo, n, GS_OP_MIN)
    do i = 1, n
       key(i) = hmin(i) * b + lo(i)
    end do
    deallocate(hi, lo, hmin)
  end subroutine gs_min_key

end module rs_colagg
