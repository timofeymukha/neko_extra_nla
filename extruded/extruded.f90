!> extruded: a direct solver for Neko's pressure (Helmholtz) equation on
!! extruded meshes, injected through Neko's type registry (no change to Neko).
!!
!! Build:  makeneko extruded.f90 ext_spchol.f90 ...   and use
!!         "preconditioner": {"type": "extruded"} (inside CG / GMRES), or the
!!         solver type "extruded" (the direct solve plus "refine" steps).
!!
!! The mesh is a 2D quad mesh (curved elements allowed) extruded along z in
!! layers of any thickness, periodic in z or not. Every element must be one
!! layer, with x, y independent of its extrusion coordinate; the element
!! orientation is arbitrary (the extrusion axis is found per element). Then
!! H = h1 (B_z (x) A_xy + K_z (x) B_xy) + h2 B_z (x) B_xy exactly, and with the
!! B_z-orthonormal eigenbasis S of the 1D problem (K_z S = B_z S Lambda) the
!! solve splits into one 2D problem per z mode,
!!    (h1 A_xy + (h1 lambda_k + h2) B_xy) u_k = (S^T f)_k,   u = S u_k.
!! Each distinct mode is factored once (sparse supernodal Cholesky, own code,
!! ext_spchol) and stored per supernode as G_s = [L11^{-1}; L21 L11^{-1}], so
!! that both triangular solves are matrix-vector products.
!!
!! Layout: the free (x, y) nodes are split into P ranges; an exchange brings
!! the element layout to z-pencils (all z of a range of 2D nodes), the z
!! transform is a GEMM there, a second exchange brings whole mode planes to
!! the rank that owns the mode's factor (modes with equal lambda, the cos/sin
!! pairs of a periodic z, share one factor); and back.
!!
!! Everything is derived from Neko's data: the GLL coordinates (2D numbering
!! from the element corners, merged within a tolerance), the z layers, the
!! periodic facets, the Dirichlet masks of the bc list. Requirements checked
!! at setup: one extrusion axis per element, the Dirichlet set independent of
!! z, uniform h1 and h2; a singular problem (no Dirichlet node, h2 = 0) is
!! refused for now.
module extruded
  use num_types, only : rp, dp, i8
  use precon, only : pc_t, register_precon, precon_allocate
  use krylov, only : ksp_t, ksp_monitor_t, register_krylov, krylov_allocate
  use ax_product, only : ax_t
  use field, only : field_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use scalar_bc_projector, only : scalar_bc_projector_t
  use vector_bc_projector, only : vector_bc_projector_t
  use device_math, only : device_glsc3, device_sub2, device_add2, device_copy
  use math, only : glsc3, sub2, add2
  use coefs, only : coef_t
  use bc_list, only : bc_list_t
  use bc, only : bc_t, BC_DIRICHLET, BC_NEUMANN
  use json_module, only : json_file
  use json_utils, only : json_get_or_default
  use comm, only : NEKO_COMM, pe_rank, pe_size
  use mpi_f08, only : MPI_Allreduce, MPI_Allgather, MPI_Allgatherv, MPI_Alltoall, MPI_Alltoallv, &
       MPI_IN_PLACE, MPI_INTEGER, MPI_INTEGER8, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_MAX, MPI_MIN, &
       MPI_LOR, MPI_LOGICAL, MPI_Wtime
  use utils, only : neko_error
  use logger, only : neko_log, LOG_SIZE
  use neko_config, only : NEKO_BCKND_DEVICE, NEKO_DEVICE_MPI
  use device, only : device_get_ptr, device_memcpy, device_map, device_unmap, device_sync, &
       HOST_TO_DEVICE, DEVICE_TO_HOST
  use ext_spchol, only : spchol_t
  use, intrinsic :: iso_c_binding
  implicit none
  private

  public :: extruded_register_types

  !> One factor (a group of modes with equal lambda): G blocks in the
  !! supernodal layout of the shared symbolic factorization, the Jacobi
  !! scaling, and the (one or two) local mode indices it serves.
  type :: slot_t
     real(kind=dp), allocatable :: g(:)
     real(kind=dp), allocatable :: d(:)
     integer :: nm = 0
     integer :: m(2) = 0       !< local mode index (1-based, in this rank's mode list)
     real(kind=dp) :: lam = 0.0_dp
  end type slot_t

  type, public, extends(pc_t) :: ext_pc_t
     type(coef_t), pointer :: coef => null()
     type(bc_list_t), pointer :: bclst => null()
     logical :: ready = .false.
     logical :: verbose = .false.
     integer :: lx = 0, n = 0, P = 1, me = 0
     !> the 2D problem: free nodes, CSR of the full matrices (1-based)
     integer :: nfree = 0, nz = 0
     integer, allocatable :: ia(:), ja(:)
     real(kind=dp), allocatable :: av(:), bv(:)
     !> z: S (nz, nz) B_z-orthonormal eigenvectors, lam (nz)
     real(kind=dp), allocatable :: S(:,:), lam(:)
     !> pencils: free-node ranges f0(r) .. f0(r+1)-1 (1-based), this rank's nr
     integer, allocatable :: f0(:)
     integer :: nr = 0
     !> element layout <-> pencils
     integer :: nheld = 0, nrecv = 0
     integer, allocatable :: gsend(:)     !< per held key: one local point
     integer, allocatable :: sidx(:)      !< per local point: held key (0: Dirichlet)
     integer, allocatable :: rpos(:)      !< per received key: position in the pencil (nr, nz)
     integer, allocatable :: hs_cnt(:), hs_dsp(:), hr_cnt(:), hr_dsp(:)
     !> pencils <-> mode planes: modes of every rank, S's columns in send order
     integer, allocatable :: nmr(:)       !< modes per rank
     integer, allocatable :: mglob(:)     !< this rank's modes (global index, 1-based)
     real(kind=dp), allocatable :: Sp(:,:)   !< S(:, send order)
     integer, allocatable :: ms_cnt(:), ms_dsp(:), mr_cnt(:), mr_dsp(:)
     !> factors
     type(spchol_t) :: chol
     type(slot_t), allocatable :: slot(:)
     integer, allocatable :: lev(:), lorder(:)
     !> work
     real(kind=dp), allocatable :: sb(:), rb(:), X(:,:), Y(:,:), mp(:), wv(:,:), wy(:,:)
     real(kind=rp), allocatable :: hr(:), hz(:)
     real(kind=dp) :: h1 = 1.0_dp, h2 = 0.0_dp
     !> device path (HIP / CUDA backends): the kernels in ext_dev.hip
     logical :: dev = .false.
     type(c_ptr) :: ctx = C_NULL_PTR
     real(kind=dp), allocatable :: hstage(:,:)   !< host staging without device MPI
     logical :: stage = .false.     !< exchanges through the host (no device MPI, or EXTRUDED_HOST_EXCHANGE set)
     integer :: nsolve = 0
     real(kind=dp) :: t_setup = 0.0_dp
     real(kind=dp) :: t_solve = 0.0_dp, t_phase(5) = 0.0_dp   !< accumulated (logged every 50 solves)
     real(kind=dp) :: t_max = 0.0_dp
     integer :: ntimed = 0
   contains
     procedure, pass(this) :: init => ext_init
     procedure, pass(this) :: solve => ext_solve
     procedure, pass(this) :: update => ext_update
     procedure, pass(this) :: free => ext_free
  end type ext_pc_t

  !> The direct solve as a Krylov solver type "extruded": x = M f, then
  !! "refine" steps x += M (f - A x) with Neko's operator.
  type, public, extends(ksp_t) :: ext_ksp_t
     real(kind=rp), allocatable :: w(:), w2(:)
     type(c_ptr) :: w_d = C_NULL_PTR, w2_d = C_NULL_PTR
     integer :: n = 0, refine = 1
   contains
     procedure, pass(this) :: init => ext_ksp_init
     procedure, pass(this) :: solve => ext_ksp_solve
     procedure, pass(this) :: solve_coupled => ext_ksp_solve_coupled
     procedure, pass(this) :: free => ext_ksp_free
  end type ext_ksp_t

  interface
     function ext_dev_setup(me, P, n, nheld, nrecv, nr, nz, nfree, nslot, nm, gsend, sidx, rpos, &
          hs_cnt, hs_dsp, hr_cnt, hr_dsp, ms_cnt, ms_dsp, mr_cnt, mr_dsp, f0, Sp, vmode, d, perm, &
          nsuper, sfc, srp, srow, nsrow, slp) bind(c, name = 'ext_dev_setup') result(ctx)
       import c_ptr, c_int, c_double, c_long
       integer(c_int), value :: me, P, n, nheld, nrecv, nr, nz, nfree, nslot, nm, nsuper
       integer(c_long), value :: nsrow
       integer(c_int) :: gsend(*), sidx(*), rpos(*), hs_cnt(*), hs_dsp(*), hr_cnt(*), hr_dsp(*), &
            ms_cnt(*), ms_dsp(*), mr_cnt(*), mr_dsp(*), f0(*), vmode(*), perm(*), sfc(*), srp(*), srow(*)
       real(c_double) :: Sp(*), d(*)
       integer(c_long) :: slp(*)
       type(c_ptr) :: ctx
     end function ext_dev_setup
     subroutine ext_dev_factor(ctx, k, g) bind(c, name = 'ext_dev_factor')
       import c_ptr, c_int, c_double
       type(c_ptr), value :: ctx
       integer(c_int), value :: k
       real(c_double) :: g(*)
     end subroutine ext_dev_factor
     function ext_dev_ptr(ctx, which) bind(c, name = 'ext_dev_ptr') result(p)
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: which
       type(c_ptr) :: p
     end function ext_dev_ptr
     subroutine ext_dev_gather(ctx, r, rpb) bind(c, name = 'ext_dev_gather')
       import c_ptr, c_int
       type(c_ptr), value :: ctx, r
       integer(c_int), value :: rpb
     end subroutine ext_dev_gather
     subroutine ext_dev_scatter(ctx, z, rpb) bind(c, name = 'ext_dev_scatter')
       import c_ptr, c_int
       type(c_ptr), value :: ctx, z
       integer(c_int), value :: rpb
     end subroutine ext_dev_scatter
     subroutine ext_dev_forward(ctx) bind(c, name = 'ext_dev_forward')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine ext_dev_forward
     subroutine ext_dev_solve(ctx) bind(c, name = 'ext_dev_solve')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine ext_dev_solve
     subroutine ext_dev_backward(ctx) bind(c, name = 'ext_dev_backward')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine ext_dev_backward
  end interface

contains

  subroutine extruded_register_types()
    procedure(precon_allocate), pointer :: alloc
    procedure(krylov_allocate), pointer :: alloc_ksp
    alloc => ext_allocate
    call register_precon("extruded", alloc)
    alloc_ksp => ext_ksp_allocate
    call register_krylov("extruded", alloc_ksp)
  end subroutine extruded_register_types

  subroutine ext_allocate(obj)
    class(pc_t), allocatable, intent(inout) :: obj
    allocate(ext_pc_t :: obj)
  end subroutine ext_allocate

  subroutine ext_ksp_allocate(obj)
    class(ksp_t), allocatable, intent(inout) :: obj
    allocate(ext_ksp_t :: obj)
  end subroutine ext_ksp_allocate

  subroutine ext_init(this, coef, bclst, json)
    class(ext_pc_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json
    call this%free()
    this%coef => coef
    this%bclst => bclst
    call json_get_or_default(json, 'verbose', this%verbose, .false.)
    this%ready = .false.
  end subroutine ext_init

  subroutine ext_update(this)
    class(ext_pc_t), intent(inout) :: this
    if (.not. associated(this%coef)) call neko_error('extruded: preconditioner never initialized ' // &
         '(Neko develop >= 77243ce needed)')
    if (.not. this%ready) call ext_setup(this)
  end subroutine ext_update

  ! ---------------------------------------------------------------------------
  ! setup
  ! ---------------------------------------------------------------------------

  subroutine ext_setup(this)
    class(ext_pc_t), intent(inout), target :: this
    type(coef_t), pointer :: coef
    character(len=320) :: log_buf
    integer :: lx, N, nelv, gnel, e, i, j, k, a, b, c, p, q, ierr, r, s, m, nv, ned, nq, nn, nzl, ez
    integer :: off, cnt, ii, jj, t, l(4), ss, dir, iu(2), iv(2), pt(2, 4), ip, jp, nlev, g, ng
    real(kind=dp), allocatable :: xc(:,:,:), xa(:,:,:), zr(:,:), allz(:), zlev(:)
    real(kind=dp), allocatable :: xyg(:,:,:,:), D(:,:), w(:), Kz(:,:), wz(:), vtmp(:,:)
    real(kind=dp) :: span, tol, dxy, dz, best, zmin, zmax, t0, h, zz, tolz
    integer, allocatable :: eax(:), cnts(:), dsps(:), vid(:), e2q(:), qrep(:), qc(:,:), eid(:,:)
    integer, allocatable :: node(:), zix(:), fid(:), keyo(:), cnts8(:)
    integer(kind=i8), allocatable :: key(:), ukey(:), rkey(:)
    integer, allocatable :: sperm(:), owner(:), gr(:,:), qown(:)
    logical, allocatable :: dflag(:), d2(:), dnode(:)
    logical :: periodic, ok
    integer :: nbad
    real(kind=dp), allocatable :: xf(:), yf(:), zf(:)

    t0 = MPI_Wtime()
    coef => this%coef
    lx = coef%Xh%lx; N = lx - 1
    nelv = coef%msh%nelv
    gnel = coef%msh%glb_nelv
    this%lx = lx
    this%n = lx**3 * nelv
    this%P = pe_size; this%me = pe_rank
    allocate(xf(this%n), yf(this%n), zf(this%n))
    xf = real(reshape(coef%dof%x%x, [this%n]), dp)
    yf = real(reshape(coef%dof%y%x, [this%n]), dp)
    zf = real(reshape(coef%dof%z%x, [this%n]), dp)
    allocate(D(lx, lx), w(lx))
    D = real(coef%Xh%dx, dp)
    w = real(coef%Xh%wx, dp)

    ! h1, h2 (uniform: checked)
    call ext_coefficients(this)

    ! --- 1. the extrusion axis of every element and its 2D face ------------
    allocate(eax(nelv), xc(2, 4, nelv), zr(2, nelv))
    span = 0.0_dp
    if (nelv .gt. 0) span = max(maxval(xf) - minval(xf), maxval(yf) - minval(yf))
    call MPI_Allreduce(MPI_IN_PLACE, span, 1, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    tol = max(1.0e-7_dp * span, 4.0_dp * real(epsilon(1.0_rp), dp) * span)
    nbad = 0
    do e = 1, nelv
       eax(e) = 0
       best = huge(1.0_dp)
       do a = 1, 3
          ! variation of (x, y) along local axis a, and of z
          dxy = 0.0_dp; dz = 0.0_dp
          do k = 1, lx
             do j = 1, lx
                do i = 1, lx - 1
                   call along(a, i, j, k, p, q)
                   dxy = max(dxy, abs(xof(p) - xof(q)) + abs(yof(p) - yof(q)))
                   dz = max(dz, abs(zof(p) - zof(q)))
                end do
             end do
          end do
          if (dxy .lt. tol .and. dz .gt. tol) then
             if (eax(e) .ne. 0) nbad = nbad + 1
             eax(e) = a
          end if
       end do
       if (eax(e) .eq. 0) then
          nbad = nbad + 1
          cycle
       end if
       ! corners of the face at extrusion index 1, in the face's (u, v) order
       pt(:, 1) = [1, 1]; pt(:, 2) = [lx, 1]; pt(:, 3) = [lx, lx]; pt(:, 4) = [1, lx]
       do c = 1, 4
          p = face_point(eax(e), pt(1, c), pt(2, c), 1, e)
          xc(1, c, e) = xof(p); xc(2, c, e) = yof(p)
       end do
       zr(1, e) = zof(face_point(eax(e), 1, 1, 1, e))
       zr(2, e) = zof(face_point(eax(e), 1, 1, lx, e))
    end do
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    if (nbad .gt. 0) then
       write(log_buf, '(A,I0,A)') 'extruded: ', nbad, ' elements without a unique extrusion axis along z'
       call neko_error(trim(log_buf))
    end if

    ! --- 2. global corners, vertices, 2D elements --------------------------
    allocate(cnts(0:pe_size - 1), dsps(0:pe_size - 1))
    call MPI_Allgather(nelv, 1, MPI_INTEGER, cnts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    dsps(0) = 0
    do r = 1, pe_size - 1
       dsps(r) = dsps(r - 1) + cnts(r - 1)
    end do
    allocate(xa(2, 4, gnel), allz(2 * gnel))
    call MPI_Allgatherv(xc, 8 * nelv, MPI_DOUBLE_PRECISION, xa, 8 * cnts, 8 * dsps, &
         MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    call MPI_Allgatherv(zr, 2 * nelv, MPI_DOUBLE_PRECISION, allz, 2 * cnts, 2 * dsps, &
         MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    ! vertices: merge the 4 gnel corner points within tol
    call merge_points(reshape(xa, [2, 4 * gnel]), 4 * gnel, tol, vid, nv)
    ! 2D elements: the sets of 4 vertex ids
    allocate(key(gnel), sperm(gnel), e2q(gnel))
    do e = 1, gnel
       l = vid(4 * (e - 1) + 1:4 * e)
       call isort4(l)
       ! the 3 smallest vertex ids identify the quad (packed: nv < 2^21)
       key(e) = int(l(1), i8) * 4398046511104_i8 + int(l(2), i8) * 2097152_i8 + int(l(3), i8)
    end do
    if (nv .ge. 2097152) call neko_error('extruded: more than 2^21 vertices in the 2D mesh')
    do e = 1, gnel
       sperm(e) = e
    end do
    call sort_key_index(key, sperm, gnel)
    nq = 0
    allocate(qrep(gnel))
    do i = 1, gnel
       if (i .eq. 1) then
          nq = 1
       else if (key(sperm(i)) .ne. key(sperm(i - 1))) then
          nq = nq + 1
       end if
       e2q(sperm(i)) = nq
       if (i .eq. 1) then
          qrep(nq) = sperm(i)
       else if (key(sperm(i)) .ne. key(sperm(i - 1))) then
          qrep(nq) = sperm(i)
       end if
       qrep(nq) = min(qrep(nq), sperm(i))
    end do
    ! corner ids of each 2D element in the cyclic order of its representative
    allocate(qc(4, nq))
    do q = 1, nq
       qc(:, q) = vid(4 * (qrep(q) - 1) + 1:4 * qrep(q))
    end do
    ! edges
    call quad_edges(qc, nq, eid, ned)
    nn = nv + ned * (N - 1) + nq * (N - 1)**2
    ! z layers
    allocate(zlev(2 * gnel))
    zlev = allz
    ! z tolerance: relative to the span, and above the rounding of Neko's real
    ! (single precision builds store the coordinates in fp32)
    tolz = max(1.0e-7_dp * (maxval(allz) - minval(allz)), &
         64.0_dp * real(epsilon(1.0_rp), dp) * max(abs(maxval(allz)), abs(minval(allz))))
    call sort_unique_tol(zlev, 2 * gnel, tolz, nzl)
    ez = nzl - 1
    zmin = zlev(1); zmax = zlev(nzl)
    periodic = z_periodic(coef, zmin, zmax, tolz)
    this%nz = ez * N
    if (.not. periodic) this%nz = this%nz + 1
    if (this%verbose .or. .true.) then
       write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A,L1,A,I0)') 'extruded: 2D mesh ', nq, ' quads, ', nv, &
            ' vertices, ', nn, ' nodes at N = ', N, '; z periodic ', periodic, ', layers ', ez
       call neko_log%message(log_buf)
    end if

    ! --- 3. per local point: 2D node and z index ---------------------------
    allocate(node(this%n), zix(this%n))
    do e = 1, nelv
       l = vid(4 * (coef%msh%offset_el + e - 1) + 1:4 * (coef%msh%offset_el + e))
       q = e2q(coef%msh%offset_el + e)
       ! layer of the element and the direction of z along the extrusion axis
       s = locate(zlev, nzl, min(zr(1, e), zr(2, e)))
       do k = 1, lx
          do jj = 1, lx
             do ii = 1, lx
                p = face_point(eax(e), ii, jj, k, e)
                node(p) = node2d(ii, jj, l, q)
                if (zr(1, e) .le. zr(2, e)) then
                   zix(p) = (s - 1) * N + (k - 1)
                else
                   zix(p) = (s - 1) * N + (lx - k)
                end if
                if (periodic .and. zix(p) .eq. ez * N) zix(p) = 0
             end do
          end do
       end do
    end do

    ! --- 4. Dirichlet nodes (must not depend on z), free numbering ---------
    allocate(dflag(this%n))
    dflag = .false.
    do i = 1, this%bclst%size()
       call mark_dirichlet(this%bclst%get(i), dflag, this%n)
    end do
    allocate(d2(nn))
    d2 = .false.
    do p = 1, this%n
       if (dflag(p)) d2(node(p)) = .true.
    end do
    call MPI_Allreduce(MPI_IN_PLACE, d2, nn, MPI_LOGICAL, MPI_LOR, NEKO_COMM, ierr)
    nbad = 0
    do p = 1, this%n
       if (d2(node(p)) .neqv. dflag(p)) nbad = nbad + 1
    end do
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    if (nbad .gt. 0) then
       write(log_buf, '(A,I0,A)') 'extruded: the Dirichlet set depends on z (', nbad, ' points)'
       call neko_error(trim(log_buf))
    end if
    allocate(fid(nn))
    this%nfree = 0
    do i = 1, nn
       if (d2(i)) then
          fid(i) = 0
       else
          this%nfree = this%nfree + 1
          fid(i) = this%nfree
       end if
    end do
    if (this%nfree .eq. nn .and. this%h2 .eq. 0.0_dp) &
         call neko_error('extruded: no Dirichlet node and h2 = 0 (singular): not supported yet')

    ! --- 5. the 2D operators -----------------------------------------------
    ! canonical GLL coordinates of every 2D element, from the rank owning its representative
    allocate(qown(nq))
    do q = 1, nq
       qown(q) = 0
       do r = pe_size - 1, 0, -1
          if (qrep(q) .gt. dsps(r)) then
             qown(q) = r
             exit
          end if
       end do
    end do
    allocate(xyg(2, lx, lx, nq))
    xyg = 0.0_dp
    do q = 1, nq
       if (qown(q) .ne. pe_rank) cycle
       e = qrep(q) - coef%msh%offset_el
       l = vid(4 * (qrep(q) - 1) + 1:4 * qrep(q))
       call canonical(l, ss, dir)
       do jj = 1, lx
          do ii = 1, lx
             call to_canonical(ii, jj, l, ss, dir, ip, jp)
             p = face_point(eax(e), ii, jj, 1, e)
             xyg(1, ip, jp, q) = xof(p); xyg(2, ip, jp, q) = yof(p)
          end do
       end do
    end do
    call MPI_Allreduce(MPI_IN_PLACE, xyg, 2 * lx * lx * nq, MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM, ierr)
    call assemble_2d(this, xyg, qc, eid, nq, nv, ned, nn, fid, D, w)
    deallocate(xyg)

    ! --- 6. z: 1D operators and their eigenbasis ----------------------------
    allocate(Kz(this%nz, this%nz), wz(this%nz))
    Kz = 0.0_dp; wz = 0.0_dp
    do s = 1, ez
       h = zlev(s + 1) - zlev(s)
       do b = 1, lx
          q = mod((s - 1) * N + b - 1, this%nz) + 1
          wz(q) = wz(q) + 0.5_dp * h * w(b)
          do a = 1, lx
             p = mod((s - 1) * N + a - 1, this%nz) + 1
             zz = 0.0_dp
             do c = 1, lx
                zz = zz + D(c, a) * w(c) * D(c, b)
             end do
             Kz(p, q) = Kz(p, q) + 2.0_dp / h * zz
          end do
       end do
    end do
    allocate(this%S(this%nz, this%nz), this%lam(this%nz))
    call gen_eig(Kz, wz, this%nz, this%S, this%lam)

    ! --- 7. pencils and the element-layout exchange -------------------------
    allocate(this%f0(0:pe_size))
    do r = 0, pe_size
       this%f0(r) = 1 + int((int(this%nfree, i8) * r) / pe_size)
    end do
    this%nr = this%f0(pe_rank + 1) - this%f0(pe_rank)
    call layout_exchange(this, node, zix, fid)

    ! --- 8. modes: groups of equal lambda, owners, factors ------------------
    call setup_modes(this)
    call setup_factors(this)

    this%t_setup = MPI_Wtime() - t0
    write(log_buf, '(A,I0,A,I0,A,I0,A,F8.1,A)') 'extruded: ', this%nfree, ' free 2D nodes, ', this%nz, &
         ' z modes, ', size(this%slot), ' factors on rank 0; setup ', this%t_setup, ' s'
    call neko_log%message(log_buf)
    allocate(this%hr(this%n), this%hz(this%n))
    this%ready = .true.

  contains

    real(kind=dp) function xof(p)
      integer, intent(in) :: p
      xof = xf(p)
    end function xof
    real(kind=dp) function yof(p)
      integer, intent(in) :: p
      yof = yf(p)
    end function yof
    real(kind=dp) function zof(p)
      integer, intent(in) :: p
      zof = zf(p)
    end function zof

    !> linear indices of two neighbours along local axis a (i -> i + 1 on a)
    subroutine along(ax, i1, j1, k1, p1, p2)
      integer, intent(in) :: ax, i1, j1, k1
      integer, intent(out) :: p1, p2
      p1 = face_point(ax, j1, k1, i1, e)
      p2 = face_point(ax, j1, k1, i1 + 1, e)
    end subroutine along

    !> the local point with face coordinates (u, v) and extrusion index t
    integer function face_point(ax, u, v, tt, el)
      integer, intent(in) :: ax, u, v, tt, el
      integer :: idx(3)
      select case (ax)
      case (1)
         idx = [tt, u, v]
      case (2)
         idx = [u, tt, v]
      case default
         idx = [u, v, tt]
      end select
      face_point = idx(1) + lx * ((idx(2) - 1) + lx * ((idx(3) - 1) + lx * (el - 1)))
    end function face_point

    !> global 2D node of face point (u, v) of an element with corners l (cyclic)
    integer function node2d(u, v, lc, qq)
      integer, intent(in) :: u, v, lc(4), qq
      integer :: ec, tpos, ssc, dd, uc, vc, va, vb
      if ((u .eq. 1 .or. u .eq. lx) .and. (v .eq. 1 .or. v .eq. lx)) then
         if (u .eq. 1 .and. v .eq. 1) node2d = lc(1)
         if (u .eq. lx .and. v .eq. 1) node2d = lc(2)
         if (u .eq. lx .and. v .eq. lx) node2d = lc(3)
         if (u .eq. 1 .and. v .eq. lx) node2d = lc(4)
         return
      end if
      if (v .eq. 1 .or. v .eq. lx .or. u .eq. 1 .or. u .eq. lx) then
         if (v .eq. 1) then
            va = lc(1); vb = lc(2); tpos = u - 1
         else if (u .eq. lx) then
            va = lc(2); vb = lc(3); tpos = v - 1
         else if (v .eq. lx) then
            va = lc(4); vb = lc(3); tpos = u - 1
         else
            va = lc(1); vb = lc(4); tpos = v - 1
         end if
         ec = edge_id(min(va, vb), max(va, vb))
         if (va .gt. vb) tpos = N - tpos
         node2d = nv + (ec - 1) * (N - 1) + tpos
         return
      end if
      call canonical(lc, ssc, dd)
      call to_canonical(u, v, lc, ssc, dd, uc, vc)
      node2d = nv + ned * (N - 1) + (qq - 1) * (N - 1)**2 + (uc - 1) + (vc - 2) * (N - 1)
    end function node2d

    integer function edge_id(va, vb)
      integer, intent(in) :: va, vb
      integer :: lo, hi, mid
      ! eid(1:2, :) sorted by (va, vb)
      lo = 1; hi = ned
      edge_id = 0
      do while (lo .le. hi)
         mid = (lo + hi) / 2
         if (eid(1, mid) .lt. va .or. (eid(1, mid) .eq. va .and. eid(2, mid) .lt. vb)) then
            lo = mid + 1
         else if (eid(1, mid) .eq. va .and. eid(2, mid) .eq. vb) then
            edge_id = mid
            return
         else
            hi = mid - 1
         end if
      end do
      call neko_error('extruded: edge not found')
    end function edge_id

    !> canonical frame of a quad with corners lc: start at the smallest id,
    !! towards its smaller neighbour
    subroutine canonical(lc, ssc, dd)
      integer, intent(in) :: lc(4)
      integer, intent(out) :: ssc, dd
      ssc = minloc(lc, 1)
      if (lc(mod(ssc, 4) + 1) .lt. lc(mod(ssc + 2, 4) + 1)) then
         dd = 1
      else
         dd = -1
      end if
    end subroutine canonical

    subroutine to_canonical(u, v, lc, ssc, dd, uc, vc)
      integer, intent(in) :: u, v, lc(4), ssc, dd
      integer, intent(out) :: uc, vc
      integer :: c0(2), c1(2), c2(2), du(2), dv(2), cc(2, 4)
      cc(:, 1) = [1, 1]; cc(:, 2) = [lx, 1]; cc(:, 3) = [lx, lx]; cc(:, 4) = [1, lx]
      c0 = cc(:, ssc)
      c1 = cc(:, modulo(ssc - 1 + dd, 4) + 1)
      c2 = cc(:, modulo(ssc - 1 - dd, 4) + 1)
      du = (c1 - c0) / N
      dv = (c2 - c0) / N
      uc = 1 + (u - c0(1)) * du(1) + (v - c0(2)) * du(2)
      vc = 1 + (u - c0(1)) * dv(1) + (v - c0(2)) * dv(2)
    end subroutine to_canonical

  end subroutine ext_setup

  !> h1 and h2 (uniform; a non-uniform field is refused)
  subroutine ext_coefficients(this)
    class(ext_pc_t), intent(inout) :: this
    real(kind=rp), allocatable :: h(:)
    real(kind=dp) :: v(2), vmin(2)
    integer :: ierr
    type(c_ptr) :: h_d
    allocate(h(this%n))
    v = -huge(1.0_dp); vmin = huge(1.0_dp)
    if (this%n .gt. 0) then
       if (NEKO_BCKND_DEVICE .eq. 1) then
          h_d = this%coef%h1_d
          call device_memcpy(h, h_d, this%n, DEVICE_TO_HOST, sync = .true.)
       else
          h = reshape(this%coef%h1, [this%n])
       end if
       v(1) = maxval(h); vmin(1) = minval(h)
       v(2) = 0.0_dp; vmin(2) = 0.0_dp
       if (this%coef%ifh2) then
          if (NEKO_BCKND_DEVICE .eq. 1) then
             h_d = this%coef%h2_d
             call device_memcpy(h, h_d, this%n, DEVICE_TO_HOST, sync = .true.)
          else
             h = reshape(this%coef%h2, [this%n])
          end if
          v(2) = maxval(h); vmin(2) = minval(h)
       end if
    end if
    call MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, vmin, 2, MPI_DOUBLE_PRECISION, MPI_MIN, NEKO_COMM, ierr)
    if (any(v - vmin .gt. 1.0e-6_dp * abs(v))) &
         call neko_error('extruded: h1 or h2 not uniform (only the pressure / constant coefficients)')
    this%h1 = v(1)
    this%h2 = v(2)
  end subroutine ext_coefficients

  subroutine mark_dirichlet(b, flag, n)
    class(bc_t), intent(in) :: b
    logical, intent(inout) :: flag(:)
    integer, intent(in) :: n
    integer :: i, m, cnt
    cnt = 0
    if (allocated(b%msk)) cnt = b%msk(0)
    if (cnt .eq. 0) return
    if (b%bc_type .eq. BC_DIRICHLET) then
       do i = 1, cnt
          m = b%msk(i)
          if (m .ge. 1 .and. m .le. n) flag(m) = .true.
       end do
    else if (b%bc_type .ne. BC_NEUMANN) then
       call neko_error('extruded: only Dirichlet and Neumann conditions are supported')
    end if
  end subroutine mark_dirichlet

  !> z periodic: a periodic facet of the mesh lies at zmin or zmax
  function z_periodic(coef, zmin, zmax, tz) result(per)
    type(coef_t), intent(in) :: coef
    real(kind=dp), intent(in) :: zmin, zmax, tz
    logical :: per
    integer :: m, f, e, lx, i, j, k, i0, i1, j0, j1, k0, k1, flag, ierr, other
    real(kind=dp) :: lo, hi, zv
    lx = coef%Xh%lx
    flag = 0
    other = 0
    do m = 1, coef%msh%periodic%size
       f = coef%msh%periodic%facet_el(m)%x(1)
       e = coef%msh%periodic%facet_el(m)%x(2)
       i0 = 1; i1 = lx; j0 = 1; j1 = lx; k0 = 1; k1 = lx
       select case (f)
       case (1); i1 = 1
       case (2); i0 = lx
       case (3); j1 = 1
       case (4); j0 = lx
       case (5); k1 = 1
       case default; k0 = lx
       end select
       lo = huge(1.0_dp); hi = -huge(1.0_dp)
       do k = k0, k1
          do j = j0, j1
             do i = i0, i1
                zv = real(coef%dof%z%x(i, j, k, e), dp)
                lo = min(lo, zv); hi = max(hi, zv)
             end do
          end do
       end do
       if (hi - lo .lt. tz .and. (abs(lo - zmin) .lt. tz .or. abs(lo - zmax) .lt. tz)) then
          flag = 1
       else
          other = 1
       end if
    end do
    call MPI_Allreduce(MPI_IN_PLACE, flag, 1, MPI_INTEGER, MPI_MAX, NEKO_COMM, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, other, 1, MPI_INTEGER, MPI_MAX, NEKO_COMM, ierr)
    if (other .eq. 1) call neko_error('extruded: periodic facets other than the z ends are not supported')
    per = flag .eq. 1
  end function z_periodic

  ! ---------------------------------------------------------------------------
  ! 2D assembly
  ! ---------------------------------------------------------------------------

  !> A_xy (stiffness) and B_xy (lumped mass) on the free nodes, CSR of the full
  !! matrix (1-based), from the canonical GLL coordinates of every quad.
  subroutine assemble_2d(this, xyg, qc, eid, nq, nv, ned, nn, fid, D, w)
    class(ext_pc_t), intent(inout) :: this
    integer, intent(in) :: nq, nv, ned, nn
    real(kind=dp), intent(in) :: xyg(:, :, :, :), D(:,:), w(:)
    integer, intent(in) :: qc(:,:), eid(:,:), fid(:)
    integer :: lx, N, q, i, j, a, b, p, k, nl, nf, row, col, pos, cnt, i2, j2
    integer, allocatable :: gl(:,:), nep(:), ne(:), mark(:), rowlen(:)
    real(kind=dp), allocatable :: Ke(:,:), xr(:,:), xs(:,:), yr(:,:), ys(:,:), g11(:,:), g12(:,:), g22(:,:)
    real(kind=dp), allocatable :: ur(:,:), us(:,:), fr(:,:), fs(:,:), bm(:)
    real(kind=dp) :: jac
    lx = this%lx; N = lx - 1; nl = lx * lx; nf = this%nfree
    ! local node lists (canonical frame) of every quad
    allocate(gl(nl, nq))
    do q = 1, nq
       call canonical_nodes(q, gl(:, q))
    end do
    ! pattern: node -> quads, then rows as unions
    allocate(nep(nn + 1), ne(nl * nq))
    nep = 0
    do q = 1, nq
       do a = 1, nl
          nep(gl(a, q) + 1) = nep(gl(a, q) + 1) + 1
       end do
    end do
    nep(1) = 1
    do i = 1, nn
       nep(i + 1) = nep(i + 1) + nep(i)
    end do
    allocate(rowlen(nn))
    rowlen = 0
    do q = 1, nq
       do a = 1, nl
          i = gl(a, q)
          ne(nep(i) + rowlen(i)) = q
          rowlen(i) = rowlen(i) + 1
       end do
    end do
    allocate(this%ia(nf + 1), mark(nf))
    mark = 0
    this%ia(1) = 1
    do i = 1, nn
       if (fid(i) .eq. 0) cycle
       cnt = 0
       do k = nep(i), nep(i + 1) - 1
          q = ne(k)
          do b = 1, nl
             col = fid(gl(b, q))
             if (col .eq. 0) cycle
             if (mark(col) .ne. fid(i)) then
                mark(col) = fid(i)
                cnt = cnt + 1
             end if
          end do
       end do
       this%ia(fid(i) + 1) = this%ia(fid(i)) + cnt
    end do
    allocate(this%ja(this%ia(nf + 1) - 1), this%av(this%ia(nf + 1) - 1), this%bv(nf))
    mark = 0
    do i = 1, nn
       if (fid(i) .eq. 0) cycle
       row = fid(i)
       pos = this%ia(row)
       do k = nep(i), nep(i + 1) - 1
          q = ne(k)
          do b = 1, nl
             col = fid(gl(b, q))
             if (col .eq. 0) cycle
             if (mark(col) .ne. row) then
                mark(col) = row
                this%ja(pos) = col
                pos = pos + 1
             end if
          end do
       end do
       call isort(this%ja(this%ia(row):this%ia(row + 1) - 1))
    end do
    this%av = 0.0_dp
    this%bv = 0.0_dp
    ! element matrices
    allocate(Ke(nl, nl), xr(lx, lx), xs(lx, lx), yr(lx, lx), ys(lx, lx), g11(lx, lx), g12(lx, lx), &
         g22(lx, lx), ur(lx, lx), us(lx, lx), fr(lx, lx), fs(lx, lx), bm(nl))
    do q = 1, nq
       ! derivatives: r along the first index, s along the second
       do j = 1, lx
          do i = 1, lx
             xr(i, j) = dot_product(D(i, :), xyg(1, :, j, q))
             yr(i, j) = dot_product(D(i, :), xyg(2, :, j, q))
             xs(i, j) = dot_product(D(j, :), xyg(1, i, :, q))
             ys(i, j) = dot_product(D(j, :), xyg(2, i, :, q))
             jac = abs(xr(i, j) * ys(i, j) - xs(i, j) * yr(i, j))
             g11(i, j) = w(i) * w(j) * (xs(i, j)**2 + ys(i, j)**2) / jac
             g12(i, j) = -w(i) * w(j) * (xr(i, j) * xs(i, j) + yr(i, j) * ys(i, j)) / jac
             g22(i, j) = w(i) * w(j) * (xr(i, j)**2 + yr(i, j)**2) / jac
             bm(i + lx * (j - 1)) = w(i) * w(j) * jac
          end do
       end do
       do b = 1, nl
          ! u = e_b: ur = D u, us = u D^T
          ur = 0.0_dp; us = 0.0_dp
          i = mod(b - 1, lx) + 1; j = (b - 1) / lx + 1
          ur(:, j) = D(:, i)
          us(i, :) = D(:, j)
          fr = g11 * ur + g12 * us
          fs = g12 * ur + g22 * us
          ! K(:, b) = Dr^T fr + Ds^T fs
          do j2 = 1, lx
             do i2 = 1, lx
                Ke(i2 + lx * (j2 - 1), b) = dot_product(D(:, i2), fr(:, j2)) + dot_product(D(:, j2), fs(i2, :))
             end do
          end do
       end do
       ! add into the CSR
       do a = 1, nl
          row = fid(gl(a, q))
          if (row .eq. 0) cycle
          this%bv(row) = this%bv(row) + bm(a)
          do b = 1, nl
             col = fid(gl(b, q))
             if (col .eq. 0) cycle
             pos = find_col(this%ja, this%ia(row), this%ia(row + 1) - 1, col)
             this%av(pos) = this%av(pos) + Ke(a, b)
          end do
       end do
    end do

  contains

    !> the 2D nodes of quad q at its canonical (i, j), i fastest
    subroutine canonical_nodes(qq, lst)
      integer, intent(in) :: qq
      integer, intent(out) :: lst(:)
      integer :: lc(4), c, ii, jj, va, vb, tpos, ec, ssc, dd
      ! canonical corner order
      call canon(qc(:, qq), ssc, dd)
      do c = 1, 4
         lc(c) = qc(modulo(ssc - 1 + dd * (c - 1), 4) + 1, qq)
      end do
      do jj = 1, lx
         do ii = 1, lx
            if ((ii .eq. 1 .or. ii .eq. lx) .and. (jj .eq. 1 .or. jj .eq. lx)) then
               if (ii .eq. 1 .and. jj .eq. 1) lst(1) = lc(1)
               if (ii .eq. lx .and. jj .eq. 1) lst(lx) = lc(2)
               if (ii .eq. lx .and. jj .eq. lx) lst(nl) = lc(3)
               if (ii .eq. 1 .and. jj .eq. lx) lst(1 + lx * N) = lc(4)
               cycle
            end if
            if (jj .eq. 1 .or. jj .eq. lx .or. ii .eq. 1 .or. ii .eq. lx) then
               if (jj .eq. 1) then
                  va = lc(1); vb = lc(2); tpos = ii - 1
               else if (ii .eq. lx) then
                  va = lc(2); vb = lc(3); tpos = jj - 1
               else if (jj .eq. lx) then
                  va = lc(4); vb = lc(3); tpos = ii - 1
               else
                  va = lc(1); vb = lc(4); tpos = jj - 1
               end if
               ec = find_edge(min(va, vb), max(va, vb))
               if (va .gt. vb) tpos = N - tpos
               lst(ii + lx * (jj - 1)) = nv + (ec - 1) * (N - 1) + tpos
               cycle
            end if
            lst(ii + lx * (jj - 1)) = nv + ned * (N - 1) + (qq - 1) * (N - 1)**2 + (ii - 1) + (jj - 2) * (N - 1)
         end do
      end do
    end subroutine canonical_nodes

    subroutine canon(lc, ssc, dd)
      integer, intent(in) :: lc(4)
      integer, intent(out) :: ssc, dd
      ssc = minloc(lc, 1)
      if (lc(mod(ssc, 4) + 1) .lt. lc(mod(ssc + 2, 4) + 1)) then
         dd = 1
      else
         dd = -1
      end if
    end subroutine canon

    integer function find_edge(va, vb)
      integer, intent(in) :: va, vb
      integer :: lo, hi, mid
      lo = 1; hi = ned
      find_edge = 0
      do while (lo .le. hi)
         mid = (lo + hi) / 2
         if (eid(1, mid) .lt. va .or. (eid(1, mid) .eq. va .and. eid(2, mid) .lt. vb)) then
            lo = mid + 1
         else if (eid(1, mid) .eq. va .and. eid(2, mid) .eq. vb) then
            find_edge = mid
            return
         else
            hi = mid - 1
         end if
      end do
      call neko_error('extruded: edge not found (assembly)')
    end function find_edge

  end subroutine assemble_2d

  integer function find_col(ja, lo0, hi0, col)
    integer, intent(in) :: ja(:), lo0, hi0, col
    integer :: lo, hi, mid
    lo = lo0; hi = hi0
    do while (lo .le. hi)
       mid = (lo + hi) / 2
       if (ja(mid) .lt. col) then
          lo = mid + 1
       else if (ja(mid) .gt. col) then
          hi = mid - 1
       else
          find_col = mid
          return
       end if
    end do
    find_col = -1
    call neko_error('extruded: column not in the pattern')
  end function find_col

  ! ---------------------------------------------------------------------------
  ! z eigenbasis
  ! ---------------------------------------------------------------------------

  !> K S = diag(wz) S Lambda, S^T diag(wz) S = I, lambda ascending, the
  !! smallest set to 0 when it is at round-off
  subroutine gen_eig(K, wz, nz, S, lam)
    integer, intent(in) :: nz
    real(kind=dp), intent(in) :: K(nz, nz), wz(nz)
    real(kind=dp), intent(out) :: S(nz, nz), lam(nz)
    real(kind=dp), allocatable :: A(:,:), work(:)
    integer :: i, j, info, lwork
    allocate(A(nz, nz))
    do j = 1, nz
       do i = 1, nz
          A(i, j) = K(i, j) / sqrt(wz(i) * wz(j))
       end do
    end do
    lwork = max(1, 3 * nz * nz)
    allocate(work(lwork))
    call dsyev('V', 'U', nz, A, nz, lam, work, lwork, info)
    if (info .ne. 0) call neko_error('extruded: dsyev failed')
    do j = 1, nz
       do i = 1, nz
          S(i, j) = A(i, j) / sqrt(wz(i))
       end do
    end do
    if (abs(lam(1)) .lt. 1.0e-10_dp * abs(lam(nz))) lam(1) = 0.0_dp
  end subroutine gen_eig

  ! ---------------------------------------------------------------------------
  ! layouts
  ! ---------------------------------------------------------------------------

  !> The element layout <-> z-pencil exchange: every rank sends each (free
  !! node, z) it holds once to the owner of the node's range.
  subroutine layout_exchange(this, node, zix, fid)
    class(ext_pc_t), intent(inout) :: this
    integer, intent(in) :: node(:), zix(:), fid(:)
    integer(kind=i8), allocatable :: key(:), ukey(:), rkey(:)
    integer, allocatable :: idx(:), owner(:), sc(:), sd(:), rc(:), rd(:), fill(:)
    integer :: p, n, nu, i, r, ierr, f, np
    integer(kind=i8) :: nf8
    n = this%n; np = pe_size
    nf8 = int(this%nfree, i8)
    allocate(key(n), idx(n))
    nu = 0
    do p = 1, n
       f = fid(node(p))
       if (f .eq. 0) then
          key(p) = huge(1_i8)
       else
          key(p) = int(zix(p), i8) * nf8 + (f - 1)
       end if
       idx(p) = p
    end do
    call sort_key_index(key, idx, n)
    ! unique held keys, gsend (one local point each), sidx (per point)
    allocate(this%sidx(n), ukey(n), this%gsend(n))
    this%sidx = 0
    do i = 1, n
       p = idx(i)
       if (key(p) .eq. huge(1_i8)) cycle
       if (nu .eq. 0) then
          nu = 1
          ukey(1) = key(p); this%gsend(1) = p
       else if (key(p) .ne. ukey(nu)) then
          nu = nu + 1
          ukey(nu) = key(p); this%gsend(nu) = p
       end if
       this%sidx(p) = nu
    end do
    this%nheld = nu
    ! owners by the free-node ranges; the keys are sorted by (z, node), not by
    ! owner: reorder the held keys by owner
    allocate(owner(nu), sc(0:np - 1), sd(0:np - 1), rc(0:np - 1), rd(0:np - 1), fill(0:np - 1))
    sc = 0
    do i = 1, nu
       f = int(mod(ukey(i), nf8)) + 1
       owner(i) = owner_of(f)
       sc(owner(i)) = sc(owner(i)) + 1
    end do
    sd(0) = 0
    do r = 1, np - 1
       sd(r) = sd(r - 1) + sc(r - 1)
    end do
    block
      integer, allocatable :: neworder(:), gs2(:), map(:)
      integer(kind=i8), allocatable :: uk2(:)
      allocate(neworder(nu), gs2(nu), uk2(nu), map(nu))
      fill = 0
      do i = 1, nu
         neworder(i) = sd(owner(i)) + fill(owner(i)) + 1
         fill(owner(i)) = fill(owner(i)) + 1
      end do
      do i = 1, nu
         uk2(neworder(i)) = ukey(i)
         gs2(neworder(i)) = this%gsend(i)
         map(i) = neworder(i)
      end do
      ukey(1:nu) = uk2
      this%gsend(1:nu) = gs2
      do p = 1, n
         if (this%sidx(p) .gt. 0) this%sidx(p) = map(this%sidx(p))
      end do
    end block
    call MPI_Alltoall(sc, 1, MPI_INTEGER, rc, 1, MPI_INTEGER, NEKO_COMM, ierr)
    rd(0) = 0
    do r = 1, np - 1
       rd(r) = rd(r - 1) + rc(r - 1)
    end do
    this%nrecv = rd(np - 1) + rc(np - 1)
    allocate(rkey(max(this%nrecv, 1)))
    call MPI_Alltoallv(ukey, sc, sd, MPI_INTEGER8, rkey, rc, rd, MPI_INTEGER8, NEKO_COMM, ierr)
    allocate(this%rpos(max(this%nrecv, 1)))
    do i = 1, this%nrecv
       f = int(mod(rkey(i), nf8)) + 1
       this%rpos(i) = (f - this%f0(this%me)) + 1 + this%nr * int(rkey(i) / nf8)
    end do
    call move_alloc(sc, this%hs_cnt); call move_alloc(sd, this%hs_dsp)
    call move_alloc(rc, this%hr_cnt); call move_alloc(rd, this%hr_dsp)
    allocate(this%sb(max(this%nheld, 1)), this%rb(max(this%nrecv, 1)))
  contains
    integer function owner_of(ff)
      integer, intent(in) :: ff
      integer :: lo, hi, mid
      lo = 0; hi = np - 1
      do while (lo .lt. hi)
         mid = (lo + hi + 1) / 2
         if (this%f0(mid) .le. ff) then
            lo = mid
         else
            hi = mid - 1
         end if
      end do
      owner_of = lo
    end function owner_of
  end subroutine layout_exchange

  !> Mode groups (equal lambda, at most 2), group g on rank mod(g - 1, P); the
  !! pencil -> mode-plane exchange counts; S's columns in send order.
  subroutine setup_modes(this)
    class(ext_pc_t), intent(inout) :: this
    integer :: nz, k, g, ng, r, P, i, ierr, c, mi
    integer, allocatable :: gofm(:), gsz(:), gm(:,:), mown(:), order(:)
    real(kind=dp) :: tolam
    nz = this%nz; P = pe_size
    allocate(gofm(nz), gsz(nz), gm(2, nz), mown(nz))
    tolam = 1.0e-9_dp * max(1.0_dp, abs(this%lam(nz)))
    ng = 0
    k = 1
    do while (k .le. nz)
       ng = ng + 1
       gsz(ng) = 1; gm(1, ng) = k
       if (k .lt. nz) then
          ! equal within round-off of Neko's real (fp32 coordinates split the pairs slightly)
          if (abs(this%lam(k + 1) - this%lam(k)) .le. max(1.0e-9_dp, 100.0_dp * real(epsilon(1.0_rp), dp)) * &
               max(1.0_dp, abs(this%lam(k)))) then
             gsz(ng) = 2; gm(2, ng) = k + 1
          end if
       end if
       k = k + gsz(ng)
    end do
    allocate(this%nmr(0:P - 1))
    this%nmr = 0
    do g = 1, ng
       r = mod(g - 1, P)
       do i = 1, gsz(g)
          mown(gm(i, g)) = r
       end do
       this%nmr(r) = this%nmr(r) + gsz(g)
    end do
    ! this rank's modes and slots
    allocate(this%mglob(max(this%nmr(this%me), 1)), this%slot((ng - this%me + P - 1) / P))
    mi = 0
    c = 0
    do g = this%me + 1, ng, P
       c = c + 1
       this%slot(c)%nm = gsz(g)
       this%slot(c)%lam = this%lam(gm(1, g))
       do i = 1, gsz(g)
          mi = mi + 1
          this%mglob(mi) = gm(i, g)
          this%slot(c)%m(i) = mi
       end do
    end do
    ! send order: modes of rank 0, 1, ... each in its local order
    allocate(this%Sp(nz, nz))
    c = 0
    do r = 0, P - 1
       do g = r + 1, ng, P
          do i = 1, gsz(g)
             c = c + 1
             this%Sp(:, c) = this%S(:, gm(i, g))
          end do
       end do
    end do
    ! counts: B forward sends to rank d its nmr(d) columns of my nr nodes
    allocate(this%ms_cnt(0:P - 1), this%ms_dsp(0:P - 1), this%mr_cnt(0:P - 1), this%mr_dsp(0:P - 1))
    c = 0
    do r = 0, P - 1
       this%ms_cnt(r) = this%nmr(r) * this%nr
       this%ms_dsp(r) = c
       c = c + this%ms_cnt(r)
    end do
    c = 0
    do r = 0, P - 1
       this%mr_cnt(r) = this%nmr(this%me) * (this%f0(r + 1) - this%f0(r))
       this%mr_dsp(r) = c
       c = c + this%mr_cnt(r)
    end do
    allocate(this%X(max(this%nr, 1), nz), this%Y(max(this%nr, 1), nz), this%mp(max(c, 1)))
  end subroutine setup_modes

  !> One symbolic factorization, then the numeric factor of every slot,
  !! stored as G blocks; levels of the supernodal tree (for the device).
  subroutine setup_factors(this)
    class(ext_pc_t), intent(inout) :: this
    integer :: c, nf, info, s, k, m, nb, i, j, ns, q
    real(kind=dp), allocatable :: a(:), dall(:,:)
    integer, allocatable :: vmode(:), f0(:)
    integer(kind=i8) :: o
    character(len=320) :: log_buf
    nf = this%nfree
    ns = size(this%slot)
    if (ns .eq. 0) call neko_error('extruded: more ranks than mode groups (not supported yet)')
    this%dev = NEKO_BCKND_DEVICE .eq. 1
    block
      integer :: ln
      call get_environment_variable('EXTRUDED_HOST_EXCHANGE', length = ln)
      this%stage = (.not. NEKO_DEVICE_MPI) .or. ln .gt. 0
    end block
    allocate(a(size(this%av)))
    do c = 1, ns
       allocate(this%slot(c)%d(nf))
       call shifted(this, this%slot(c)%lam, a, this%slot(c)%d)
    end do
    do c = 1, ns
       associate(sl => this%slot(c))
         call shifted(this, sl%lam, a, sl%d)
         if (c .eq. 1) then
            call this%chol%init(nf, this%ia, this%ja, a, info)
            write(log_buf, '(A,F7.1,A,ES10.3,A,F6.1,F6.1,F6.1,A)') 'extruded: factor stored nnz/node ', &
                 real(this%chol%nnz_l, dp) / nf, ', flops ', this%chol%flops, ', order/symbolic/numeric ', &
                 this%chol%t_order, this%chol%t_symbolic, this%chol%t_numeric, ' s (rank 0)'
            if (pe_rank .eq. 0) call neko_log%message(log_buf)
            if (info .ne. 0) call neko_error('extruded: factorization failed (not positive definite)')
            if (this%dev) then
               allocate(dall(nf, ns), vmode(2 * ns), f0(pe_size + 1))
               do q = 1, ns
                  dall(:, q) = this%slot(q)%d
                  vmode(2 * q - 1) = this%slot(q)%m(1) - 1
                  vmode(2 * q) = -1
                  if (this%slot(q)%nm .eq. 2) vmode(2 * q) = this%slot(q)%m(2) - 1
               end do
               f0 = this%f0(0:pe_size)
               this%ctx = ext_dev_setup(this%me, pe_size, this%n, this%nheld, this%nrecv, this%nr, this%nz, nf, &
                    ns, this%nmr(this%me), this%gsend, this%sidx, this%rpos, this%hs_cnt, this%hs_dsp, &
                    this%hr_cnt, this%hr_dsp, this%ms_cnt, this%ms_dsp, this%mr_cnt, this%mr_dsp, f0, &
                    this%Sp, vmode, dall, this%chol%perm, this%chol%nsuper, this%chol%sfc, this%chol%srp, &
                    this%chol%srow, int(size(this%chol%srow), c_long), this%chol%slp)
               deallocate(dall)
            end if
         else
            call this%chol%factor(this%ia, this%ja, a, info)
            if (info .ne. 0) call neko_error('extruded: factorization failed (not positive definite)')
         end if
         ! G = [L11^{-1}; L21 L11^{-1}] per supernode
         allocate(sl%g(size(this%chol%lx)))
         sl%g = this%chol%lx
         do s = 1, this%chol%nsuper
            k = this%chol%sfc(s + 1) - this%chol%sfc(s)
            m = this%chol%srp(s + 1) - this%chol%srp(s)
            o = this%chol%slp(s)
            call dtrtri('L', 'N', k, sl%g(o + 1), m, info)
            if (info .ne. 0) call neko_error('extruded: dtrtri failed')
            do j = 2, k
               do i = 1, j - 1
                  sl%g(o + i + int(j - 1, i8) * m) = 0.0_dp
               end do
            end do
            nb = m - k
            if (nb .gt. 0) call dtrmm('R', 'L', 'N', 'N', nb, k, 1.0_dp, sl%g(o + 1), m, sl%g(o + k + 1), m)
         end do
         if (this%dev) then
            call ext_dev_factor(this%ctx, c - 1, sl%g)
            deallocate(sl%g)
         end if
       end associate
    end do
    allocate(this%wv(nf, 2), this%wy(nf, 2))
  end subroutine setup_factors

  !> a = D (h1 (A + lam B) + h2 B) D with D the inverse square root of the diagonal
  subroutine shifted(this, lam, a, d)
    class(ext_pc_t), intent(in) :: this
    real(kind=dp), intent(in) :: lam
    real(kind=dp), intent(out) :: a(:), d(:)
    integer :: i, k
    do i = 1, this%nfree
       do k = this%ia(i), this%ia(i + 1) - 1
          a(k) = this%h1 * this%av(k)
          if (this%ja(k) .eq. i) then
             a(k) = a(k) + (this%h1 * lam + this%h2) * this%bv(i)
             d(i) = 1.0_dp / sqrt(a(k))
          end if
       end do
    end do
    do i = 1, this%nfree
       do k = this%ia(i), this%ia(i + 1) - 1
          a(k) = a(k) * d(i) * d(this%ja(k))
       end do
    end do
  end subroutine shifted

  ! ---------------------------------------------------------------------------
  ! solve (host path)
  ! ---------------------------------------------------------------------------

  subroutine ext_solve(this, z, r, n)
    integer, intent(in) :: n
    class(ext_pc_t), intent(inout) :: this
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    type(c_ptr) :: r_d, z_d
    integer(c_int) :: rpb
    real(kind=dp) :: t0, t1, t2
    character(len=320) :: log_buf
    rpb = int(storage_size(1.0_rp) / 8, c_int)
    if (.not. this%ready) call neko_error('extruded: solve before update')
    if (n .ne. this%n) call neko_error('extruded: solve with another size')
    this%nsolve = this%nsolve + 1
    t0 = MPI_Wtime()
    if (this%dev) then
       call device_sync()
       r_d = device_get_ptr(r)
       z_d = device_get_ptr(z)
       call ext_dev_gather(this%ctx, r_d, rpb)
       t1 = MPI_Wtime(); this%t_phase(1) = this%t_phase(1) + t1 - t0
       call dev_exchange(this, 1, 2, this%hs_cnt, this%hs_dsp, this%hr_cnt, this%hr_dsp, int(this%nheld, i8), int(this%nrecv, i8))
       t2 = MPI_Wtime(); this%t_phase(2) = this%t_phase(2) + t2 - t1
       call ext_dev_forward(this%ctx)
       call dev_exchange(this, 3, 4, this%ms_cnt, this%ms_dsp, this%mr_cnt, this%mr_dsp, &
            int(this%nr, i8) * this%nz, sum(int(this%mr_cnt, i8)))
       t1 = MPI_Wtime(); this%t_phase(3) = this%t_phase(3) + t1 - t2
       call ext_dev_solve(this%ctx)
       t2 = MPI_Wtime(); this%t_phase(4) = this%t_phase(4) + t2 - t1
       call dev_exchange(this, 4, 3, this%mr_cnt, this%mr_dsp, this%ms_cnt, this%ms_dsp, &
            sum(int(this%mr_cnt, i8)), int(this%nr, i8) * this%nz)
       call ext_dev_backward(this%ctx)
       call dev_exchange(this, 2, 1, this%hr_cnt, this%hr_dsp, this%hs_cnt, this%hs_dsp, int(this%nrecv, i8), int(this%nheld, i8))
       call ext_dev_scatter(this%ctx, z_d, rpb)
       t1 = MPI_Wtime(); this%t_phase(5) = this%t_phase(5) + t1 - t2
    else if (NEKO_BCKND_DEVICE .eq. 1) then
       r_d = device_get_ptr(r)
       z_d = device_get_ptr(z)
       call device_memcpy(this%hr, r_d, n, DEVICE_TO_HOST, sync = .true.)
       call host_apply(this, this%hz, this%hr)
       call device_memcpy(this%hz, z_d, n, HOST_TO_DEVICE, sync = .true.)
    else
       this%hr = r
       call host_apply(this, this%hz, this%hr)
       z = this%hz
    end if
    ! timers: the first 4 solves (code loading, rocBLAS) are not counted
    if (this%nsolve .le. 4) then
       this%t_phase = 0.0_dp
    else
       this%ntimed = this%ntimed + 1
       this%t_solve = this%t_solve + MPI_Wtime() - t0
       this%t_max = max(this%t_max, MPI_Wtime() - t0)
    end if
    if (mod(this%nsolve, 50) .eq. 0 .and. this%ntimed .gt. 0) then
       write(log_buf, '(A,I0,A,F8.2,A,F8.2,A,5F7.2,A)') 'extruded: ', this%nsolve, ' solves, ', &
            1.0e3_dp * this%t_solve / this%ntimed, ' ms per solve (max ', 1.0e3_dp * this%t_max, &
            '; gather, exchange, z+exchange, sweep, back: ', 1.0e3_dp * this%t_phase / this%ntimed, ', rank 0)'
       call neko_log%message(log_buf)
    end if
  end subroutine ext_solve

  !> all-to-all-v between two device buffers of the context (device MPI, or
  !! staged through the host)
  subroutine dev_exchange(this, from, to, scnt, sdsp, rcnt, rdsp, ns, nr)
    class(ext_pc_t), intent(inout), target :: this
    integer, intent(in) :: from, to, scnt(:), sdsp(:), rcnt(:), rdsp(:)
    integer(kind=i8), intent(in) :: ns, nr
    real(kind=dp), pointer :: sb(:), rb(:)
    integer :: ierr
    type(c_ptr) :: ps, pr, hs, hr
    integer(kind=i8) :: m
    if (.not. this%stage) then
       call c_f_pointer(ext_dev_ptr(this%ctx, from), sb, [max(ns, 1_i8)])
       call c_f_pointer(ext_dev_ptr(this%ctx, to), rb, [max(nr, 1_i8)])
       call MPI_Alltoallv(sb, scnt, sdsp, MPI_DOUBLE_PRECISION, rb, rcnt, rdsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    else
       ! no device-aware MPI: through host buffers
       m = max(ns, nr, 1_i8)
       if (allocated(this%hstage)) then
          if (size(this%hstage, 1, kind=i8) .lt. m) deallocate(this%hstage)
       end if
       if (.not. allocated(this%hstage)) allocate(this%hstage(m, 2))
       ps = ext_dev_ptr(this%ctx, from)
       pr = ext_dev_ptr(this%ctx, to)
       hs = c_loc(this%hstage(1, 1))
       hr = c_loc(this%hstage(1, 2))
       if (ns .gt. 0) call device_memcpy(hs, ps, int(8 * ns, c_size_t), DEVICE_TO_HOST, sync = .true.)
       call MPI_Alltoallv(this%hstage(:, 1), scnt, sdsp, MPI_DOUBLE_PRECISION, this%hstage(:, 2), rcnt, rdsp, &
            MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
       if (nr .gt. 0) call device_memcpy(hr, pr, int(8 * nr, c_size_t), HOST_TO_DEVICE, sync = .true.)  ! (host, device)
    end if
  end subroutine dev_exchange

  subroutine host_apply(this, z, r)
    class(ext_pc_t), intent(inout) :: this
    real(kind=rp), intent(out) :: z(:)
    real(kind=rp), intent(in) :: r(:)
    integer :: i, ierr, nz, c, q, m, s, nf, src, ns, p
    integer(kind=i8) :: off
    nz = this%nz; nf = this%nfree
    ! element layout -> pencils
    do i = 1, this%nheld
       this%sb(i) = real(r(this%gsend(i)), dp)
    end do
    call MPI_Alltoallv(this%sb, this%hs_cnt, this%hs_dsp, MPI_DOUBLE_PRECISION, this%rb, this%hr_cnt, &
         this%hr_dsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    this%X = 0.0_dp
    do i = 1, this%nrecv
       this%X(mod(this%rpos(i) - 1, max(this%nr, 1)) + 1, (this%rpos(i) - 1) / max(this%nr, 1) + 1) = this%rb(i)
    end do
    ! z transform: Y = X Sp
    if (this%nr .gt. 0) call dgemm('N', 'N', this%nr, nz, nz, 1.0_dp, this%X, max(this%nr, 1), this%Sp, nz, &
         0.0_dp, this%Y, max(this%nr, 1))
    ! pencils -> mode planes: mp = [src][local mode][n_src]
    call MPI_Alltoallv(this%Y, this%ms_cnt, this%ms_dsp, MPI_DOUBLE_PRECISION, this%mp, this%mr_cnt, &
         this%mr_dsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    do c = 1, size(this%slot)
       associate(sl => this%slot(c))
         do q = 1, sl%nm
            m = sl%m(q)
            do i = 1, nf
               p = this%chol%perm(i)
               this%wv(i, q) = sl%d(p) * this%mp(mpos(p, m))
            end do
         end do
         call host_sweep(this, sl%g, sl%nm)
         do q = 1, sl%nm
            m = sl%m(q)
            do i = 1, nf
               p = this%chol%perm(i)
               this%mp(mpos(p, m)) = sl%d(p) * this%wv(i, q)
            end do
         end do
       end associate
    end do
    call MPI_Alltoallv(this%mp, this%mr_cnt, this%mr_dsp, MPI_DOUBLE_PRECISION, this%Y, this%ms_cnt, &
         this%ms_dsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    if (this%nr .gt. 0) call dgemm('N', 'T', this%nr, nz, nz, 1.0_dp, this%Y, max(this%nr, 1), this%Sp, nz, &
         0.0_dp, this%X, max(this%nr, 1))
    do i = 1, this%nrecv
       this%rb(i) = this%X(mod(this%rpos(i) - 1, max(this%nr, 1)) + 1, (this%rpos(i) - 1) / max(this%nr, 1) + 1)
    end do
    call MPI_Alltoallv(this%rb, this%hr_cnt, this%hr_dsp, MPI_DOUBLE_PRECISION, this%sb, this%hs_cnt, &
         this%hs_dsp, MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    do p = 1, this%n
       if (this%sidx(p) .gt. 0) then
          z(p) = real(this%sb(this%sidx(p)), rp)
       else
          z(p) = 0.0_rp
       end if
    end do
  contains
    !> position of free node p (1-based) of local mode m in mp
    integer function mpos(pp, mm)
      integer, intent(in) :: pp, mm
      integer :: sr, lo, hi, mid
      lo = 0; hi = pe_size - 1
      do while (lo .lt. hi)
         mid = (lo + hi + 1) / 2
         if (this%f0(mid) .le. pp) then
            lo = mid
         else
            hi = mid - 1
         end if
      end do
      sr = lo
      mpos = this%mr_dsp(sr) + (mm - 1) * (this%f0(sr + 1) - this%f0(sr)) + (pp - this%f0(sr)) + 1
    end function mpos
  end subroutine host_apply

  !> forward and backward sweep with the G blocks on wv(:, 1:nv) (permuted)
  subroutine host_sweep(this, g, nv)
    class(ext_pc_t), intent(inout) :: this
    real(kind=dp), intent(in) :: g(:)
    integer, intent(in) :: nv
    integer :: s, f, k, m, r0, i, q
    integer(kind=i8) :: o
    real(kind=dp), allocatable :: t(:), v(:)
    allocate(t(this%chol%maxm), v(this%chol%maxm))
    associate(ch => this%chol)
      do q = 1, nv
         ! forward: t = G b(f:f+k-1); y = t(1:k); b(rows) -= t(k+1:)
         do s = 1, ch%nsuper
            f = ch%sfc(s); k = ch%sfc(s + 1) - f
            r0 = ch%srp(s); m = ch%srp(s + 1) - r0
            o = ch%slp(s)
            call dgemv('N', m, k, 1.0_dp, g(o + 1), m, this%wv(f, q), 1, 0.0_dp, t, 1)
            this%wy(f:f + k - 1, q) = t(1:k)
            do i = k + 1, m
               this%wv(ch%srow(r0 + i - 1), q) = this%wv(ch%srow(r0 + i - 1), q) - t(i)
            end do
         end do
         ! backward: x(f:) = G^T [y(f:); -x(rows)]
         do s = ch%nsuper, 1, -1
            f = ch%sfc(s); k = ch%sfc(s + 1) - f
            r0 = ch%srp(s); m = ch%srp(s + 1) - r0
            o = ch%slp(s)
            v(1:k) = this%wy(f:f + k - 1, q)
            do i = k + 1, m
               v(i) = -this%wv(ch%srow(r0 + i - 1), q)
            end do
            call dgemv('T', m, k, 1.0_dp, g(o + 1), m, v, 1, 0.0_dp, t, 1)
            this%wv(f:f + k - 1, q) = t(1:k)
         end do
      end do
    end associate
  end subroutine host_sweep

  ! ---------------------------------------------------------------------------
  ! utilities
  ! ---------------------------------------------------------------------------

  !> merge points within tol: ids 1..nu in the order of their first occurrence
  subroutine merge_points(xy, np, tol, id, nu)
    integer, intent(in) :: np
    real(kind=dp), intent(in) :: xy(2, np), tol
    integer, allocatable, intent(out) :: id(:)
    integer, intent(out) :: nu
    integer(kind=i8), allocatable :: ck(:)
    integer, allocatable :: ord(:), par(:), root_id(:)
    integer(kind=i8) :: cx, cy, kk
    integer :: i, j, a, b, lo, hi, dx, dy, p
    real(kind=dp) :: h, x0, y0
    h = 4.0_dp * tol
    x0 = minval(xy(1, :)); y0 = minval(xy(2, :))
    allocate(ck(np), ord(np), par(np))
    do i = 1, np
       cx = int((xy(1, i) - x0) / h, i8); cy = int((xy(2, i) - y0) / h, i8)
       ck(i) = cx * 2147483648_i8 + cy
       ord(i) = i
       par(i) = i
    end do
    call sort_key_index(ck, ord, np)
    do i = 1, np
       a = ord(i)
       cx = ck(a) / 2147483648_i8; cy = mod(ck(a), 2147483648_i8)
       do dx = -1, 1
          do dy = -1, 1
             kk = (cx + dx) * 2147483648_i8 + (cy + dy)
             ! binary search for the first point of cell kk
             lo = 1; hi = np
             do while (lo .lt. hi)
                j = (lo + hi) / 2
                if (ck(ord(j)) .lt. kk) then
                   lo = j + 1
                else
                   hi = j
                end if
             end do
             j = lo
             do while (j .le. np)
                b = ord(j)
                if (ck(b) .ne. kk) exit
                if (b .ne. a) then
                   if (abs(xy(1, a) - xy(1, b)) .le. tol .and. abs(xy(2, a) - xy(2, b)) .le. tol) &
                        call unite(a, b)
                end if
                j = j + 1
             end do
          end do
       end do
    end do
    allocate(id(np), root_id(np))
    root_id = 0
    nu = 0
    do i = 1, np
       p = find(i)
       if (root_id(p) .eq. 0) then
          nu = nu + 1
          root_id(p) = nu
       end if
       id(i) = root_id(p)
    end do
  contains
    recursive integer function find(x) result(rt)
      integer, intent(in) :: x
      if (par(x) .eq. x) then
         rt = x
      else
         rt = find(par(x))
         par(x) = rt
      end if
    end function find
    subroutine unite(x, y)
      integer, intent(in) :: x, y
      integer :: rx, ry
      rx = find(x); ry = find(y)
      if (rx .ne. ry) par(max(rx, ry)) = min(rx, ry)
    end subroutine unite
  end subroutine merge_points

  !> unique edges of the quads (sorted (va, vb), va < vb)
  subroutine quad_edges(qc, nq, eid, ned)
    integer, intent(in) :: qc(:,:), nq
    integer, allocatable, intent(out) :: eid(:,:)
    integer, intent(out) :: ned
    integer(kind=i8), allocatable :: k(:)
    integer, allocatable :: ord(:)
    integer :: q, c, a, b, i
    allocate(k(4 * nq), ord(4 * nq))
    do q = 1, nq
       do c = 1, 4
          a = qc(c, q); b = qc(mod(c, 4) + 1, q)
          k(4 * (q - 1) + c) = int(min(a, b), i8) * 2147483648_i8 + max(a, b)
          ord(4 * (q - 1) + c) = 4 * (q - 1) + c
       end do
    end do
    call sort_key_index(k, ord, 4 * nq)
    allocate(eid(2, 4 * nq))
    ned = 0
    do i = 1, 4 * nq
       if (i .gt. 1) then
          if (k(ord(i)) .eq. k(ord(i - 1))) cycle
       end if
       ned = ned + 1
       eid(1, ned) = int(k(ord(i)) / 2147483648_i8)
       eid(2, ned) = int(mod(k(ord(i)), 2147483648_i8))
    end do
  end subroutine quad_edges

  !> sorted unique values with a tolerance (in place, count nu)
  subroutine sort_unique_tol(a, n, tol, nu)
    real(kind=dp), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(kind=dp), intent(in) :: tol
    integer, intent(out) :: nu
    integer :: i
    call dsort(a(1:n))
    nu = 1
    do i = 2, n
       if (a(i) - a(nu) .gt. tol) then
          nu = nu + 1
          a(nu) = a(i)
       end if
    end do
  end subroutine sort_unique_tol

  integer function locate(zl, nzl, zv)
    real(kind=dp), intent(in) :: zl(:), zv
    integer, intent(in) :: nzl
    integer :: i
    locate = 1
    do i = 1, nzl - 1
       if (abs(zl(i) - zv) .lt. abs(zl(locate) - zv)) locate = i
    end do
  end function locate

  subroutine isort4(a)
    integer, intent(inout) :: a(4)
    integer :: i, j, t
    do i = 2, 4
       t = a(i); j = i - 1
       do while (j .ge. 1)
          if (a(j) .le. t) exit
          a(j + 1) = a(j); j = j - 1
       end do
       a(j + 1) = t
    end do
  end subroutine isort4

  subroutine isort(a)
    integer, intent(inout) :: a(:)
    integer :: i, j, t
    do i = 2, size(a)
       t = a(i); j = i - 1
       do while (j .ge. 1)
          if (a(j) .le. t) exit
          a(j + 1) = a(j); j = j - 1
       end do
       a(j + 1) = t
    end do
  end subroutine isort

  !> heapsort of a(:) (real)
  subroutine dsort(a)
    real(kind=dp), intent(inout) :: a(:)
    integer :: n, i, j, k
    real(kind=dp) :: t
    n = size(a)
    do i = n / 2, 1, -1
       call sift(i, n)
    end do
    do i = n, 2, -1
       t = a(1); a(1) = a(i); a(i) = t
       call sift(1, i - 1)
    end do
  contains
    subroutine sift(lo, hi)
      integer, intent(in) :: lo, hi
      integer :: r, c
      real(kind=dp) :: v
      r = lo; v = a(r)
      do
         c = 2 * r
         if (c .gt. hi) exit
         if (c .lt. hi) then
            if (a(c + 1) .gt. a(c)) c = c + 1
         end if
         if (a(c) .le. v) exit
         a(r) = a(c); r = c
      end do
      a(r) = v
    end subroutine sift
  end subroutine dsort

  !> sort idx(1:n) so that key(idx) ascends (heapsort, stable enough for ties
  !! broken by idx)
  subroutine sort_key_index(key, idx, n)
    integer(kind=i8), intent(in) :: key(:)
    integer, intent(inout) :: idx(:)
    integer, intent(in) :: n
    integer :: i, t
    do i = n / 2, 1, -1
       call sift(i, n)
    end do
    do i = n, 2, -1
       t = idx(1); idx(1) = idx(i); idx(i) = t
       call sift(1, i - 1)
    end do
  contains
    logical function gt(a, b)
      integer, intent(in) :: a, b
      gt = key(a) .gt. key(b) .or. (key(a) .eq. key(b) .and. a .gt. b)
    end function gt
    subroutine sift(lo, hi)
      integer, intent(in) :: lo, hi
      integer :: r, c, v
      r = lo; v = idx(r)
      do
         c = 2 * r
         if (c .gt. hi) exit
         if (c .lt. hi) then
            if (gt(idx(c + 1), idx(c))) c = c + 1
         end if
         if (.not. gt(idx(c), v)) exit
         idx(r) = idx(c); r = c
      end do
      idx(r) = v
    end subroutine sift
  end subroutine sort_key_index

  ! ---------------------------------------------------------------------------
  ! the solver type
  ! ---------------------------------------------------------------------------

  subroutine ext_ksp_init(this, n, max_iter, M, rel_tol, abs_tol, monitor)
    class(ext_ksp_t), target, intent(inout) :: this
    integer, intent(in) :: max_iter
    class(pc_t), optional, intent(in), target :: M
    integer, intent(in) :: n
    real(kind=rp), optional, intent(in) :: rel_tol
    real(kind=rp), optional, intent(in) :: abs_tol
    logical, optional, intent(in) :: monitor
    call this%free()
    this%n = n
    if (present(M)) this%M => M
    ! refinement steps: max_iterations - 1 (at least one solve)
    this%refine = max(0, min(max_iter - 1, 3))
    call this%ksp_init(max_iter, rel_tol = rel_tol, abs_tol = abs_tol, monitor = monitor)
  end subroutine ext_ksp_init

  subroutine ext_ksp_free(this)
    class(ext_ksp_t), intent(inout) :: this
    call this%ksp_free()
    if (allocated(this%w)) then
       if (c_associated(this%w_d)) call device_unmap(this%w, this%w_d)
       deallocate(this%w)
    end if
    if (allocated(this%w2)) then
       if (c_associated(this%w2_d)) call device_unmap(this%w2, this%w2_d)
       deallocate(this%w2)
    end if
  end subroutine ext_ksp_free

  function ext_ksp_norm(f, coef, n) result(nrm)
    integer, intent(in) :: n
    real(kind=rp), intent(in) :: f(n)
    type(coef_t), intent(in) :: coef
    real(kind=rp) :: nrm
    type(c_ptr) :: f_d
    if (NEKO_BCKND_DEVICE .eq. 1) then
       f_d = device_get_ptr(f)
       nrm = sqrt(device_glsc3(f_d, f_d, coef%mult_d, n) / coef%volume)
    else
       nrm = sqrt(glsc3(f, f, coef%mult, n) / coef%volume)
    end if
  end function ext_ksp_norm

  !> x = M f, then "refine" times: w = f - A x, x += M w; the true residual
  !! after the last step is reported
  function ext_ksp_solve(this, Ax, x, f, n, coef, bc_projector, gs_h, niter) result(ksp_results)
    class(ext_ksp_t), intent(inout) :: this
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(in) :: f
    type(coef_t), intent(inout) :: coef
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    integer, optional, intent(in) :: niter
    type(ksp_monitor_t) :: ksp_results
    type(c_ptr) :: f_d, x_d
    real(kind=rp), allocatable, target :: fc(:)
    integer :: it

    if (.not. associated(this%M)) call neko_error('the "extruded" solver needs the "extruded" preconditioner')
    if (.not. allocated(this%w)) then
       allocate(this%w(n), this%w2(n))
       this%w = 0.0_rp; this%w2 = 0.0_rp
       if (NEKO_BCKND_DEVICE .eq. 1) then
          call device_map(this%w, this%w_d, n)
          call device_map(this%w2, this%w2_d, n)
       end if
    end if
    ksp_results%res_start = ext_ksp_norm(f, coef, n)
    call this%monitor_start('extruded')
    ! the preconditioner's solve takes intent(inout) arguments: a copy of f
    if (NEKO_BCKND_DEVICE .eq. 1) then
       f_d = device_get_ptr(f)
       call device_copy(this%w2_d, f_d, n)
    else
       this%w2 = f
    end if
    call this%M%solve(x%x, this%w2, n)
    do it = 1, this%refine + 1
       ! w = f - A x
       call Ax%compute(this%w, x%x, coef, x%msh, x%Xh)
       call gs_h%op(this%w, n, GS_OP_ADD)
       call bc_projector%apply(this%w, n)
       if (NEKO_BCKND_DEVICE .eq. 1) then
          f_d = device_get_ptr(f)
          call device_sub2(this%w_d, f_d, n)      ! w = A x - f
       else
          call sub2(this%w, f, n)
       end if
       ksp_results%res_final = ext_ksp_norm(this%w, coef, n)
       call this%monitor_iter(it, ksp_results%res_final)
       ksp_results%iter = it
       ! done at the last step, or as soon as the tolerance is met
       if (it .eq. this%refine + 1 .or. ksp_results%res_final .lt. this%abs_tol) exit
       ! x -= M (A x - f)
       call this%M%solve(this%w2, this%w, n)
       if (NEKO_BCKND_DEVICE .eq. 1) then
          x_d = x%x_d
          call device_sub2(x_d, this%w2_d, n)
       else
          call sub2(x%x, this%w2, n)
       end if
    end do
    call this%monitor_stop()
    ksp_results%converged = .true.
  end function ext_ksp_solve

  function ext_ksp_solve_coupled(this, Ax, x, y, z, fx, fy, fz, n, coef, bc_projector, gs_h, niter) &
       result(ksp_results)
    class(ext_ksp_t), intent(inout) :: this
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    type(field_t), intent(inout) :: y
    type(field_t), intent(inout) :: z
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(in) :: fx
    real(kind=rp), dimension(n), intent(in) :: fy
    real(kind=rp), dimension(n), intent(in) :: fz
    type(coef_t), intent(inout) :: coef
    class(vector_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    integer, optional, intent(in) :: niter
    type(ksp_monitor_t), dimension(3) :: ksp_results
    call neko_error('extruded: the solver type is for the pressure only')
  end function ext_ksp_solve_coupled

  subroutine ext_free(this)
    class(ext_pc_t), intent(inout) :: this
    this%ready = .false.
    if (allocated(this%slot)) deallocate(this%slot)
    call this%chol%free()
  end subroutine ext_free

end module extruded
