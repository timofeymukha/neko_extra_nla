!> box_fdm: the fast-diagonalization direct solver of box meshes as a Neko
!! preconditioner, injected through Neko's type registry (no change to Neko).
!!
!! Build:  makeneko box_fdm.f90 fdm_dev.hip   (fdm_dev.cu on NVIDIA;
!!         makeneko box_fdm.f90 fdm_dev_stub.f90 with a CPU Neko: the host
!!         path, any number of ranks) and use "preconditioner": {"type":
!!         "fdm"} for the pressure, the velocity or a scalar.
!!
!! On an axis-aligned box of hexahedral elements (any element sizes per
!! direction, periodic, Dirichlet or natural faces) Neko's Helmholtz operator
!! H = h1 A + h2 B with constant h1, h2 is the tensor product of 1D assembled
!! operators, and H^{-1} = S (h1 Lambda + h2)^{-1} S^T with S = Sx x Sy x Sz
!! the B-orthonormal 1D eigenbases. The preconditioner applies that exactly on
!! the free unique nodes, so CG and GMRES converge in one iteration (the
!! pressure's residual to round-off times its condition number); with a
!! non-uniform h1 or h2 (variable viscosity, SVV, the stress formulation) it
!! is still a preconditioner (a warning says so); a deformed box is refused
!! (its points are off the tensor-product grid). Singular operators (the
!! pressure: no Dirichlet node, h2 = 0) get the pseudo-inverse (the constant
!! mode dropped).
!!
!! Everything is derived from the dofmap: the element edges per direction
!! from the elements' extents, the global 1D GLL grids, each point's (ix, iy,
!! iz), the periodic directions from the mesh's periodic zone, the Dirichlet
!! faces from the masks of the bc list's Dirichlet conditions.
!!
!! Several ranks: the free grid is split into z-slabs (the x and y
!! transforms are local), one all-to-all transposes it into z-pencils for the
!! z transform and the scaling, and back; the element layout and the slabs
!! are connected by index lists (every rank sends the nodes it holds).
!! Options: "rank_grid": [Py, Pz] (Py > 1) selects a 2D pencil decomposition
!! (x whole, then y whole, then z whole, transposes inside groups of Py and
!! Pz ranks) for P beyond the number of z planes; "exchange_precision":
!! "single" moves the exchanges as floats (the transforms stay double; the
!! pressure solution is then accurate to ~1e-5 relative, velocity and scalar
!! to ~1e-7). The CPU backend runs the same steps on host buffers (BLAS
!! dgemm, MPI on host memory), with x applied from the right so that
!! uniform periodic directions use Bloch bases ("bloch") and mirror-
!! symmetric ones even/odd bases ("fold"): fewer flops. Limits: the Dirichlet nodes must form whole
!! box faces, no mixed (symmetry) conditions.
module box_fdm
  use num_types, only : rp, dp
  use precon, only : pc_t, register_precon, precon_allocate
  use krylov, only : ksp_t, ksp_monitor_t, register_krylov, krylov_allocate
  use ax_product, only : ax_t
  use field, only : field_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use scalar_bc_projector, only : scalar_bc_projector_t
  use vector_bc_projector, only : vector_bc_projector_t
  use device_math, only : device_glsc3, device_sub2
  use math, only : glsc3, sub2
  use coefs, only : coef_t
  use bc_list, only : bc_list_t
  use bc, only : bc_t, BC_DIRICHLET, BC_NEUMANN
  use json_module, only : json_file
  use json_utils, only : json_get_or_default, json_get
  use comm, only : NEKO_COMM, pe_rank, pe_size
  use mpi_f08, only : MPI_Comm, MPI_Comm_split, MPI_Comm_free, MPI_Request, MPI_Ialltoallv, &
       MPI_Waitall, MPI_Wait, MPI_STATUS_IGNORE, MPI_STATUSES_IGNORE, &
       MPI_Allreduce, MPI_Allgather, MPI_Allgatherv, &
       MPI_Alltoall, MPI_Alltoallv, MPI_Wtime, MPI_Barrier, &
       MPI_IN_PLACE, MPI_INTEGER, MPI_INTEGER8, MPI_DOUBLE_PRECISION, MPI_REAL4, &
       MPI_SUM, MPI_MAX, MPI_MIN
  use utils, only : neko_error
  use logger, only : neko_log, LOG_SIZE
  use neko_config, only : NEKO_BCKND_DEVICE, NEKO_DEVICE_MPI
  use device, only : device_get_ptr, glb_cmd_queue, device_map, device_unmap
  use, intrinsic :: iso_c_binding
  implicit none
  private

  public :: box_fdm_register_types

  !> The real Bloch basis of a periodic direction with equal elements (host
  !! path): S = (C x I_N)^T blockdiag(V_b), C the real orthogonal transform
  !! over the E element indices (rows k = 0, (E/2), then cos/sin pairs), V_b
  !! the B-orthonormal eigenbases of the blocks (N or 2N columns, contiguous
  !! in the transformed index m + N q). Applied from the right to a matrix
  !! whose columns are the direction's nodes: E + 2N MACs per point instead
  !! of the dense basis's E N (prototypes/bloch_real.py).
  type :: bloch_t
     logical :: on = .false.
     integer :: E = 0, N = 0, nb = 0
     real(kind=rp), allocatable :: C(:,:)      !< (E, E)
     real(kind=rp), allocatable :: V(:,:,:)    !< (2N, 2N, nb), block b in V(1:sz, 1:sz, b)
     integer, allocatable :: off(:), sz(:)     !< 0-based first column and size of block b
  end type bloch_t

  !> The even/odd basis of a mirror-symmetric non-periodic direction (host
  !! path, see setup_fold): ne even modes first, then no odd ones.
  type :: fold_t
     logical :: on = .false.
     integer :: n = 0, ne = 0, no = 0
     real(kind=rp), allocatable :: Ve(:,:), Vo(:,:)
  end type fold_t

  type, public, extends(pc_t) :: fdm_pc_t
     type(coef_t), pointer :: coef => null()
     type(bc_list_t), pointer :: bclst => null()
     logical :: ready = .false.         !< set up (at the first update: the bc masks exist then)
     integer :: lx = 0, n = 0
     integer :: ne(3) = 0               !< elements per direction
     integer :: nfull(3) = 0            !< distinct node coordinates per direction
     integer :: nu(3) = 0               !< unique nodes (nfull - 1 if periodic)
     integer :: nf(3) = 0               !< free nodes (no Dirichlet)
     integer :: lo(3) = 0               !< 1 if the low end is Dirichlet (index offset)
     logical :: periodic(3) = .false.
     logical :: dir_lo(3) = .false., dir_hi(3) = .false.
     integer :: zero_mode(3) = 0        !< 1-based index of the zero eigenvalue, 0 if none
     real(kind=rp), allocatable :: Sx(:,:), Sy(:,:), Sz(:,:)
     real(kind=rp), allocatable :: lamx(:), lamy(:), lamz(:)
     !> The distributed layout: z-slabs of the free grid (zs0, nzl planes of
     !! this rank) for the x and y transforms, z-pencils (ncl columns of the
     !! (x, y) column index from cs_me) for the z transform
     integer :: P = 1, me = 0, nzl = 0, ncl = 0, cs_me = 0, zs0 = 0
     integer(kind=8) :: nheld = 0, nrecv = 0, nslab = 0, npencil = 0
     integer, allocatable :: gsend(:)   !< per held free node: one element-layout copy
     integer, allocatable :: rpos(:)    !< per received value: its slab position
     integer, allocatable :: sidx(:)    !< per element-layout point: held node or 0
     integer, allocatable :: cdest(:), cstride(:)   !< packed position of a slab column
     integer, allocatable :: fs_cnt(:), fs_dsp(:), fr_cnt(:), fr_dsp(:)   !< node exchange
     integer, allocatable :: ts_cnt(:), ts_dsp(:), tr_cnt(:), tr_dsp(:)   !< transpose
     real(kind=rp), allocatable :: hs(:), hr(:)     !< host staging (no device MPI)
     !> Overlap of the slab scheme's transposes with the transforms: the
     !! planes in nch chunks, one non-blocking all-to-all per chunk
     integer :: nch = 1
     integer, allocatable :: zc(:,:)                !< (0:nch, P) chunk bounds of every rank
     integer, allocatable :: cs_cnt(:,:), cs_dsp(:,:), cr_cnt(:,:), cr_dsp(:,:)   !< (P, nch) forward
     !> The 2D (pencil) scheme ("rank_grid": [Py, Pz], Py > 1): layouts A
     !! (x whole), B (y whole), C (z whole); transposes inside the k-group
     !! (comm_k, Py ranks) and the j-group (comm_j, Pz ranks)
     logical :: twod = .false.
     integer :: Py = 1, Pz = 1, jme = 0, kme = 0
     integer :: nyA = 0, nzA = 0, nxB = 0, nyC = 0, x0B = 0, y0C = 0
     integer(kind=8) :: nA = 0, nB = 0, nC = 0
     type(MPI_Comm) :: comm_k, comm_j
     integer, allocatable :: t1s_cnt(:), t1s_dsp(:), t1r_cnt(:), t1r_dsp(:)   !< A <-> B
     integer, allocatable :: t2s_cnt(:), t2s_dsp(:), t2r_cnt(:), t2r_dsp(:)   !< B <-> C
     integer, allocatable :: posA(:), tabB(:), posB(:), tabC(:)              !< 1-based permutations
     !> The host path (CPU backend): the device path's buffers in host
     !! memory. sb (nheld) and rb (nrecv) the node exchange, a and t the slab
     !! and its packed form, pb and pc the pencil; s0 ... s3 the 2D scheme.
     !! On one rank rb aliases sb and pb aliases t (no exchange). The host
     !! slab runs y fastest in a plane (x is applied from the right).
     real(kind=rp), pointer, contiguous :: hsb(:) => null(), hrb(:) => null()
     real(kind=rp), pointer, contiguous :: ha(:) => null(), ht(:) => null()
     real(kind=rp), pointer, contiguous :: hpb(:) => null(), hpc(:) => null()
     real(kind=rp), pointer, contiguous :: hs0(:) => null(), hs1(:) => null()
     real(kind=rp), pointer, contiguous :: hs2(:) => null(), hs3(:) => null()
     real(kind=c_float), allocatable :: x4s(:), x4r(:)   !< single-precision exchanges
     real(kind=rp), allocatable :: lxy(:)                !< lamx + lamy of the pencil columns
     type(c_ptr) :: ctx = C_NULL_PTR
     real(kind=rp) :: h1 = 1.0_rp, h2 = 0.0_rp
     logical :: ifh2 = .false.
     integer :: nsolve = 0
     logical :: verbose = .false.
     logical :: profile = .false.       !< time the solve's phases (synchronizing)
     real(kind=rp) :: tprof(15) = 0.0_rp
     integer :: nupdate = 0
     logical :: warned = .false.        !< non-uniform h1 or h2 reported
     logical :: residual = .false.      !< the fdm solver type computes the true residual
     logical :: single = .false.        !< the exchanges in single precision
     logical :: use_bloch = .true.      !< host path: Bloch bases of uniform periodic directions
     type(bloch_t) :: bl(3)
     type(fold_t) :: fd(3)
     logical :: use_fold = .true.       !< host path: even/odd bases of mirror-symmetric directions
     real(kind=rp), pointer, contiguous :: hwk(:) => null()   !< host: the Bloch transforms' workspace
   contains
     procedure, pass(this) :: init => fdm_init
     procedure, pass(this) :: solve => fdm_solve
     procedure, pass(this) :: update => fdm_update
     procedure, pass(this) :: free => fdm_free
  end type fdm_pc_t

  !> The direct solve as a Krylov solver type "fdm": x = H^{-1} f once, no
  !! Krylov vectors (GMRES keeps 2 x 30 fields), one iteration reported.
  !! Needs the "fdm" preconditioner (the solve is its). The true residual
  !! (one operator application) is computed only with "residual": true in
  !! the preconditioner's dictionary; otherwise 0 is reported.
  type, public, extends(ksp_t) :: fdm_ksp_t
     real(kind=rp), allocatable :: w(:), w2(:), w3(:)
     type(c_ptr) :: w_d = C_NULL_PTR, w2_d = C_NULL_PTR, w3_d = C_NULL_PTR
     integer :: n = 0
     integer :: ncall = 0                 !< solves so far
     real(kind=rp) :: last_res(3) = -1.0_rp   !< the last true residuals measured
   contains
     procedure, pass(this) :: init => fdm_ksp_init
     procedure, pass(this) :: solve => fdm_ksp_solve
     procedure, pass(this) :: solve_coupled => fdm_ksp_solve_coupled
     procedure, pass(this) :: free => fdm_ksp_free
  end type fdm_ksp_t

  interface
     function fdm_dev_setup(stream, nfx, nfy, nfz, nzl, ncl, cs_me, n, nheld, nrecv, one_rank, single, &
          Sx, Sy, Sz, lamx, lamy, lamz, gsend, rpos, sidx, cdest, cstride) &
          bind(c, name = 'fdm_dev_setup') result(ctx)
       import c_ptr, c_int, c_double, c_long_long
       type(c_ptr), value :: stream
       integer(c_int), value :: nfx, nfy, nfz, nzl, ncl, cs_me, n, one_rank, single
       integer(c_long_long), value :: nheld, nrecv
       real(c_double) :: Sx(*), Sy(*), Sz(*), lamx(*), lamy(*), lamz(*)
       integer(c_int) :: gsend(*), rpos(*), sidx(*), cdest(*), cstride(*)
       type(c_ptr) :: ctx
     end function fdm_dev_setup
     subroutine fdm_dev_free(ctx) bind(c, name = 'fdm_dev_free')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine fdm_dev_free
     function fdm_dev_ptr(ctx, which) bind(c, name = 'fdm_dev_ptr') result(p)
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: which
       type(c_ptr) :: p
     end function fdm_dev_ptr
     subroutine fdm_dev_copy(ctx, from, to, count) bind(c, name = 'fdm_dev_copy')
       import c_ptr, c_int, c_long_long
       type(c_ptr), value :: ctx
       integer(c_int), value :: from, to
       integer(c_long_long), value :: count
     end subroutine fdm_dev_copy
     subroutine fdm_dev_sync(ctx) bind(c, name = 'fdm_dev_sync')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine fdm_dev_sync
     subroutine fdm_dev_d2h(ctx, which, host, count) bind(c, name = 'fdm_dev_d2h')
       import c_ptr, c_int, c_double, c_long_long
       type(c_ptr), value :: ctx
       integer(c_int), value :: which
       real(c_double) :: host(*)
       integer(c_long_long), value :: count
     end subroutine fdm_dev_d2h
     subroutine fdm_dev_h2d(ctx, which, host, count) bind(c, name = 'fdm_dev_h2d')
       import c_ptr, c_int, c_double, c_long_long
       type(c_ptr), value :: ctx
       integer(c_int), value :: which
       real(c_double) :: host(*)
       integer(c_long_long), value :: count
     end subroutine fdm_dev_h2d
     subroutine fdm_dev2_setup(ctx, nA, nB, nC, nyA, nzA, nxB, nyC, x0B, y0C, posA, tabB, posB, tabC) &
          bind(c, name = 'fdm_dev2_setup')
       import c_ptr, c_int, c_long_long
       type(c_ptr), value :: ctx
       integer(c_long_long), value :: nA, nB, nC
       integer(c_int), value :: nyA, nzA, nxB, nyC, x0B, y0C
       integer(c_int) :: posA(*), tabB(*), posB(*), tabC(*)
     end subroutine fdm_dev2_setup
     subroutine fdm_dev2_step(ctx, which, h1, h2, null_idx) bind(c, name = 'fdm_dev2_step')
       import c_ptr, c_int, c_double, c_long_long
       type(c_ptr), value :: ctx
       integer(c_int), value :: which
       real(c_double), value :: h1, h2
       integer(c_long_long), value :: null_idx
     end subroutine fdm_dev2_step
     subroutine fdm_dev_chunks(ctx, nch) bind(c, name = 'fdm_dev_chunks')
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: nch
     end subroutine fdm_dev_chunks
     subroutine fdm_dev_event_record(ctx, k) bind(c, name = 'fdm_dev_event_record')
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: k
     end subroutine fdm_dev_event_record
     subroutine fdm_dev_event_sync(ctx, k) bind(c, name = 'fdm_dev_event_sync')
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: k
     end subroutine fdm_dev_event_sync
     subroutine fdm_dev_slab_scatter(ctx) bind(c, name = 'fdm_dev_slab_scatter')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine fdm_dev_slab_scatter
     subroutine fdm_dev_slab_in_planes(ctx, z0, z1) bind(c, name = 'fdm_dev_slab_in_planes')
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: z0, z1
     end subroutine fdm_dev_slab_in_planes
     subroutine fdm_dev_slab_out_planes(ctx, z0, z1) bind(c, name = 'fdm_dev_slab_out_planes')
       import c_ptr, c_int
       type(c_ptr), value :: ctx
       integer(c_int), value :: z0, z1
     end subroutine fdm_dev_slab_out_planes
     subroutine fdm_dev_slab_gather(ctx) bind(c, name = 'fdm_dev_slab_gather')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine fdm_dev_slab_gather
     subroutine fdm_dev_gather(ctx, r_d) bind(c, name = 'fdm_dev_gather')
       import c_ptr
       type(c_ptr), value :: ctx, r_d
     end subroutine fdm_dev_gather
     subroutine fdm_dev_slab_in(ctx) bind(c, name = 'fdm_dev_slab_in')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine fdm_dev_slab_in
     subroutine fdm_dev_pencil(ctx, h1, h2, null_idx) bind(c, name = 'fdm_dev_pencil')
       import c_ptr, c_double, c_long_long
       type(c_ptr), value :: ctx
       real(c_double), value :: h1, h2
       integer(c_long_long), value :: null_idx
     end subroutine fdm_dev_pencil
     subroutine fdm_dev_slab_out(ctx) bind(c, name = 'fdm_dev_slab_out')
       import c_ptr
       type(c_ptr), value :: ctx
     end subroutine fdm_dev_slab_out
     subroutine fdm_dev_scatter(ctx, z_d) bind(c, name = 'fdm_dev_scatter')
       import c_ptr
       type(c_ptr), value :: ctx, z_d
     end subroutine fdm_dev_scatter
     function fdm_dev_nonuniform(stream, h_d, n, ref) bind(c, name = 'fdm_dev_nonuniform') result(cnt)
       import c_ptr, c_double, c_long_long
       type(c_ptr), value :: stream, h_d
       integer(c_long_long), value :: n
       real(c_double), value :: ref
       integer(c_long_long) :: cnt
     end function fdm_dev_nonuniform
     subroutine fdm_dev_download(dst, src_d, bytes, stream) &
          bind(c, name = 'fdm_dev_download')
       import c_ptr, c_size_t, c_double
       real(c_double) :: dst(*)
       type(c_ptr), value :: src_d, stream
       integer(c_size_t), value :: bytes
     end subroutine fdm_dev_download
  end interface

contains

  !> Called by makeneko's driver: adds the "fdm" preconditioner type.
  subroutine box_fdm_register_types()
    procedure(precon_allocate), pointer :: alloc
    procedure(krylov_allocate), pointer :: alloc_ksp
    alloc => fdm_allocate
    call register_precon("fdm", alloc)
    alloc_ksp => fdm_ksp_allocate
    call register_krylov("fdm", alloc_ksp)
  end subroutine box_fdm_register_types

  subroutine fdm_ksp_allocate(obj)
    class(ksp_t), allocatable, intent(inout) :: obj
    allocate(fdm_ksp_t :: obj)
  end subroutine fdm_ksp_allocate

  subroutine fdm_allocate(obj)
    class(pc_t), allocatable, intent(inout) :: obj
    allocate(fdm_pc_t :: obj)
  end subroutine fdm_allocate

  ! ---------------------------------------------------------------------------
  ! setup
  ! ---------------------------------------------------------------------------

  !> Stores the pointers; the setup itself waits for the first update(),
  !! because Neko creates the preconditioner before the boundary conditions
  !! are assigned (their masks are empty here).
  subroutine fdm_init(this, coef, bclst, json)
    class(fdm_pc_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json
    character(len=:), allocatable :: prec
    integer, allocatable :: grid(:)
    call this%free()
    this%coef => coef
    this%bclst => bclst
    this%nsolve = 0
    this%nupdate = 0
    this%warned = .false.
    this%tprof = 0.0_rp
    call json_get_or_default(json, 'verbose', this%verbose, .false.)
    call json_get_or_default(json, 'profile', this%profile, .false.)
    call json_get_or_default(json, 'residual', this%residual, .false.)
    call json_get_or_default(json, 'exchange_precision', prec, 'double')
    call json_get_or_default(json, 'chunks', this%nch, 1)
    call json_get_or_default(json, 'bloch', this%use_bloch, .true.)
    call json_get_or_default(json, 'fold', this%use_fold, .true.)
    if (this%nch .lt. 1 .or. this%nch .gt. 16) call neko_error('fdm: chunks must be 1 ... 16')
    this%Py = 1
    this%Pz = pe_size
    if (json%valid_path('rank_grid')) then
       call json_get(json, 'rank_grid', grid)
       if (size(grid) .ne. 2) call neko_error('fdm: rank_grid must be [Py, Pz]')
       this%Py = grid(1)
       this%Pz = grid(2)
       if (this%Py * this%Pz .ne. pe_size) call neko_error('fdm: rank_grid does not match the number of ranks')
    end if
    this%twod = this%Py .gt. 1
    this%single = .false.
    if (trim(prec) .eq. 'single') then
       this%single = .true.
    else if (trim(prec) .ne. 'double') then
       call neko_error('fdm: exchange_precision is "double" or "single"')
    end if
    if (this%single .and. pe_size .eq. 1) then
       call neko_log%message('fdm: exchange_precision "single" ignored on one rank (no exchanges)')
       this%single = .false.
    end if
    this%ready = .false.
  end subroutine fdm_init

  subroutine fdm_setup(this)
    class(fdm_pc_t), intent(inout), target :: this
    type(coef_t), pointer :: coef
    type(bc_list_t), pointer :: bclst
    character(len=LOG_SIZE) :: log_buf
    integer :: lx, n, nelv, d, i, e, p, ierr, nf, np, me, q, k, c, m
    integer(kind=8) :: ncol
    integer :: Py, Pz, jme, kme, jo, ko, ix, iy, iz, xl, yl, zl, r, j
    integer(kind=8) :: nfree, maxbuf, u8, izf, i8
    integer, allocatable :: xr(:), yr(:), zr(:), yr2(:), dest(:), slot(:), fill(:), gs2(:)
    integer(kind=8), allocatable :: hu2(:)
    real(kind=rp) :: t0
    integer, allocatable :: zs(:), cs(:), perm(:)
    integer(kind=8), allocatable :: hu(:), hr(:), ukey(:)
    real(kind=rp), allocatable :: edges(:,:)   ! (0:maxE, 3)
    real(kind=rp), allocatable :: grid(:,:)    ! (nfull max, 3)
    integer, allocatable :: idx(:,:)           ! (n, 3): 1-based node index per direction
    logical, allocatable :: flag(:)
    integer :: maxE, maxfull, cnt(2, 3, 2), bad, iu(3)
    real(kind=rp) :: tol(3)

    coef => this%coef
    bclst => this%bclst
    if (rp .ne. dp) call neko_error('fdm: needs a double precision build (rp = dp)')
    lx = coef%Xh%lx
    if (coef%Xh%ly .ne. lx .or. coef%Xh%lz .ne. lx) call neko_error('fdm: lx = ly = lz required')
    if (lx .lt. 2) call neko_error('fdm: polynomial order >= 1 required')
    n = coef%dof%size()
    nelv = coef%msh%nelv
    this%lx = lx
    this%n = n

    call neko_log%section('fdm preconditioner')
    t0 = MPI_Wtime()

    ! 1. element edges per direction (global), the 1D GLL grids
    call element_edges(coef, edges, this%ne, maxE)
    do d = 1, 3
       tol(d) = 1.0e-8_rp * (edges(this%ne(d), d) - edges(0, d))
       this%nfull(d) = this%ne(d) * (lx - 1) + 1
    end do
    maxfull = maxval(this%nfull)
    allocate(grid(maxfull, 3))
    grid = 0.0_rp
    do d = 1, 3
       do e = 0, this%ne(d) - 1
          do i = 1, lx
             grid(e * (lx - 1) + i, d) = edges(e, d) + &
                  0.5_rp * (edges(e + 1, d) - edges(e, d)) * (coef%Xh%zg(i, 1) + 1.0_rp)
          end do
       end do
    end do

    call phase_time(this, t0, 'edges')
    ! 2. each point's node indices
    allocate(idx(n, 3))
    call point_indices(coef%dof%x%x, coef%dof%y%x, coef%dof%z%x, n, grid, maxfull, this%nfull, tol, idx)

    call phase_time(this, t0, 'points')
    ! 3. periodic directions from the mesh's periodic facets
    call periodic_directions(coef, idx, lx, this%nfull, this%periodic)
    do d = 1, 3
       this%nu(d) = this%nfull(d)
       if (this%periodic(d)) this%nu(d) = this%nfull(d) - 1
    end do

    do d = 1, 3
       ! found by the review of 2026-10-13: with two elements in a periodic
       ! direction the solve is not Neko's inverse (residual ratios 0.05-1;
       ! Neko's periodic connectivity is degenerate there; not understood)
       if (this%periodic(d) .and. this%ne(d) .eq. 2) &
            call neko_error('fdm: a periodic direction with 2 elements is not supported')
    end do
    call phase_time(this, t0, 'periodic')
    ! 4. Dirichlet faces from the bc list's Dirichlet masks
    allocate(flag(n))
    flag = .false.
    do i = 1, bclst%size()
       call mark_dirichlet(bclst%get(i), flag, n, this%verbose)
    end do
    ! cnt(1, d, side): points on the face; cnt(2, d, side): flagged among them
    cnt = 0
    do p = 1, n
       do d = 1, 3
          if (this%periodic(d)) cycle
          if (idx(p, d) .eq. 1) then
             cnt(1, d, 1) = cnt(1, d, 1) + 1
             if (flag(p)) cnt(2, d, 1) = cnt(2, d, 1) + 1
          else if (idx(p, d) .eq. this%nfull(d)) then
             cnt(1, d, 2) = cnt(1, d, 2) + 1
             if (flag(p)) cnt(2, d, 2) = cnt(2, d, 2) + 1
          end if
       end do
    end do
    call MPI_Allreduce(MPI_IN_PLACE, cnt, 12, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    do d = 1, 3
       if (this%periodic(d)) cycle
       if (cnt(2, d, 1) .gt. 0 .and. cnt(2, d, 1) .lt. cnt(1, d, 1)) &
            call neko_error('fdm: a Dirichlet condition covers part of a box face (low end)')
       if (cnt(2, d, 2) .gt. 0 .and. cnt(2, d, 2) .lt. cnt(1, d, 2)) &
            call neko_error('fdm: a Dirichlet condition covers part of a box face (high end)')
       this%dir_lo(d) = cnt(2, d, 1) .gt. 0 .and. cnt(2, d, 1) .eq. cnt(1, d, 1)
       this%dir_hi(d) = cnt(2, d, 2) .gt. 0 .and. cnt(2, d, 2) .eq. cnt(1, d, 2)
    end do
    bad = 0
    do p = 1, n
       if (.not. flag(p)) cycle
       if (.not. on_dirichlet_face(this, idx(p, :))) bad = bad + 1
    end do
    call MPI_Allreduce(MPI_IN_PLACE, bad, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    if (bad .gt. 0) call neko_error('fdm: Dirichlet nodes that are not on a whole box face')

    call phase_time(this, t0, 'dirichlet')
    ! 5. the 1D operators and eigenbases
    do d = 1, 3
       this%lo(d) = 0
       if (.not. this%periodic(d) .and. this%dir_lo(d)) this%lo(d) = 1
       nf = this%nu(d)
       if (.not. this%periodic(d)) then
          if (this%dir_lo(d)) nf = nf - 1
          if (this%dir_hi(d)) nf = nf - 1
       end if
       if (nf .lt. 1) call neko_error('fdm: a direction has no free node')
       this%nf(d) = nf
    end do
    call setup_direction(this, 1, edges(0:this%ne(1), 1), this%Sx, this%lamx)
    call setup_direction(this, 2, edges(0:this%ne(2), 2), this%Sy, this%lamy)
    call setup_direction(this, 3, edges(0:this%ne(3), 3), this%Sz, this%lamz)
    ! the host path applies x and z from the right: Bloch bases there for
    ! uniform periodic directions
    if (NEKO_BCKND_DEVICE .eq. 0 .and. this%use_bloch) then
       call setup_bloch(this, 1, edges(0:this%ne(1), 1), this%lamx)
       call setup_bloch(this, 3, edges(0:this%ne(3), 3), this%lamz)
    end if
    if (NEKO_BCKND_DEVICE .eq. 0 .and. this%use_fold) then
       call setup_fold(this, 1, edges(0:this%ne(1), 1), this%Sx, this%lamx)
       call setup_fold(this, 2, edges(0:this%ne(2), 2), this%Sy, this%lamy)
       call setup_fold(this, 3, edges(0:this%ne(3), 3), this%Sz, this%lamz)
    end if

    call phase_time(this, t0, 'operators')
    ! 6. the distributed layout: z-slabs of the free grid for the x and y
    ! transforms, z-pencils (ranges of the (x, y) column index) for z
    nfree = int(this%nf(1), 8) * this%nf(2) * this%nf(3)
    ncol = int(this%nf(1), 8) * this%nf(2)
    if (ncol .ge. int(huge(1), 8)) call neko_error('fdm: more than 2^31 free nodes in a plane')
    np = pe_size
    me = pe_rank
    this%P = np
    this%me = me
    allocate(zs(0:np), cs(0:np))
    do q = 0, np
       zs(q) = int((int(this%nf(3), 8) * q) / np)
       cs(q) = int((ncol * q) / np)
    end do
    this%nzl = zs(me + 1) - zs(me)
    this%ncl = cs(me + 1) - cs(me)
    this%cs_me = cs(me)
    this%zs0 = zs(me)
    this%nslab = ncol * this%nzl
    this%npencil = int(this%ncl, 8) * this%nf(3)
    if (max(this%nslab, this%npencil, int(n, 8)) .ge. int(huge(1), 8)) &
         call neko_error('fdm: a rank holds more than 2^31 values (32-bit local indices)')
    ! the free nodes this rank holds, one representative point each, in
    ! grid order (x fastest, then y, then z: grouped by the slab owner):
    ! the points sorted by their 64-bit global free index, then unique; no
    ! array over the global grid
    allocate(this%sidx(n), ukey(n), perm(n))
    m = 0
    do p = 1, n
       this%sidx(p) = 0
       do d = 1, 3
          iu(d) = mod(idx(p, d) - 1, this%nu(d)) - this%lo(d)   ! 0-based free index
          if (iu(d) .lt. 0 .or. iu(d) .ge. this%nf(d)) iu(d) = -1
       end do
       if (any(iu .lt. 0)) cycle
       m = m + 1
       ukey(m) = 1_8 + iu(1) + int(this%nf(1), 8) * (iu(2) + int(this%nf(2), 8) * iu(3))
       perm(m) = p
    end do
    call radixsort_i8(ukey, perm, m)
    k = 0
    do i = 1, m
       if (i .eq. 1) then
          k = 1
       else if (ukey(i) .ne. ukey(k)) then
          k = k + 1
          ukey(k) = ukey(i)
          perm(k) = perm(i)
       end if
    end do
    this%nheld = k
    allocate(this%gsend(max(this%nheld, 1_8)), hu(max(this%nheld, 1_8)))
    this%gsend(1:k) = perm(1:k)
    hu(1:k) = ukey(1:k)
    deallocate(ukey, perm)
    ! sidx(p): the position of the point's node in the held list
    do p = 1, n
       do d = 1, 3
          iu(d) = mod(idx(p, d) - 1, this%nu(d)) - this%lo(d)
          if (iu(d) .lt. 0 .or. iu(d) .ge. this%nf(d)) iu(d) = -1
       end do
       if (any(iu .lt. 0)) cycle
       u8 = 1_8 + iu(1) + int(this%nf(1), 8) * (iu(2) + int(this%nf(2), 8) * iu(3))
       this%sidx(p) = bsearch_i8(hu, int(this%nheld), u8)
       if (this%sidx(p) .le. 0) call neko_error('fdm: a point is not in the held list')
    end do
    if (this%twod) then
       ! ---- the 2D scheme: rank me = j + Py k; layouts A (x whole, y in Y_j,
       ! z in Z_k), B (x in X_j, y whole, z in Z_k), C (x in X_j, y in Y'_k, z whole)
       Py = this%Py
       Pz = this%Pz
       jme = mod(me, Py)
       kme = me / Py
       this%jme = jme
       this%kme = kme
       allocate(xr(0:Py), yr(0:Py), zr(0:Pz), yr2(0:Pz))
       do q = 0, Py
          xr(q) = int((int(this%nf(1), 8) * q) / Py)
          yr(q) = int((int(this%nf(2), 8) * q) / Py)
       end do
       do q = 0, Pz
          zr(q) = int((int(this%nf(3), 8) * q) / Pz)
          yr2(q) = int((int(this%nf(2), 8) * q) / Pz)
       end do
       this%nyA = yr(jme + 1) - yr(jme)
       this%nzA = zr(kme + 1) - zr(kme)
       this%nxB = xr(jme + 1) - xr(jme)
       this%nyC = yr2(kme + 1) - yr2(kme)
       this%x0B = xr(jme)
       this%y0C = yr2(kme)
       this%nA = int(this%nf(1), 8) * this%nyA * this%nzA
       this%nB = int(this%nxB, 8) * this%nf(2) * this%nzA
       this%nC = int(this%nxB, 8) * this%nyC * this%nf(3)
       if (max(this%nA, this%nB, this%nC) .ge. int(huge(1), 8)) &
            call neko_error('fdm: a rank holds more than 2^31 values (32-bit local indices)')
       ! node exchange: a held node goes to the A-owner of its (y, z); the
       ! held list is regrouped by destination rank
       allocate(this%fs_cnt(np), this%fs_dsp(np), this%fr_cnt(np), this%fr_dsp(np))
       allocate(dest(max(this%nheld, 1_8)), slot(max(this%nheld, 1_8)), fill(np))
       this%fs_cnt = 0
       do k = 1, int(this%nheld)
          u8 = hu(k) - 1
          iy = int(mod(u8 / this%nf(1), int(this%nf(2), 8)))
          iz = int(u8 / (int(this%nf(1), 8) * this%nf(2)))
          jo = range_owner(iy, yr, Py)
          ko = range_owner(iz, zr, Pz)
          dest(k) = jo + Py * ko
          this%fs_cnt(dest(k) + 1) = this%fs_cnt(dest(k) + 1) + 1
       end do
       call MPI_Alltoall(this%fs_cnt, 1, MPI_INTEGER, this%fr_cnt, 1, MPI_INTEGER, NEKO_COMM, ierr)
       this%fs_dsp(1) = 0
       this%fr_dsp(1) = 0
       do q = 2, np
          this%fs_dsp(q) = this%fs_dsp(q - 1) + this%fs_cnt(q - 1)
          this%fr_dsp(q) = this%fr_dsp(q - 1) + this%fr_cnt(q - 1)
       end do
       fill = this%fs_dsp
       do k = 1, int(this%nheld)
          fill(dest(k) + 1) = fill(dest(k) + 1) + 1
          slot(k) = fill(dest(k) + 1)
       end do
       allocate(gs2(max(this%nheld, 1_8)), hu2(max(this%nheld, 1_8)))
       do k = 1, int(this%nheld)
          gs2(slot(k)) = this%gsend(k)
          hu2(slot(k)) = hu(k)
       end do
       this%gsend(1:this%nheld) = gs2(1:this%nheld)
       hu(1:this%nheld) = hu2(1:this%nheld)
       do p = 1, n
          if (this%sidx(p) .gt. 0) this%sidx(p) = slot(this%sidx(p))
       end do
       deallocate(dest, slot, fill, gs2, hu2)
       this%nrecv = sum(this%fr_cnt)
       allocate(hr(max(this%nrecv, 1_8)), this%rpos(max(this%nrecv, 1_8)))
       call MPI_Alltoallv(hu, this%fs_cnt, this%fs_dsp, MPI_INTEGER8, hr, this%fr_cnt, this%fr_dsp, &
            MPI_INTEGER8, NEKO_COMM, ierr)
       do k = 1, int(this%nrecv)
          u8 = hr(k) - 1
          ix = int(mod(u8, int(this%nf(1), 8)))
          iy = int(mod(u8 / this%nf(1), int(this%nf(2), 8)))
          iz = int(u8 / (int(this%nf(1), 8) * this%nf(2)))
          if (NEKO_BCKND_DEVICE .eq. 1) then
             this%rpos(k) = 1 + ix + this%nf(1) * ((iy - yr(jme)) + this%nyA * (iz - zr(kme)))
          else
             ! the host's layout A: y fastest in a z plane (x is applied from the right)
             this%rpos(k) = 1 + (iy - yr(jme)) + this%nyA * (ix + this%nf(1) * (iz - zr(kme)))
          end if
       end do
       ! the groups: comm_k (same k, Py ranks ordered by j), comm_j (same j, Pz ranks by k)
       call MPI_Comm_split(NEKO_COMM, kme, jme, this%comm_k, ierr)
       call MPI_Comm_split(NEKO_COMM, jme, kme, this%comm_j, ierr)
       ! A -> B inside the k-group: rank j' gets x in X_j' of every local (y, z)
       allocate(this%t1s_cnt(Py), this%t1s_dsp(Py), this%t1r_cnt(Py), this%t1r_dsp(Py))
       do q = 0, Py - 1
          this%t1s_cnt(q + 1) = (xr(q + 1) - xr(q)) * this%nyA * this%nzA
          this%t1r_cnt(q + 1) = this%nxB * (yr(q + 1) - yr(q)) * this%nzA
       end do
       this%t1s_dsp(1) = 0
       this%t1r_dsp(1) = 0
       do q = 2, Py
          this%t1s_dsp(q) = this%t1s_dsp(q - 1) + this%t1s_cnt(q - 1)
          this%t1r_dsp(q) = this%t1r_dsp(q - 1) + this%t1r_cnt(q - 1)
       end do
       allocate(this%posA(max(this%nA, 1_8)), this%tabB(max(this%nB, 1_8)))
       do i8 = 0, this%nA - 1
          if (NEKO_BCKND_DEVICE .eq. 1) then
             ix = int(mod(i8, int(this%nf(1), 8)))
             yl = int(mod(i8 / this%nf(1), int(this%nyA, 8)))
          else
             yl = int(mod(i8, int(this%nyA, 8)))
             ix = int(mod(i8 / this%nyA, int(this%nf(1), 8)))
          end if
          zl = int(i8 / (int(this%nf(1), 8) * this%nyA))
          q = range_owner(ix, xr, Py)
          this%posA(i8 + 1) = 1 + this%t1s_dsp(q + 1) + (ix - xr(q)) + (xr(q + 1) - xr(q)) * (yl + this%nyA * zl)
       end do
       r = 0
       do j = 0, Py - 1
          do zl = 0, this%nzA - 1
             do yl = 0, yr(j + 1) - yr(j) - 1
                do xl = 0, this%nxB - 1
                   r = r + 1
                   this%tabB(r) = 1 + xl + this%nxB * ((yr(j) + yl) + this%nf(2) * zl)
                end do
             end do
          end do
       end do
       if (int(r, 8) .ne. this%nB) call neko_error('fdm: tabB size mismatch')
       ! B -> C inside the j-group: rank k' gets y in Y'_k' of every local (x, z)
       allocate(this%t2s_cnt(Pz), this%t2s_dsp(Pz), this%t2r_cnt(Pz), this%t2r_dsp(Pz))
       do q = 0, Pz - 1
          this%t2s_cnt(q + 1) = this%nxB * (yr2(q + 1) - yr2(q)) * this%nzA
          this%t2r_cnt(q + 1) = this%nxB * this%nyC * (zr(q + 1) - zr(q))
       end do
       this%t2s_dsp(1) = 0
       this%t2r_dsp(1) = 0
       do q = 2, Pz
          this%t2s_dsp(q) = this%t2s_dsp(q - 1) + this%t2s_cnt(q - 1)
          this%t2r_dsp(q) = this%t2r_dsp(q - 1) + this%t2r_cnt(q - 1)
       end do
       allocate(this%posB(max(this%nB, 1_8)), this%tabC(max(this%nC, 1_8)))
       do i8 = 0, this%nB - 1
          xl = int(mod(i8, int(this%nxB, 8)))
          iy = int(mod(i8 / this%nxB, int(this%nf(2), 8)))
          zl = int(i8 / (int(this%nxB, 8) * this%nf(2)))
          q = range_owner(iy, yr2, Pz)
          this%posB(i8 + 1) = 1 + this%t2s_dsp(q + 1) + xl + this%nxB * ((iy - yr2(q)) + (yr2(q + 1) - yr2(q)) * zl)
       end do
       r = 0
       do k = 0, Pz - 1
          do zl = 0, zr(k + 1) - zr(k) - 1
             do yl = 0, this%nyC - 1
                do xl = 0, this%nxB - 1
                   r = r + 1
                   this%tabC(r) = 1 + xl + this%nxB * (yl + this%nyC * (zr(k) + zl))
                end do
             end do
          end do
       end do
       if (int(r, 8) .ne. this%nC) call neko_error('fdm: tabC size mismatch')
       ! the slab-scheme tables stay empty
       this%nzl = 0
       this%ncl = 0
       this%nslab = 0
       this%npencil = 0
       allocate(this%cdest(1), this%cstride(1))   ! unused by the 2D scheme
       this%cdest = 0
       this%cstride = 0
       allocate(this%ts_cnt(np), this%ts_dsp(np), this%tr_cnt(np), this%tr_dsp(np))
       this%ts_cnt = 0
       this%ts_dsp = 0
       this%tr_cnt = 0
       this%tr_dsp = 0
    else
       ! node exchange: a held node goes to the owner of its z plane
       allocate(this%fs_cnt(np), this%fs_dsp(np), this%fr_cnt(np), this%fr_dsp(np))
       this%fs_cnt = 0
       q = 0
       do k = 1, int(this%nheld)
          izf = (hu(k) - 1) / ncol
          do while (izf .ge. int(zs(q + 1), 8))
             q = q + 1
          end do
          this%fs_cnt(q + 1) = this%fs_cnt(q + 1) + 1
       end do
       call MPI_Alltoall(this%fs_cnt, 1, MPI_INTEGER, this%fr_cnt, 1, MPI_INTEGER, NEKO_COMM, ierr)
       this%fs_dsp(1) = 0
       this%fr_dsp(1) = 0
       do q = 2, np
          this%fs_dsp(q) = this%fs_dsp(q - 1) + this%fs_cnt(q - 1)
          this%fr_dsp(q) = this%fr_dsp(q - 1) + this%fr_cnt(q - 1)
       end do
       this%nrecv = sum(this%fr_cnt)
       allocate(hr(max(this%nrecv, 1_8)), this%rpos(max(this%nrecv, 1_8)))
       call MPI_Alltoallv(hu, this%fs_cnt, this%fs_dsp, MPI_INTEGER8, hr, this%fr_cnt, this%fr_dsp, &
            MPI_INTEGER8, NEKO_COMM, ierr)
       do k = 1, int(this%nrecv)
          u8 = hr(k) - 1
          if (NEKO_BCKND_DEVICE .eq. 1) then
             this%rpos(k) = int(1 + mod(u8, ncol) + ncol * (u8 / ncol - zs(me)))
          else
             ! the host slab: y fastest in a plane (x is applied from the right)
             ix = int(mod(u8, int(this%nf(1), 8)))
             iy = int(mod(u8 / this%nf(1), int(this%nf(2), 8)))
             this%rpos(k) = int(1 + iy + int(this%nf(2), 8) * ix + ncol * (u8 / ncol - zs(me)))
          end if
       end do
       ! the transpose: rank q gets the columns cs(q) .. cs(q+1)-1 of every
       ! local plane, packed plane by plane, which is the pencil layout
       allocate(this%ts_cnt(np), this%ts_dsp(np), this%tr_cnt(np), this%tr_dsp(np))
       allocate(this%cdest(ncol), this%cstride(ncol))
       do q = 0, np - 1
          this%ts_cnt(q + 1) = (cs(q + 1) - cs(q)) * this%nzl
          this%tr_cnt(q + 1) = this%ncl * (zs(q + 1) - zs(q))
          this%tr_dsp(q + 1) = this%ncl * zs(q)
       end do
       this%ts_dsp(1) = 0
       do q = 2, np
          this%ts_dsp(q) = this%ts_dsp(q - 1) + this%ts_cnt(q - 1)
       end do
       do q = 0, np - 1
          do c = cs(q), cs(q + 1) - 1
             this%cdest(c + 1) = this%ts_dsp(q + 1) + (c - cs(q))
             this%cstride(c + 1) = cs(q + 1) - cs(q)
          end do
       end do
       ! the chunks of planes of every rank and the per-chunk counts of the
       ! forward transpose (the way back swaps them)
       allocate(this%zc(0:this%nch, np))
       do q = 0, np - 1
          do c = 0, this%nch
             this%zc(c, q + 1) = ((zs(q + 1) - zs(q)) * c) / this%nch
          end do
       end do
       allocate(this%cs_cnt(np, this%nch), this%cs_dsp(np, this%nch), this%cr_cnt(np, this%nch), &
            this%cr_dsp(np, this%nch))
       do c = 1, this%nch
          do q = 0, np - 1
             this%cs_cnt(q + 1, c) = (cs(q + 1) - cs(q)) * (this%zc(c, me + 1) - this%zc(c - 1, me + 1))
             this%cs_dsp(q + 1, c) = this%ts_dsp(q + 1) + (cs(q + 1) - cs(q)) * this%zc(c - 1, me + 1)
             this%cr_cnt(q + 1, c) = this%ncl * (this%zc(c, q + 1) - this%zc(c - 1, q + 1))
             this%cr_dsp(q + 1, c) = this%ncl * (zs(q) + this%zc(c - 1, q + 1))
          end do
       end do

    end if
    call phase_time(this, t0, 'layout')
    if (NEKO_BCKND_DEVICE .eq. 1) then
       this%ctx = fdm_dev_setup(glb_cmd_queue, this%nf(1), this%nf(2), this%nf(3), this%nzl, this%ncl, &
            this%cs_me, n, this%nheld, this%nrecv, merge(1, 0, np .eq. 1), merge(1, 0, this%single), &
            this%Sx, this%Sy, this%Sz, this%lamx, this%lamy, this%lamz, this%gsend, this%rpos, this%sidx, &
            this%cdest, this%cstride)
       if (np .gt. 1 .and. .not. NEKO_DEVICE_MPI .and. this%single) &
            call neko_error('fdm: single-precision exchanges need device MPI')
       if (this%twod .and. .not. NEKO_DEVICE_MPI) call neko_error('fdm: rank_grid needs device MPI')
       if (this%twod) call fdm_dev2_setup(this%ctx, this%nA, this%nB, this%nC, this%nyA, this%nzA, this%nxB, &
            this%nyC, this%x0B, this%y0C, this%posA, this%tabB, this%posB, this%tabC)
       if (.not. this%twod) call fdm_dev_chunks(this%ctx, this%nch)
       ! the index tables live on the device from here on (the APU's host
       ! memory is the GPU's)
       deallocate(this%gsend, this%rpos, this%sidx, this%cdest, this%cstride)
       if (this%twod) deallocate(this%posA, this%tabB, this%posB, this%tabC)
       if (np .gt. 1 .and. .not. NEKO_DEVICE_MPI) then
          maxbuf = max(this%nheld, this%nrecv, this%nslab, this%npencil, 1_8)
          allocate(this%hs(maxbuf), this%hr(maxbuf))
       end if
    else
       call host_buffers(this)
    end if
    call phase_time(this, t0, 'device')

    write(log_buf, '(A,I0,A,I0,A,I0,A,I0)') 'Elements   : ', this%ne(1), ' x ', this%ne(2), ' x ', &
         this%ne(3), ', lx ', lx
    call neko_log%message(log_buf)
    write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A)') 'Free nodes : ', this%nf(1), ' x ', this%nf(2), ' x ', &
         this%nf(3), ' (', nfree, ')'
    call neko_log%message(log_buf)
    write(log_buf, '(A,I0,A,I0,A,I0)') 'Ranks      : ', np, ', this one holds ', this%nheld, &
         ' nodes, receives ', this%nrecv
    call neko_log%message(log_buf)
    if (this%twod) then
       write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A,I0,A,I0)') 'Layout     : rank grid ', this%Py, ' x ', this%Pz, &
            ', A ', this%nyA, ' x ', this%nzA, ' (y, z), B x ', this%nxB, ', C y ', this%nyC
    else
       write(log_buf, '(A,I0,A,I0,A,A)') 'Layout     : slab planes ', this%nzl, ', pencil columns ', this%ncl, &
            ', exchanges in ', merge('single', 'double', this%single)
    end if
    call neko_log%message(log_buf)
    do d = 1, 3
       write(log_buf, '(A,I0,A,L1,A,L1,A,L1,A,I0)') 'Direction ', d, ': periodic ', this%periodic(d), &
            ', Dirichlet low ', this%dir_lo(d), ' high ', this%dir_hi(d), ', zero mode ', this%zero_mode(d)
       call neko_log%message(log_buf)
    end do
    call neko_log%end_section()
    this%ready = .true.
  end subroutine fdm_setup

  !> Logs the wall time of a setup phase (verbose) and restarts the clock.
  subroutine phase_time(this, t0, what)
    class(fdm_pc_t), intent(in) :: this
    real(kind=rp), intent(inout) :: t0
    character(len=*), intent(in) :: what
    character(len=LOG_SIZE) :: log_buf
    if (this%verbose) then
       write(log_buf, '(A,A,A,F8.2,A)') 'setup phase ', what, ': ', MPI_Wtime() - t0, ' s'
       call neko_log%message(log_buf)
    end if
    t0 = MPI_Wtime()
  end subroutine phase_time

  !> The element edges per direction: the union over elements of their
  !! coordinate extents, merged over ranks.
  subroutine element_edges(coef, edges, ne, maxE)
    type(coef_t), intent(in) :: coef
    real(kind=rp), allocatable, intent(out) :: edges(:,:)
    integer, intent(out) :: ne(3), maxE
    real(kind=rp), allocatable :: cand(:), glob(:), per_dir(:,:), tmp(:,:)
    integer, allocatable :: counts(:), displs(:)
    integer :: lx, nelv, e, d, i, j, k, m, nc, ierr, nglob, p
    real(kind=rp) :: lo, hi, v, span, tol
    integer :: nd(3)

    lx = coef%Xh%lx
    nelv = coef%msh%nelv
    allocate(counts(pe_size), displs(pe_size))
    allocate(per_dir(2 * nelv + 2, 3))
    nd = 0
    do d = 1, 3
       allocate(cand(2 * nelv))
       do e = 1, nelv
          lo = huge(1.0_rp)
          hi = -huge(1.0_rp)
          do k = 1, lx
             do j = 1, lx
                do i = 1, lx
                   select case (d)
                   case (1)
                      v = coef%dof%x%x(i, j, k, e)
                   case (2)
                      v = coef%dof%y%x(i, j, k, e)
                   case default
                      v = coef%dof%z%x(i, j, k, e)
                   end select
                   lo = min(lo, v)
                   hi = max(hi, v)
                end do
             end do
          end do
          cand(2 * e - 1) = lo
          cand(2 * e) = hi
       end do
       span = maxval(cand) - minval(cand)
       call MPI_Allreduce(MPI_IN_PLACE, span, 1, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
       tol = 1.0e-8_rp * span
       call sort_unique(cand, 2 * nelv, tol, nc)
       call MPI_Allgather(nc, 1, MPI_INTEGER, counts, 1, MPI_INTEGER, NEKO_COMM, ierr)
       displs(1) = 0
       do p = 2, pe_size
          displs(p) = displs(p - 1) + counts(p - 1)
       end do
       nglob = sum(counts)
       allocate(glob(nglob))
       call MPI_Allgatherv(cand, nc, MPI_DOUBLE_PRECISION, glob, counts, displs, &
            MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
       call sort_unique(glob, nglob, tol, m)
       nd(d) = m
       if (m .gt. size(per_dir, 1)) then   ! keep the directions already stored
          allocate(tmp(m, 3))
          tmp = 0.0_rp
          tmp(1:size(per_dir, 1), :) = per_dir
          call move_alloc(tmp, per_dir)
       end if
       per_dir(1:m, d) = glob(1:m)
       deallocate(cand, glob)
    end do
    ne = nd - 1
    maxE = maxval(ne)
    allocate(edges(0:maxE, 3))
    edges = 0.0_rp
    do d = 1, 3
       edges(0:ne(d), d) = per_dir(1:nd(d), d)
    end do
  end subroutine element_edges

  !> Sorts a(1:n) and keeps one of each group of values within tol: m values.
  subroutine sort_unique(a, n, tol, m)
    real(kind=rp), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(kind=rp), intent(in) :: tol
    integer, intent(out) :: m
    integer :: i
    call heapsort(a, n)
    m = 0
    do i = 1, n
       if (m .eq. 0) then
          m = 1
          a(1) = a(i)
       else if (a(i) - a(m) .gt. tol) then
          m = m + 1
          a(m) = a(i)
       end if
    end do
  end subroutine sort_unique

  subroutine heapsort(a, n)
    real(kind=rp), intent(inout) :: a(:)
    integer, intent(in) :: n
    integer :: i
    real(kind=rp) :: t
    if (n .lt. 2) return
    do i = n / 2, 1, -1
       call sift(a, i, n)
    end do
    do i = n, 2, -1
       t = a(1)
       a(1) = a(i)
       a(i) = t
       call sift(a, 1, i - 1)
    end do
  contains
    subroutine sift(a, start, end)
      real(kind=rp), intent(inout) :: a(:)
      integer, intent(in) :: start, end
      integer :: root, child
      real(kind=rp) :: tmp
      root = start
      do while (2 * root .le. end)
         child = 2 * root
         if (child + 1 .le. end) then
            if (a(child + 1) .gt. a(child)) child = child + 1
         end if
         if (a(root) .lt. a(child)) then
            tmp = a(root)
            a(root) = a(child)
            a(child) = tmp
            root = child
         else
            return
         end if
      end do
    end subroutine sift
  end subroutine heapsort

  !> The range q (0-based) of the partition r(0:P) that contains i.
  pure function range_owner(i, r, P) result(q)
    integer, intent(in) :: i, P
    integer, intent(in) :: r(0:P)
    integer :: q
    q = 0
    do while (q .lt. P - 1 .and. i .ge. r(q + 1))
       q = q + 1
    end do
  end function range_owner

  !> Sorts the 64-bit keys key(1:n) ascending, carrying idx along: an LSD
  !! radix sort with 16-bit digits over the bits the largest key uses (stable,
  !! O(n) per pass; the heapsort below took 9 of the 15 s of setup on 9 M
  !! points).
  subroutine radixsort_i8(key, idx, n)
    integer(kind=8), intent(inout) :: key(:)
    integer, intent(inout) :: idx(:)
    integer, intent(in) :: n
    integer(kind=8), allocatable :: k2(:)
    integer, allocatable :: i2(:)
    integer(kind=8) :: kmax, cnt(0:65535), pos
    integer :: i, pass, npass, d
    if (n .lt. 2) return
    kmax = maxval(key(1:n))
    if (kmax .lt. 0) call neko_error('fdm: negative sort key')
    npass = 1
    do while (ishft(kmax, -16 * npass) .gt. 0)
       npass = npass + 1
    end do
    allocate(k2(n), i2(n))
    do pass = 0, npass - 1
       cnt = 0
       do i = 1, n
          d = int(iand(ishft(key(i), -16 * pass), 65535_8))
          cnt(d) = cnt(d) + 1
       end do
       pos = 0
       do d = 0, 65535
          kmax = cnt(d)
          cnt(d) = pos
          pos = pos + kmax
       end do
       do i = 1, n
          d = int(iand(ishft(key(i), -16 * pass), 65535_8))
          cnt(d) = cnt(d) + 1
          k2(cnt(d)) = key(i)
          i2(cnt(d)) = idx(i)
       end do
       key(1:n) = k2
       idx(1:n) = i2
    end do
    deallocate(k2, i2)
  end subroutine radixsort_i8

  !> Sorts the 64-bit keys key(1:n) ascending, carrying idx along (heapsort;
  !! kept as the reference of the radix sort).
  subroutine heapsort_i8(key, idx, n)
    integer(kind=8), intent(inout) :: key(:)
    integer, intent(inout) :: idx(:)
    integer, intent(in) :: n
    integer :: i
    if (n .lt. 2) return
    do i = n / 2, 1, -1
       call sift8(i, n)
    end do
    do i = n, 2, -1
       call swap8(1, i)
       call sift8(1, i - 1)
    end do
  contains
    subroutine swap8(a, b)
      integer, intent(in) :: a, b
      integer(kind=8) :: tk
      integer :: ti
      tk = key(a); key(a) = key(b); key(b) = tk
      ti = idx(a); idx(a) = idx(b); idx(b) = ti
    end subroutine swap8
    subroutine sift8(start, end)
      integer, intent(in) :: start, end
      integer :: root, child
      root = start
      do while (2 * root .le. end)
         child = 2 * root
         if (child + 1 .le. end) then
            if (key(child + 1) .gt. key(child)) child = child + 1
         end if
         if (key(root) .lt. key(child)) then
            call swap8(root, child)
            root = child
         else
            return
         end if
      end do
    end subroutine sift8
  end subroutine heapsort_i8

  !> Position of v in the sorted keys key(1:n), 0 if absent.
  pure function bsearch_i8(key, n, v) result(pos)
    integer(kind=8), intent(in) :: key(:), v
    integer, intent(in) :: n
    integer :: pos, lo, hi, mid
    pos = 0
    lo = 1
    hi = n
    do while (lo .le. hi)
       mid = (lo + hi) / 2
       if (key(mid) .eq. v) then
          pos = mid
          return
       else if (key(mid) .lt. v) then
          lo = mid + 1
       else
          hi = mid - 1
       end if
    end do
  end function bsearch_i8

  !> The 1D node index of every point per direction (the coordinate arrays
  !! are passed in sequence association as vectors of length n).
  subroutine point_indices(xs, ys, zs, n, grid, maxfull, nfull, tol, idx)
    integer, intent(in) :: n, maxfull, nfull(3)
    real(kind=rp), intent(in) :: xs(n), ys(n), zs(n), grid(maxfull, 3), tol(3)
    integer, intent(out) :: idx(n, 3)
    integer :: p, d
    real(kind=rp) :: x(3)
    character(len=LOG_SIZE) :: log_buf
    do p = 1, n
       x(1) = xs(p)
       x(2) = ys(p)
       x(3) = zs(p)
       do d = 1, 3
          idx(p, d) = nearest_index(grid(1:nfull(d), d), x(d))
          if (abs(grid(idx(p, d), d) - x(d)) .gt. tol(d)) then
             write(log_buf, '(A,I0,A,I0,A,ES12.4)') 'fdm: point ', p, &
                  ' is off the 1D GLL grid of direction ', d, ' by ', abs(grid(idx(p, d), d) - x(d))
             call neko_error(trim(log_buf))
          end if
       end do
    end do
  end subroutine point_indices

  !> Index of the entry of the sorted grid nearest to x.
  pure function nearest_index(grid, x) result(i)
    real(kind=rp), intent(in) :: grid(:), x
    integer :: i, lo, hi, mid
    lo = 1
    hi = size(grid)
    do while (hi - lo .gt. 1)
       mid = (lo + hi) / 2
       if (grid(mid) .le. x) then
          lo = mid
       else
          hi = mid
       end if
    end do
    if (abs(grid(hi) - x) .lt. abs(grid(lo) - x)) then
       i = hi
    else
       i = lo
    end if
  end function nearest_index

  !> periodic(d) from the mesh's periodic facets: a facet is normal to the
  !! direction whose node index is constant on it, and must lie on the box's
  !! boundary.
  subroutine periodic_directions(coef, idx, lx, nfull, periodic)
    type(coef_t), intent(in) :: coef
    integer, intent(in) :: idx(:,:), lx, nfull(3)
    logical, intent(out) :: periodic(3)
    integer :: m, f, e, i, j, k, d, p, lo(3), hi(3), flags(3), ierr, i0, i1, j0, j1, k0, k1

    flags = 0
    do m = 1, coef%msh%periodic%size
       f = coef%msh%periodic%facet_el(m)%x(1)
       e = coef%msh%periodic%facet_el(m)%x(2)
       i0 = 1; i1 = lx; j0 = 1; j1 = lx; k0 = 1; k1 = lx
       select case (f)
       case (1)
          i1 = 1
       case (2)
          i0 = lx
       case (3)
          j1 = 1
       case (4)
          j0 = lx
       case (5)
          k1 = 1
       case default
          k0 = lx
       end select
       lo = huge(1)
       hi = -huge(1)
       do k = k0, k1
          do j = j0, j1
             do i = i0, i1
                p = i + lx * ((j - 1) + lx * ((k - 1) + lx * (e - 1)))
                do d = 1, 3
                   lo(d) = min(lo(d), idx(p, d))
                   hi(d) = max(hi(d), idx(p, d))
                end do
             end do
          end do
       end do
       d = 0
       do p = 1, 3
          if (lo(p) .eq. hi(p)) d = p
       end do
       if (d .eq. 0) call neko_error('fdm: a periodic facet is not aligned with an axis')
       if (lo(d) .ne. 1 .and. lo(d) .ne. nfull(d)) &
            call neko_error('fdm: a periodic facet is not on the box boundary')
       flags(d) = 1
    end do
    call MPI_Allreduce(MPI_IN_PLACE, flags, 3, MPI_INTEGER, MPI_MAX, NEKO_COMM, ierr)
    periodic = flags .eq. 1
  end subroutine periodic_directions

  !> Flags the nodes of a Dirichlet condition; other types are ignored (mixed
  !! conditions are refused).
  subroutine mark_dirichlet(bc, flag, n, verbose)
    class(bc_t), intent(in) :: bc
    logical, intent(inout) :: flag(:)
    integer, intent(in) :: n
    logical, intent(in) :: verbose
    character(len=LOG_SIZE) :: log_buf
    integer :: m, i, cnt
    cnt = 0
    if (allocated(bc%msk)) cnt = bc%msk(0)
    if (verbose) then
       write(log_buf, '(A,I0,A,I0,A)') 'bc type ', bc%bc_type, ' with ', cnt, ' masked nodes'
       call neko_log%message(log_buf)
    end if
    if (cnt .eq. 0) return
    if (bc%bc_type .eq. BC_DIRICHLET) then
       do i = 1, cnt
          m = bc%msk(i)
          if (m .ge. 1 .and. m .le. n) flag(m) = .true.
       end do
    else if (bc%bc_type .ne. BC_NEUMANN) then
       write(log_buf, '(A,I0,A)') 'fdm: a condition of type ', bc%bc_type, &
            ' (not Dirichlet or Neumann) is not supported'
       call neko_error(trim(log_buf))
    end if
  end subroutine mark_dirichlet

  pure function on_dirichlet_face(this, ix) result(yes)
    class(fdm_pc_t), intent(in) :: this
    integer, intent(in) :: ix(3)
    logical :: yes
    integer :: d
    yes = .false.
    do d = 1, 3
       if (this%periodic(d)) cycle
       if (this%dir_lo(d) .and. ix(d) .eq. 1) yes = .true.
       if (this%dir_hi(d) .and. ix(d) .eq. this%nfull(d)) yes = .true.
    end do
  end function on_dirichlet_face

  !> The assembled 1D stiffness and mass of direction d on its free nodes and
  !! their B-orthonormal eigenbasis: A S = B S Lambda, S^T B S = I.
  subroutine setup_direction(this, d, edges, S, lam)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: d
    real(kind=rp), intent(in) :: edges(0:)
    real(kind=rp), allocatable, intent(out) :: S(:,:), lam(:)
    real(kind=rp), allocatable :: K(:,:), A(:,:), B(:), At(:,:), work(:), Bf(:)
    integer, allocatable :: free(:)
    integer :: lx, N, E, nu, nf, e_, i, j, gi, gj, info, lwork, m
    real(kind=rp) :: h, lmax
    character(len=LOG_SIZE) :: log_buf

    lx = this%lx
    N = lx - 1
    E = this%ne(d)
    nu = this%nu(d)
    nf = this%nf(d)
    call reference_stiffness(this%coef%Xh%zg(:, 1), this%coef%Xh%wx, lx, K)
    allocate(A(nu, nu), B(nu))
    A = 0.0_rp
    B = 0.0_rp
    do e_ = 0, E - 1
       h = edges(e_ + 1) - edges(e_)
       do j = 1, lx
          gj = mod(e_ * N + j - 1, nu) + 1
          B(gj) = B(gj) + this%coef%Xh%wx(j) * h / 2.0_rp
          do i = 1, lx
             gi = mod(e_ * N + i - 1, nu) + 1
             A(gi, gj) = A(gi, gj) + (2.0_rp / h) * K(i, j)
          end do
       end do
    end do
    allocate(free(nf))
    m = 0
    do i = 1, nu
       if (.not. this%periodic(d)) then
          if (i .eq. 1 .and. this%dir_lo(d)) cycle
          if (i .eq. nu .and. this%dir_hi(d)) cycle
       end if
       m = m + 1
       free(m) = i
    end do
    allocate(At(nf, nf), Bf(nf), lam(nf), S(nf, nf))
    do j = 1, nf
       Bf(j) = B(free(j))
       do i = 1, nf
          At(i, j) = A(free(i), free(j)) / sqrt(B(free(i)) * B(free(j)))
       end do
    end do
    lwork = max(1, 3 * nf - 1) * 4
    allocate(work(lwork))
    call dsyev('V', 'U', nf, At, nf, lam, work, lwork, info)
    if (info .ne. 0) then
       write(log_buf, '(A,I0)') 'fdm: dsyev failed, info ', info
       call neko_error(trim(log_buf))
    end if
    do j = 1, nf
       do i = 1, nf
          S(i, j) = At(i, j) / sqrt(Bf(i))
       end do
    end do
    this%zero_mode(d) = 0
    if (this%periodic(d) .or. .not. (this%dir_lo(d) .or. this%dir_hi(d))) then
       lmax = maxval(abs(lam))
       if (abs(lam(1)) .gt. 1.0e-9_rp * lmax + 1.0e-14_rp) then
          write(log_buf, '(A,I0,A,ES12.4)') 'fdm: direction ', d, &
               ' should have a zero eigenvalue, smallest is ', lam(1)
          call neko_error(trim(log_buf))
       end if
       lam(1) = 0.0_rp
       this%zero_mode(d) = 1
    end if
  end subroutine setup_direction

  !> The Bloch basis of direction d (bloch_t) when it is periodic with equal
  !! elements; lam is replaced by the eigenvalues in the Bloch order (the
  !! zero mode stays first). Checked against the dense eigenvalues.
  subroutine setup_bloch(this, d, edges, lam)
    class(fdm_pc_t), intent(inout), target :: this
    integer, intent(in) :: d
    real(kind=rp), intent(in) :: edges(0:)
    real(kind=rp), intent(inout) :: lam(:)
    real(kind=rp), allocatable :: K(:,:), A(:,:), B(:), T(:,:), W(:,:), At(:,:), Ab(:,:), work(:), lb(:), ls(:)
    integer :: lx, N, E, nf, e_, i, j, gi, gj, q, m, ib, o, sz, info, lwork, row, kk
    real(kind=rp) :: h0, pi, leak, amax, err, lmax
    character(len=LOG_SIZE) :: log_buf
    type(bloch_t), pointer :: bl
    bl => this%bl(d)
    bl%on = .false.
    if (.not. this%periodic(d)) return
    E = this%ne(d)
    lx = this%lx
    N = lx - 1
    nf = E * N
    if (nf .ne. this%nf(d)) return
    h0 = edges(1) - edges(0)
    do e_ = 1, E - 1
       if (abs((edges(e_ + 1) - edges(e_)) - h0) .gt. 1.0e-10_rp * h0) then
          write(log_buf, '(A,I0,A)') 'Direction ', d, ': periodic with unequal elements, dense basis'
          call neko_log%message(log_buf)
          return
       end if
    end do
    call reference_stiffness(this%coef%Xh%zg(:, 1), this%coef%Xh%wx, lx, K)
    allocate(A(nf, nf), B(nf))
    A = 0.0_rp
    B = 0.0_rp
    do e_ = 0, E - 1
       do j = 1, lx
          gj = mod(e_ * N + j - 1, nf) + 1
          B(gj) = B(gj) + this%coef%Xh%wx(j) * h0 / 2.0_rp
          do i = 1, lx
             gi = mod(e_ * N + i - 1, nf) + 1
             A(gi, gj) = A(gi, gj) + (2.0_rp / h0) * K(i, j)
          end do
       end do
    end do
    ! C: rows k = 0, (E/2 for even E), then the (cos k, sin k) pairs
    pi = acos(-1.0_rp)
    allocate(bl%C(E, E))
    bl%C(1, :) = 1.0_rp / sqrt(real(E, rp))
    row = 1
    if (mod(E, 2) .eq. 0) then
       row = 2
       do e_ = 0, E - 1
          bl%C(2, e_ + 1) = real(1 - 2 * mod(e_, 2), rp) / sqrt(real(E, rp))
       end do
    end if
    do kk = 1, (E - 1) / 2
       do e_ = 0, E - 1
          bl%C(row + 1, e_ + 1) = sqrt(2.0_rp / E) * cos(2.0_rp * pi * kk * e_ / E)
          bl%C(row + 2, e_ + 1) = sqrt(2.0_rp / E) * sin(2.0_rp * pi * kk * e_ / E)
       end do
       row = row + 2
    end do
    ! T = C x I_N (node m + N e -> transformed index m + N q); At = T A T^T
    allocate(T(nf, nf), W(nf, nf), At(nf, nf))
    T = 0.0_rp
    do q = 0, E - 1
       do e_ = 0, E - 1
          do m = 0, N - 1
             T(m + N * q + 1, m + N * e_ + 1) = bl%C(q + 1, e_ + 1)
          end do
       end do
    end do
    call dgemm('N', 'N', nf, nf, nf, 1.0_rp, T, nf, A, nf, 0.0_rp, W, nf)
    call dgemm('N', 'T', nf, nf, nf, 1.0_rp, W, nf, T, nf, 0.0_rp, At, nf)
    ! the blocks: k = 0 (with E/2): N or 2N columns, then 2N per pair
    bl%E = E
    bl%N = N
    bl%nb = 1 + (E - 1) / 2
    allocate(bl%off(bl%nb), bl%sz(bl%nb), bl%V(2 * N, 2 * N, bl%nb))
    bl%V = 0.0_rp
    bl%off(1) = 0
    bl%sz(1) = merge(2 * N, N, mod(E, 2) .eq. 0)
    do ib = 2, bl%nb
       bl%off(ib) = bl%off(ib - 1) + bl%sz(ib - 1)
       bl%sz(ib) = 2 * N
    end do
    ! the coupling between blocks must vanish (it is round-off)
    amax = maxval(abs(At))
    leak = 0.0_rp
    do ib = 1, bl%nb
       o = bl%off(ib)
       sz = bl%sz(ib)
       do j = o + 1, o + sz
          do i = 1, nf
             if (i .le. o .or. i .gt. o + sz) leak = max(leak, abs(At(i, j)))
          end do
       end do
    end do
    if (leak .gt. 1.0e-10_rp * amax) then
       write(log_buf, '(A,I0,A,ES10.2,A)') 'fdm: direction ', d, ': Bloch blocks couple (', leak / amax, &
            '), dense basis kept'
       call neko_log%message(log_buf)
       deallocate(bl%C, bl%V, bl%off, bl%sz)
       return
    end if
    ! the B-orthonormal eigenbasis of every block (the transformed mass is
    ! B again: B depends on m only)
    lwork = max(1, 6 * N - 1) * 4
    allocate(work(lwork), Ab(2 * N, 2 * N), lb(2 * N), ls(nf))
    do ib = 1, bl%nb
       o = bl%off(ib)
       sz = bl%sz(ib)
       do j = 1, sz
          do i = 1, sz
             Ab(i, j) = At(o + i, o + j) / sqrt(B(mod(o + i - 1, N) + 1) * B(mod(o + j - 1, N) + 1))
          end do
       end do
       call dsyev('V', 'U', sz, Ab, 2 * N, lb, work, lwork, info)
       if (info .ne. 0) call neko_error('fdm: dsyev failed in a Bloch block')
       do j = 1, sz
          do i = 1, sz
             bl%V(i, j, ib) = Ab(i, j) / sqrt(B(mod(o + i - 1, N) + 1))
          end do
       end do
       ls(o + 1:o + sz) = lb(1:sz)
    end do
    ! the same spectrum as the dense basis (lam sorted ascending there)
    lmax = maxval(abs(lam))
    lb = 0.0_rp
    W(:, 1) = ls
    call heapsort(W(:, 1), nf)
    err = maxval(abs(W(:, 1) - lam)) / lmax
    if (err .gt. 1.0e-9_rp) then
       write(log_buf, '(A,I0,A,ES10.2)') 'fdm: direction ', d, ': Bloch eigenvalues differ from the dense ones by ', err
       call neko_error(trim(log_buf))
    end if
    if (abs(ls(1)) .gt. 1.0e-9_rp * lmax + 1.0e-14_rp) call neko_error('fdm: the Bloch zero mode is not first')
    ls(1) = 0.0_rp
    lam = ls
    bl%on = .true.
    write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A)') 'Direction ', d, ': Bloch basis (', E, ' elements, ', &
         bl%nb, ' blocks, ', E + 2 * N, ' MACs per point at most)'
    call neko_log%message(log_buf)
  end subroutine setup_bloch

  !> y = x S (back = .false.) or y = x S^T (back) for the Bloch basis bl of
  !! a direction and x an M x (E N) matrix (column-major, the direction's
  !! nodes as columns); w: workspace of M E N.
  subroutine bloch_right(bl, M, x, y, w, back)
    type(bloch_t), intent(in) :: bl
    integer, intent(in) :: M
    real(kind=rp), intent(in) :: x(*)
    real(kind=rp), intent(inout) :: y(*), w(*)
    logical, intent(in) :: back
    integer :: MN, b
    integer(kind=8) :: o
    if (M .eq. 0) return
    MN = M * bl%N
    if (.not. back) then
       call dgemm('N', 'T', MN, bl%E, bl%E, 1.0_rp, x, MN, bl%C, bl%E, 0.0_rp, w, MN)
       do b = 1, bl%nb
          o = int(M, 8) * bl%off(b)
          call dgemm('N', 'N', M, bl%sz(b), bl%sz(b), 1.0_rp, w(o + 1), M, bl%V(1, 1, b), 2 * bl%N, &
               0.0_rp, y(o + 1), M)
       end do
    else
       do b = 1, bl%nb
          o = int(M, 8) * bl%off(b)
          call dgemm('N', 'T', M, bl%sz(b), bl%sz(b), 1.0_rp, x(o + 1), M, bl%V(1, 1, b), 2 * bl%N, &
               0.0_rp, w(o + 1), M)
       end do
       call dgemm('N', 'N', MN, bl%E, bl%E, 1.0_rp, w, MN, bl%C, bl%E, 0.0_rp, y, MN)
    end if
  end subroutine bloch_right

  !> The even/odd basis of a non-periodic direction whose element edges and
  !! end conditions are mirror-symmetric (host path): its eigenvectors are
  !! even or odd about the middle, so S = F^T blockdiag(Ve, Vo) with F the
  !! orthogonal fold (a(i) +- a(n-1-i)) / sqrt(2) (the middle node of an odd
  !! n goes to the even part): n/2 + 1 instead of n MACs per point and pass.
  !! Built from the dense basis (each column classified by its parity); lam
  !! is reordered (even modes first: the zero mode stays first).
  subroutine setup_fold(this, d, edges, S, lam)
    class(fdm_pc_t), intent(inout), target :: this
    integer, intent(in) :: d
    real(kind=rp), intent(in) :: edges(0:)
    real(kind=rp), intent(in) :: S(:,:)
    real(kind=rp), intent(inout) :: lam(:)
    type(fold_t), pointer :: fd
    real(kind=rp), allocatable :: ls(:)
    real(kind=rp) :: span, se, so, r2
    integer :: E, n, h, i, j, ke, ko
    character(len=LOG_SIZE) :: log_buf
    fd => this%fd(d)
    fd%on = .false.
    if (this%periodic(d) .or. (this%dir_lo(d) .neqv. this%dir_hi(d))) return
    E = this%ne(d)
    span = edges(E) - edges(0)
    do i = 0, E
       if (abs((edges(i) - edges(0)) - (edges(E) - edges(E - i))) .gt. 1.0e-10_rp * span) return
    end do
    n = this%nf(d)
    if (n .lt. 4) return
    h = n / 2
    fd%n = n
    fd%ne = n - h
    fd%no = h
    allocate(fd%Ve(fd%ne, fd%ne), fd%Vo(max(fd%no, 1), max(fd%no, 1)), ls(n))
    r2 = sqrt(2.0_rp)
    ke = 0
    ko = 0
    do j = 1, n
       se = 0.0_rp
       so = 0.0_rp
       do i = 1, n
          se = se + (S(i, j) - S(n + 1 - i, j))**2   ! zero for an even column
          so = so + (S(i, j) + S(n + 1 - i, j))**2   ! zero for an odd one
       end do
       if (se .le. 1.0e-20_rp * so) then
          ke = ke + 1
          if (ke .gt. fd%ne) exit
          do i = 1, h
             fd%Ve(i, ke) = r2 * S(i, j)
          end do
          if (fd%ne .gt. h) fd%Ve(fd%ne, ke) = S(h + 1, j)
          ls(ke) = lam(j)
       else if (so .le. 1.0e-20_rp * se) then
          ko = ko + 1
          if (ko .gt. fd%no) exit
          do i = 1, h
             fd%Vo(i, ko) = r2 * S(i, j)
          end do
          ls(fd%ne + ko) = lam(j)
       else
          exit
       end if
    end do
    if (ke .ne. fd%ne .or. ko .ne. fd%no) then
       write(log_buf, '(A,I0,A)') 'fdm: direction ', d, ': no clean even/odd split, dense basis kept'
       call neko_log%message(log_buf)
       deallocate(fd%Ve, fd%Vo)
       return
    end if
    lam = ls
    fd%on = .true.
    write(log_buf, '(A,I0,A,I0,A,I0,A)') 'Direction ', d, ': even/odd basis (', fd%ne, ' + ', fd%no, ' modes)'
    call neko_log%message(log_buf)
  end subroutine setup_fold

  !> y = S^T x (back = .false.) or y = S x (back) for the even/odd basis fd
  !! and x an n x R matrix (the direction fastest); w: workspace of n R.
  subroutine fold_left(fd, R, x, y, w, back)
    type(fold_t), intent(in) :: fd
    integer, intent(in) :: R
    real(kind=rp), intent(in) :: x(fd%n, *)
    real(kind=rp), intent(inout) :: y(fd%n, *), w(fd%n, *)
    logical, intent(in) :: back
    integer :: i, j, n, h, ne
    real(kind=rp) :: c
    if (R .eq. 0) return
    n = fd%n
    h = fd%no
    ne = fd%ne
    c = 1.0_rp / sqrt(2.0_rp)
    if (.not. back) then
       do j = 1, R
          do i = 1, h
             w(i, j) = c * (x(i, j) + x(n + 1 - i, j))
             w(ne + i, j) = c * (x(i, j) - x(n + 1 - i, j))
          end do
          if (ne .gt. h) w(ne, j) = x(h + 1, j)
       end do
       call dgemm('T', 'N', ne, R, ne, 1.0_rp, fd%Ve, ne, w, n, 0.0_rp, y, n)
       call dgemm('T', 'N', h, R, h, 1.0_rp, fd%Vo, h, w(ne + 1, 1), n, 0.0_rp, y(ne + 1, 1), n)
    else
       call dgemm('N', 'N', ne, R, ne, 1.0_rp, fd%Ve, ne, x, n, 0.0_rp, w, n)
       call dgemm('N', 'N', h, R, h, 1.0_rp, fd%Vo, h, x(ne + 1, 1), n, 0.0_rp, w(ne + 1, 1), n)
       do j = 1, R
          do i = 1, h
             y(i, j) = c * (w(i, j) + w(ne + i, j))
             y(n + 1 - i, j) = c * (w(i, j) - w(ne + i, j))
          end do
          if (ne .gt. h) y(h + 1, j) = w(ne, j)
       end do
    end if
  end subroutine fold_left

  !> y = x S (back = .false.) or y = x S^T (back) for the even/odd basis fd
  !! and x an M x n matrix (the direction's nodes as columns); w: M n.
  subroutine fold_right(fd, M, x, y, w, back)
    type(fold_t), intent(in) :: fd
    integer, intent(in) :: M
    real(kind=rp), intent(in) :: x(M, *)
    real(kind=rp), intent(inout) :: y(M, *), w(M, *)
    logical, intent(in) :: back
    integer :: i, k, n, h, ne
    real(kind=rp) :: c
    if (M .eq. 0) return
    n = fd%n
    h = fd%no
    ne = fd%ne
    c = 1.0_rp / sqrt(2.0_rp)
    if (.not. back) then
       do k = 1, h
          do i = 1, M
             w(i, k) = c * (x(i, k) + x(i, n + 1 - k))
             w(i, ne + k) = c * (x(i, k) - x(i, n + 1 - k))
          end do
       end do
       if (ne .gt. h) w(1:M, ne) = x(1:M, h + 1)
       call dgemm('N', 'N', M, ne, ne, 1.0_rp, w, M, fd%Ve, ne, 0.0_rp, y, M)
       call dgemm('N', 'N', M, h, h, 1.0_rp, w(1, ne + 1), M, fd%Vo, h, 0.0_rp, y(1, ne + 1), M)
    else
       call dgemm('N', 'T', M, ne, ne, 1.0_rp, x, M, fd%Ve, ne, 0.0_rp, w, M)
       call dgemm('N', 'T', M, h, h, 1.0_rp, x(1, ne + 1), M, fd%Vo, h, 0.0_rp, w(1, ne + 1), M)
       do k = 1, h
          do i = 1, M
             y(i, k) = c * (w(i, k) + w(i, ne + k))
             y(i, n + 1 - k) = c * (w(i, k) - w(i, ne + k))
          end do
       end do
       if (ne .gt. h) y(1:M, h + 1) = w(1:M, ne)
    end if
  end subroutine fold_right

  !> y = x S_d (back = .false.) or x S_d^T (back) for x an M x nf(d) matrix
  !! (direction d's nodes as columns): Bloch, even/odd or dense basis.
  subroutine dir_right(this, d, M, x, y, back)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: d, M
    real(kind=rp), intent(in) :: x(*)
    real(kind=rp), intent(inout) :: y(*)
    logical, intent(in) :: back
    integer :: n
    if (M .eq. 0) return
    n = this%nf(d)
    if (this%bl(d)%on) then
       call bloch_right(this%bl(d), M, x, y, this%hwk, back)
    else if (this%fd(d)%on) then
       call fold_right(this%fd(d), M, x, y, this%hwk, back)
    else
       select case (d)
       case (1)
          call dgemm('N', merge('T', 'N', back), M, n, n, 1.0_rp, x, M, this%Sx, n, 0.0_rp, y, M)
       case (2)
          call dgemm('N', merge('T', 'N', back), M, n, n, 1.0_rp, x, M, this%Sy, n, 0.0_rp, y, M)
       case default
          call dgemm('N', merge('T', 'N', back), M, n, n, 1.0_rp, x, M, this%Sz, n, 0.0_rp, y, M)
       end select
    end if
  end subroutine dir_right

  !> y = S_y^T x (back = .false.) or S_y x (back) for x an nf(2) x R matrix
  !! (y fastest): even/odd or dense basis.
  subroutine y_left(this, R, x, y, back)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: R
    real(kind=rp), intent(in) :: x(*)
    real(kind=rp), intent(inout) :: y(*)
    logical, intent(in) :: back
    integer :: n
    if (R .eq. 0) return
    n = this%nf(2)
    if (this%fd(2)%on) then
       call fold_left(this%fd(2), R, x, y, this%hwk, back)
    else
       call dgemm(merge('N', 'T', back), 'N', n, R, n, 1.0_rp, this%Sy, n, x, n, 0.0_rp, y, n)
    end if
  end subroutine y_left

  !> K = D^T W D on the GLL reference element: D(i, j) = l_j'(z_i).
  subroutine reference_stiffness(z, w, lx, K)
    real(kind=rp), intent(in) :: z(:), w(:)
    integer, intent(in) :: lx
    real(kind=rp), allocatable, intent(out) :: K(:,:)
    real(kind=rp), allocatable :: D(:,:), PN(:)
    integer :: i, j, q, N
    real(kind=rp) :: p0, p1, p2
    N = lx - 1
    allocate(D(lx, lx), PN(lx), K(lx, lx))
    do i = 1, lx
       p0 = 1.0_rp
       p1 = z(i)
       if (N .eq. 0) then
          PN(i) = p0
       else
          do q = 2, N
             p2 = ((2 * q - 1) * z(i) * p1 - (q - 1) * p0) / q
             p0 = p1
             p1 = p2
          end do
          PN(i) = p1
       end if
    end do
    D = 0.0_rp
    do i = 1, lx
       do j = 1, lx
          if (i .ne. j) D(i, j) = PN(i) / (PN(j) * (z(i) - z(j)))
       end do
    end do
    D(1, 1) = -real(N * (N + 1), rp) / 4.0_rp
    D(lx, lx) = real(N * (N + 1), rp) / 4.0_rp
    do j = 1, lx
       do i = 1, lx
          K(i, j) = 0.0_rp
          do q = 1, lx
             K(i, j) = K(i, j) + w(q) * D(q, i) * D(q, j)
          end do
       end do
    end do
  end subroutine reference_stiffness

  ! ---------------------------------------------------------------------------
  ! solve
  ! ---------------------------------------------------------------------------

  !> h1, h2 of the operator about to be solved (the residual routines set
  !! them on the device; the host copies are stale there).
  subroutine fdm_update(this)
    class(fdm_pc_t), intent(inout) :: this
    real(kind=rp) :: v(1), v2(2)
    integer(kind=8) :: nbad
    integer :: ierr
    ! Neko develop before 77243ce (#2845, 2026-10-08) allocates an injected
    ! preconditioner but never calls its init: coef stays null
    if (.not. associated(this%coef)) call neko_error('fdm: the preconditioner was never initialized; ' // &
         'this Neko is too old for injected preconditioners (needs develop 77243ce, #2845, 2026-10-08)')
    if (.not. this%ready) call fdm_setup(this)
    this%ifh2 = this%coef%ifh2
    this%nupdate = this%nupdate + 1
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call fdm_dev_download(v, this%coef%h1_d, int(8, c_size_t), glb_cmd_queue)
       this%h1 = v(1)
       if (this%ifh2) then
          call fdm_dev_download(v, this%coef%h2_d, int(8, c_size_t), glb_cmd_queue)
          this%h2 = v(1)
       end if
       ! the inverse is exact for uniform coefficients only (variable
       ! viscosity, SVV, the stress formulation make it a preconditioner):
       ! checked on the first updates and then now and then
       if (.not. this%warned .and. (this%nupdate .le. 3 .or. mod(this%nupdate, 100) .eq. 0)) then
          nbad = fdm_dev_nonuniform(glb_cmd_queue, this%coef%h1_d, int(this%n, 8), this%h1)
          if (this%ifh2) nbad = nbad + fdm_dev_nonuniform(glb_cmd_queue, this%coef%h2_d, int(this%n, 8), this%h2)
          call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER8, MPI_SUM, NEKO_COMM, ierr)
          if (nbad .gt. 0) then
             call neko_log%warning('fdm: h1 or h2 is not uniform: the direct solve is only a preconditioner')
             this%warned = .true.
          end if
       end if
    else
       ! a rank without elements takes the values of the others
       v2 = -huge(1.0_rp)
       if (this%n .gt. 0) then
          v2(1) = this%coef%h1(1, 1, 1, 1)
          if (this%ifh2) v2(2) = this%coef%h2(1, 1, 1, 1)
       end if
       call MPI_Allreduce(MPI_IN_PLACE, v2, 2, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
       this%h1 = v2(1)
       if (this%ifh2) this%h2 = v2(2)
       if (.not. this%warned .and. (this%nupdate .le. 3 .or. mod(this%nupdate, 100) .eq. 0)) then
          nbad = count(abs(this%coef%h1 - this%h1) .gt. 1.0e-12_rp * (abs(this%h1) + 1.0e-300_rp), kind=8)
          if (this%ifh2) nbad = nbad + &
               count(abs(this%coef%h2 - this%h2) .gt. 1.0e-12_rp * (abs(this%h2) + 1.0e-300_rp), kind=8)
          call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER8, MPI_SUM, NEKO_COMM, ierr)
          if (nbad .gt. 0) then
             call neko_log%warning('fdm: h1 or h2 is not uniform: the direct solve is only a preconditioner')
             this%warned = .true.
          end if
       end if
    end if
    if (.not. this%ifh2) this%h2 = 0.0_rp
  end subroutine fdm_update

  subroutine fdm_solve(this, z, r, n)
    integer, intent(in) :: n
    class(fdm_pc_t), intent(inout) :: this
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    call fdm_apply(this, z, r, n)
  end subroutine fdm_solve

  !> z = H^{-1} r (the solve of the preconditioner and of the solver type).
  subroutine fdm_apply(this, z, r, n)
    integer, intent(in) :: n
    class(fdm_pc_t), intent(inout) :: this
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(in) :: r
    type(c_ptr) :: z_d, r_d
    integer :: zi(3), c0, c, ierr
    type(MPI_Request) :: req(16)
    integer(kind=8) :: null_idx
    logical :: null

    if (.not. this%ready) call neko_error('fdm: solve before update')
    if (n .ne. this%n) call neko_error('fdm: solve called with another size')
    this%nsolve = this%nsolve + 1
    null = (this%h2 .eq. 0.0_rp) .and. all(this%zero_mode .gt. 0)
    zi = -1
    null_idx = -1
    if (null) then
       zi = this%zero_mode - 1
       c0 = zi(1) + this%nf(1) * zi(2)
       if (c0 .ge. this%cs_me .and. c0 .lt. this%cs_me + this%ncl) &
            null_idx = int(c0 - this%cs_me, 8) + int(this%ncl, 8) * zi(3)
    end if
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call tick(this, 1)
       z_d = device_get_ptr(z)
       r_d = device_get_ptr(r)
       call tick(this, 2)
       call fdm_dev_gather(this%ctx, r_d)
       call tick(this, 3)
       call exchange(this, 1, 2, this%fs_cnt, this%fs_dsp, this%fr_cnt, this%fr_dsp, this%nheld, this%nrecv)
       call tick(this, 4)
       if (this%twod) then
          if (null) then
             null_idx = -1
             if (zi(1) .ge. this%x0B .and. zi(1) .lt. this%x0B + this%nxB .and. &
                 zi(2) .ge. this%y0C .and. zi(2) .lt. this%y0C + this%nyC) &
                  null_idx = int(zi(1) - this%x0B, 8) + int(this%nxB, 8) * (int(zi(2) - this%y0C, 8) + &
                  int(this%nyC, 8) * zi(3))
          end if
          call fdm_dev2_step(this%ctx, 1, this%h1, this%h2, null_idx)    ! A: x transform, pack
          call tick(this, 5)
          call exchange2(this, this%Py, this%t1s_cnt, this%t1s_dsp, this%t1r_cnt, this%t1r_dsp, this%nA, this%nB, &
               this%comm_k)
          call tick(this, 6)
          call fdm_dev2_step(this%ctx, 2, this%h1, this%h2, null_idx)    ! B: y transform, pack
          call tick(this, 7)
          call exchange2(this, this%Pz, this%t2s_cnt, this%t2s_dsp, this%t2r_cnt, this%t2r_dsp, this%nB, this%nC, &
               this%comm_j)
          call tick(this, 8)
          call fdm_dev2_step(this%ctx, 3, this%h1, this%h2, null_idx)    ! C: z transform, scaling, back, pack
          call tick(this, 9)
          call exchange2(this, this%Pz, this%t2r_cnt, this%t2r_dsp, this%t2s_cnt, this%t2s_dsp, this%nC, this%nB, &
               this%comm_j)
          call tick(this, 10)
          call fdm_dev2_step(this%ctx, 4, this%h1, this%h2, null_idx)    ! B: y back, pack
          call tick(this, 11)
          call exchange2(this, this%Py, this%t1r_cnt, this%t1r_dsp, this%t1s_cnt, this%t1s_dsp, this%nB, this%nA, &
               this%comm_k)
          call tick(this, 12)
          call fdm_dev2_step(this%ctx, 5, this%h1, this%h2, null_idx)    ! A: x back, the received values
          call tick(this, 13)
          call exchange(this, 2, 1, this%fr_cnt, this%fr_dsp, this%fs_cnt, this%fs_dsp, this%nrecv, this%nheld)
          call tick(this, 14)
          call fdm_dev_scatter(this%ctx, z_d)
          call tick(this, 15)
          if (this%profile .and. this%nsolve .eq. 1) this%tprof = 0.0_rp
          if (this%profile .and. mod(this%nsolve, 50) .eq. 0) call profile_report(this)
          return
       end if
       if (this%P .gt. 1 .and. NEKO_DEVICE_MPI .and. this%nch .gt. 1) then
          ! the transposes chunk by chunk, overlapped with the transforms
          call fdm_dev_slab_scatter(this%ctx)
          do c = 1, this%nch
             call fdm_dev_slab_in_planes(this%ctx, this%zc(c - 1, this%me + 1), this%zc(c, this%me + 1))
             call fdm_dev_event_record(this%ctx, c - 1)
          end do
          call tick(this, 5)
          do c = 1, this%nch
             call fdm_dev_event_sync(this%ctx, c - 1)
             call post_chunk(this, 3, 4, this%cs_cnt(:, c), this%cs_dsp(:, c), this%cr_cnt(:, c), this%cr_dsp(:, c), &
                  this%nslab, this%npencil, req(c))
          end do
          call MPI_Waitall(this%nch, req(1:this%nch), MPI_STATUSES_IGNORE, ierr)
          call tick(this, 6)
          call fdm_dev_pencil(this%ctx, this%h1, this%h2, null_idx)
          call tick(this, 7)
          call fdm_dev_sync(this%ctx)
          do c = 1, this%nch
             call post_chunk(this, 4, 3, this%cr_cnt(:, c), this%cr_dsp(:, c), this%cs_cnt(:, c), this%cs_dsp(:, c), &
                  this%npencil, this%nslab, req(c))
          end do
          do c = 1, this%nch
             call MPI_Wait(req(c), MPI_STATUS_IGNORE, ierr)
             call fdm_dev_slab_out_planes(this%ctx, this%zc(c - 1, this%me + 1), this%zc(c, this%me + 1))
          end do
          call tick(this, 8)
          call fdm_dev_slab_gather(this%ctx)
       else
          call fdm_dev_slab_in(this%ctx)
          call tick(this, 5)
          call exchange(this, 3, 4, this%ts_cnt, this%ts_dsp, this%tr_cnt, this%tr_dsp, this%nslab, this%npencil)
          call tick(this, 6)
          call fdm_dev_pencil(this%ctx, this%h1, this%h2, null_idx)
          call tick(this, 7)
          call exchange(this, 4, 3, this%tr_cnt, this%tr_dsp, this%ts_cnt, this%ts_dsp, this%npencil, this%nslab)
          call tick(this, 8)
          call fdm_dev_slab_out(this%ctx)
       end if
       call exchange(this, 2, 1, this%fr_cnt, this%fr_dsp, this%fs_cnt, this%fs_dsp, this%nrecv, this%nheld)
       call tick(this, 9)
       call fdm_dev_scatter(this%ctx, z_d)
       call tick(this, 10)
       if (this%profile .and. this%nsolve .eq. 1) this%tprof = 0.0_rp   ! the first solve loads rocBLAS's kernels
       if (this%profile .and. mod(this%nsolve, 50) .eq. 0) call profile_report(this)
       return
    end if

    call host_apply(this, z, r, n, null, zi, null_idx)
  end subroutine fdm_apply

  ! ---------------------------------------------------------------------------
  ! the host path (CPU backend): the steps of the device path on host buffers
  ! ---------------------------------------------------------------------------

  !> The host buffers of the solve (after the layout): the index tables stay
  !! on the host; on one rank rb aliases sb and pb aliases t.
  subroutine host_buffers(this)
    class(fdm_pc_t), intent(inout) :: this
    integer(kind=8) :: m, maxbuf
    integer :: cl, c
    allocate(this%hsb(max(this%nheld, 1_8)))
    if (this%P .eq. 1) then
       this%hrb => this%hsb
    else
       allocate(this%hrb(max(this%nrecv, 1_8)))
    end if
    if (this%twod) then
       m = max(this%nA, this%nB, this%nC, 1_8)
       allocate(this%hs0(m), this%hs1(m), this%hs2(m), this%hs3(m))
    else
       allocate(this%ha(max(this%nslab, 1_8)), this%ht(max(this%nslab, 1_8)))
       if (this%P .eq. 1) then
          this%hpb => this%ht   ! one rank: the packed slab is the pencil
       else
          allocate(this%hpb(max(this%npencil, 1_8)))
       end if
       allocate(this%hpc(max(this%npencil, 1_8)), this%lxy(max(this%ncl, 1)))
       do cl = 1, this%ncl
          c = this%cs_me + cl - 1
          this%lxy(cl) = this%lamx(mod(c, this%nf(1)) + 1) + this%lamy(c / this%nf(1) + 1)
       end do
       if (this%nch .gt. 1) call neko_log%message('fdm: "chunks" is ignored on the CPU backend')
    end if
    if (any(this%bl%on) .or. any(this%fd%on)) then
       m = max(int(this%nf(1), 8) * this%nf(2), this%npencil, this%nC, 1_8)
       if (this%fd(2)%on) m = max(m, this%nslab)   ! y from the left over the whole slab
       allocate(this%hwk(m))
    end if
    if (this%single .and. this%P .gt. 1) then
       maxbuf = max(this%nheld, this%nrecv, this%nslab, this%npencil, this%nA, this%nB, this%nC, 1_8)
       allocate(this%x4s(maxbuf), this%x4r(maxbuf))
    end if
  end subroutine host_buffers

  !> z = H^{-1} r on the host: gather the held nodes, exchange them to the
  !! slab (or layout A) owners, the transforms and the scaling between the
  !! transposes, back, scatter.
  subroutine host_apply(this, z, r, n, null, zi, null_idx_slab)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), intent(inout) :: z(n)
    real(kind=rp), intent(in) :: r(n)
    logical, intent(in) :: null
    integer, intent(in) :: zi(3)
    integer(kind=8), intent(in) :: null_idx_slab
    integer(kind=8) :: k, null_idx
    integer :: p, u

    call tick(this, 1)
    call tick(this, 2)
    do k = 1, this%nheld
       this%hsb(k) = r(this%gsend(k))
    end do
    call tick(this, 3)
    call host_exchange(this, this%hsb, this%hrb, this%fs_cnt, this%fs_dsp, this%fr_cnt, this%fr_dsp, &
         this%nheld, this%nrecv, NEKO_COMM)
    call tick(this, 4)
    if (this%twod) then
       null_idx = -1
       if (null) then
          if (zi(1) .ge. this%x0B .and. zi(1) .lt. this%x0B + this%nxB .and. &
              zi(2) .ge. this%y0C .and. zi(2) .lt. this%y0C + this%nyC) &
               null_idx = int(zi(1) - this%x0B, 8) + int(this%nxB, 8) * (int(zi(2) - this%y0C, 8) + &
               int(this%nyC, 8) * zi(3))
       end if
       call host2_step(this, 1, null_idx)
       call tick(this, 5)
       call host_exchange2(this, this%Py, this%t1s_cnt, this%t1s_dsp, this%t1r_cnt, this%t1r_dsp, &
            this%nA, this%nB, this%comm_k)
       call tick(this, 6)
       call host2_step(this, 2, null_idx)
       call tick(this, 7)
       call host_exchange2(this, this%Pz, this%t2s_cnt, this%t2s_dsp, this%t2r_cnt, this%t2r_dsp, &
            this%nB, this%nC, this%comm_j)
       call tick(this, 8)
       call host2_step(this, 3, null_idx)
       call tick(this, 9)
       call host_exchange2(this, this%Pz, this%t2r_cnt, this%t2r_dsp, this%t2s_cnt, this%t2s_dsp, &
            this%nC, this%nB, this%comm_j)
       call tick(this, 10)
       call host2_step(this, 4, null_idx)
       call tick(this, 11)
       call host_exchange2(this, this%Py, this%t1r_cnt, this%t1r_dsp, this%t1s_cnt, this%t1s_dsp, &
            this%nB, this%nA, this%comm_k)
       call tick(this, 12)
       call host2_step(this, 5, null_idx)
       call tick(this, 13)
    else
       call host_slab_in(this)
       call tick(this, 5)
       call host_exchange(this, this%ht, this%hpb, this%ts_cnt, this%ts_dsp, this%tr_cnt, this%tr_dsp, &
            this%nslab, this%npencil, NEKO_COMM)
       call tick(this, 6)
       call host_pencil(this, null_idx_slab)
       call tick(this, 7)
       call host_exchange(this, this%hpb, this%ht, this%tr_cnt, this%tr_dsp, this%ts_cnt, this%ts_dsp, &
            this%npencil, this%nslab, NEKO_COMM)
       call tick(this, 8)
       call host_slab_out(this)
    end if
    call host_exchange(this, this%hrb, this%hsb, this%fr_cnt, this%fr_dsp, this%fs_cnt, this%fs_dsp, &
         this%nrecv, this%nheld, NEKO_COMM)
    call tick(this, merge(14, 9, this%twod))
    do p = 1, n
       u = this%sidx(p)
       if (u .gt. 0) then
          z(p) = this%hsb(u)
       else
          z(p) = 0.0_rp
       end if
    end do
    call tick(this, merge(15, 10, this%twod))
    if (this%profile .and. this%nsolve .eq. 1) this%tprof = 0.0_rp
    if (this%profile .and. mod(this%nsolve, 50) .eq. 0) call profile_report(this)
  end subroutine host_apply

  !> The slab from rb (y fastest in a plane), the y transform (one GEMM)
  !! and the x transform (from the right, plane by plane; Bloch if set up),
  !! the pack for the transpose into t (on one rank t is the pencil).
  subroutine host_slab_in(this)
    class(fdm_pc_t), intent(inout) :: this
    integer(kind=8) :: k, ncol, izl, base
    integer :: nfx, nfy, nzl, c, ix, iy
    if (this%nslab .eq. 0) return
    nfx = this%nf(1)
    nfy = this%nf(2)
    nzl = this%nzl
    ncol = int(nfx, 8) * nfy
    associate(a => this%ha, t => this%ht, rb => this%hrb, rpos => this%rpos)
      do k = 1, this%nrecv
         a(rpos(k)) = rb(k)
      end do
      call y_left(this, nfx * nzl, a, t, .false.)
      do izl = 0, nzl - 1
         base = izl * ncol
         call dir_right(this, 1, nfy, t(base + 1), a(base + 1), .false.)
      end do
      do izl = 0, nzl - 1
         base = izl * ncol
         do ix = 0, nfx - 1
            do iy = 0, nfy - 1
               c = ix + nfx * iy + 1
               t(this%cdest(c) + izl * this%cstride(c) + 1) = a(base + iy + int(nfy, 8) * ix + 1)
            end do
         end do
      end do
    end associate
  end subroutine host_slab_in

  !> The z transform of the pencil pb, the scaling, back into pb.
  subroutine host_pencil(this, null_idx)
    class(fdm_pc_t), intent(inout) :: this
    integer(kind=8), intent(in) :: null_idx
    integer :: ncl, nfz, iz, cl
    integer(kind=8) :: base
    real(kind=rp) :: h1, h2, lz, den
    if (this%npencil .eq. 0) return
    ncl = this%ncl
    nfz = this%nf(3)
    h1 = this%h1
    h2 = this%h2
    associate(pb => this%hpb, pc => this%hpc)
      call dir_right(this, 3, ncl, pb, pc, .false.)
      do iz = 0, nfz - 1
         lz = this%lamz(iz + 1)
         base = int(iz, 8) * ncl
         do cl = 1, ncl
            den = h1 * (this%lxy(cl) + lz) + h2
            if (den .eq. 0.0_rp) den = 1.0_rp   ! the null mode, zeroed below
            pc(base + cl) = pc(base + cl) / den
         end do
      end do
      if (null_idx .ge. 0) pc(null_idx + 1) = 0.0_rp
      call dir_right(this, 3, ncl, pc, pb, .true.)
    end associate
  end subroutine host_pencil

  !> The unpack of t, the x and y transforms back, the received values into rb.
  subroutine host_slab_out(this)
    class(fdm_pc_t), intent(inout) :: this
    integer(kind=8) :: k, ncol, izl, base
    integer :: nfx, nfy, nzl, c, ix, iy
    if (this%nslab .eq. 0) return
    nfx = this%nf(1)
    nfy = this%nf(2)
    nzl = this%nzl
    ncol = int(nfx, 8) * nfy
    associate(a => this%ha, t => this%ht, rb => this%hrb, rpos => this%rpos)
      do izl = 0, nzl - 1
         base = izl * ncol
         do ix = 0, nfx - 1
            do iy = 0, nfy - 1
               c = ix + nfx * iy + 1
               a(base + iy + int(nfy, 8) * ix + 1) = t(this%cdest(c) + izl * this%cstride(c) + 1)
            end do
         end do
      end do
      do izl = 0, nzl - 1
         base = izl * ncol
         call dir_right(this, 1, nfy, a(base + 1), t(base + 1), .true.)
      end do
      call y_left(this, nfx * nzl, t, a, .true.)
      do k = 1, this%nrecv
         rb(k) = a(rpos(k))
      end do
    end associate
  end subroutine host_slab_out

  !> The steps of the 2D scheme on the host (as dev2_step; s2 is sent, s3
  !! received; layout A runs y fastest in a z plane, x applied from the
  !! right): 1 A: the slab from rb, x transform, pack; 2 B: unpack, y
  !! transform, pack; 3 C: unpack, z transform, scaling, z back, pack; 4 B:
  !! unpack, y back, pack; 5 A: unpack, x back, the received values into rb.
  subroutine host2_step(this, which, null_idx)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: which
    integer(kind=8), intent(in) :: null_idx
    integer(kind=8) :: i, k, pl, base
    integer :: nfx, nfy, nfz, nxB, nyC, zl, xl, yl, iz
    real(kind=rp) :: den, lz
    nfx = this%nf(1)
    nfy = this%nf(2)
    nfz = this%nf(3)
    nxB = this%nxB
    nyC = this%nyC
    associate(s0 => this%hs0, s1 => this%hs1, s2 => this%hs2, s3 => this%hs3, rb => this%hrb)
      select case (which)
      case (1)
         if (this%nA .eq. 0) return
         do k = 1, this%nrecv
            s0(this%rpos(k)) = rb(k)
         end do
         pl = int(this%nyA, 8) * nfx   ! layout A on the host: (y, x) planes
         do zl = 0, this%nzA - 1
            call dir_right(this, 1, this%nyA, s0(zl * pl + 1), s1(zl * pl + 1), .false.)
         end do
         do i = 1, this%nA
            s2(this%posA(i)) = s1(i)
         end do
      case (2)
         if (this%nB .eq. 0) return
         do i = 1, this%nB
            s0(this%tabB(i)) = s3(i)
         end do
         pl = int(nxB, 8) * nfy
         do zl = 0, this%nzA - 1
            call dir_right(this, 2, nxB, s0(zl * pl + 1), s1(zl * pl + 1), .false.)
         end do
         do i = 1, this%nB
            s2(this%posB(i)) = s1(i)
         end do
      case (3)
         if (this%nC .eq. 0) return
         do i = 1, this%nC
            s0(this%tabC(i)) = s3(i)
         end do
         pl = int(nxB, 8) * nyC
         call dir_right(this, 3, nxB * nyC, s0, s1, .false.)
         do iz = 0, nfz - 1
            lz = this%lamz(iz + 1)
            do yl = 0, nyC - 1
               base = iz * pl + int(yl, 8) * nxB
               do xl = 0, nxB - 1
                  den = this%h1 * (this%lamx(this%x0B + xl + 1) + this%lamy(this%y0C + yl + 1) + lz) + this%h2
                  if (den .eq. 0.0_rp) den = 1.0_rp   ! the null mode, zeroed below
                  s1(base + xl + 1) = s1(base + xl + 1) / den
               end do
            end do
         end do
         if (null_idx .ge. 0) s1(null_idx + 1) = 0.0_rp
         call dir_right(this, 3, nxB * nyC, s1, s0, .true.)
         do i = 1, this%nC
            s2(i) = s0(this%tabC(i))
         end do
      case (4)
         if (this%nB .eq. 0) return
         do i = 1, this%nB
            s0(i) = s3(this%posB(i))
         end do
         pl = int(nxB, 8) * nfy
         do zl = 0, this%nzA - 1
            call dir_right(this, 2, nxB, s0(zl * pl + 1), s1(zl * pl + 1), .true.)
         end do
         do i = 1, this%nB
            s2(i) = s1(this%tabB(i))
         end do
      case default
         if (this%nA .eq. 0) return
         do i = 1, this%nA
            s0(i) = s3(this%posA(i))
         end do
         pl = int(this%nyA, 8) * nfx
         do zl = 0, this%nzA - 1
            call dir_right(this, 1, this%nyA, s0(zl * pl + 1), s1(zl * pl + 1), .true.)
         end do
         do k = 1, this%nrecv
            rb(k) = s1(this%rpos(k))
         end do
      end select
    end associate
  end subroutine host2_step

  !> A transpose of the 2D scheme from s2 to s3 inside a group (a copy when
  !! the group is one rank).
  subroutine host_exchange2(this, pg, scnt, sdsp, rcnt, rdsp, ns, nr, comm)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: pg
    integer, intent(in) :: scnt(:), sdsp(:), rcnt(:), rdsp(:)
    integer(kind=8), intent(in) :: ns, nr
    type(MPI_Comm), intent(in) :: comm
    if (pg .eq. 1) then
       this%hs3(1:ns) = this%hs2(1:ns)
    else
       call host_exchange(this, this%hs2, this%hs3, scnt, sdsp, rcnt, rdsp, ns, nr, comm)
    end if
  end subroutine host_exchange2

  !> One all-to-all of the host path (nothing on one rank: the buffers
  !! alias); in floats with "exchange_precision": "single".
  subroutine host_exchange(this, sb, rb, scnt, sdsp, rcnt, rdsp, ns, nr, comm)
    class(fdm_pc_t), intent(inout) :: this
    real(kind=rp), intent(in) :: sb(:)
    real(kind=rp), intent(inout) :: rb(:)
    integer, intent(in) :: scnt(:), sdsp(:), rcnt(:), rdsp(:)
    integer(kind=8), intent(in) :: ns, nr
    type(MPI_Comm), intent(in) :: comm
    integer(kind=8) :: i
    integer :: ierr
    if (this%P .eq. 1) return
    if (this%single) then
       do i = 1, ns
          this%x4s(i) = real(sb(i), c_float)
       end do
       call MPI_Alltoallv(this%x4s, scnt, sdsp, MPI_REAL4, this%x4r, rcnt, rdsp, MPI_REAL4, comm, ierr)
       do i = 1, nr
          rb(i) = real(this%x4r(i), rp)
       end do
    else
       call MPI_Alltoallv(sb, scnt, sdsp, MPI_DOUBLE_PRECISION, rb, rcnt, rdsp, MPI_DOUBLE_PRECISION, &
            comm, ierr)
    end if
  end subroutine host_exchange


  ! ---------------------------------------------------------------------------
  ! the solver type
  ! ---------------------------------------------------------------------------

  subroutine fdm_ksp_init(this, n, max_iter, M, rel_tol, abs_tol, monitor)
    class(fdm_ksp_t), target, intent(inout) :: this
    integer, intent(in) :: max_iter
    class(pc_t), optional, intent(in), target :: M
    integer, intent(in) :: n
    real(kind=rp), optional, intent(in) :: rel_tol
    real(kind=rp), optional, intent(in) :: abs_tol
    logical, optional, intent(in) :: monitor
    call this%free()
    this%n = n
    if (present(M)) this%M => M
    call this%ksp_init(max_iter, rel_tol = rel_tol, abs_tol = abs_tol, monitor = monitor)
  end subroutine fdm_ksp_init

  subroutine fdm_ksp_free(this)
    class(fdm_ksp_t), intent(inout) :: this
    call this%ksp_free()
    if (allocated(this%w)) then
       if (c_associated(this%w_d)) call device_unmap(this%w, this%w_d)
       deallocate(this%w)
    end if
    if (allocated(this%w2)) then
       if (c_associated(this%w2_d)) call device_unmap(this%w2, this%w2_d)
       deallocate(this%w2)
    end if
    if (allocated(this%w3)) then
       if (c_associated(this%w3_d)) call device_unmap(this%w3, this%w3_d)
       deallocate(this%w3)
    end if
  end subroutine fdm_ksp_free

  !> The fdm preconditioner behind this solver (an error otherwise).
  function fdm_ksp_pc(this) result(pc)
    class(fdm_ksp_t), intent(in) :: this
    type(fdm_pc_t), pointer :: pc
    pc => null()
    if (associated(this%M)) then
       select type (M => this%M)
       type is (fdm_pc_t)
          pc => M
       end select
    end if
    if (.not. associated(pc)) &
         call neko_error('the "fdm" solver needs the "fdm" preconditioner (its solve is the solver)')
  end function fdm_ksp_pc

  !> Work array w (and w2, w3 for the coupled residual), mapped to the device.
  subroutine fdm_ksp_work(this, n, three)
    class(fdm_ksp_t), intent(inout) :: this
    integer, intent(in) :: n
    logical, intent(in) :: three
    if (.not. allocated(this%w)) then
       allocate(this%w(n))
       this%w = 0.0_rp
       if (NEKO_BCKND_DEVICE .eq. 1) call device_map(this%w, this%w_d, n)
    end if
    if (three .and. .not. allocated(this%w2)) then
       allocate(this%w2(n), this%w3(n))
       this%w2 = 0.0_rp
       this%w3 = 0.0_rp
       if (NEKO_BCKND_DEVICE .eq. 1) then
          call device_map(this%w2, this%w2_d, n)
          call device_map(this%w3, this%w3_d, n)
       end if
    end if
  end subroutine fdm_ksp_work

  !> ||f||_mult / sqrt(volume), as Neko's Krylov solvers report residuals.
  function fdm_ksp_norm(f, coef, n) result(nrm)
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
  end function fdm_ksp_norm

  function fdm_ksp_solve(this, Ax, x, f, n, coef, bc_projector, gs_h, niter) result(ksp_results)
    class(fdm_ksp_t), intent(inout) :: this
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(in) :: f
    type(coef_t), intent(inout) :: coef
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    integer, optional, intent(in) :: niter
    type(ksp_monitor_t) :: ksp_results
    type(fdm_pc_t), pointer :: pc
    type(c_ptr) :: f_d

    pc => fdm_ksp_pc(this)
    this%ncall = this%ncall + 1
    ksp_results%res_start = fdm_ksp_norm(f, coef, n)
    call this%monitor_start('fdm')
    call fdm_apply(pc, x%x, f, n)
    ksp_results%iter = 1
    ! the true residual costs one operator application: measured on the
    ! first ten solves and every 100th (always with "residual": true);
    ! otherwise the last measured value is reported
    if (pc%residual .or. this%ncall .le. 10 .or. mod(this%ncall, 100) .eq. 0) then
       call fdm_ksp_work(this, n, .false.)
       call Ax%compute(this%w, x%x, coef, x%msh, x%Xh)
       call gs_h%op(this%w, n, GS_OP_ADD)
       call bc_projector%apply(this%w, n)
       if (NEKO_BCKND_DEVICE .eq. 1) then
          f_d = device_get_ptr(f)
          call device_sub2(this%w_d, f_d, n)
       else
          call sub2(this%w, f, n)
       end if
       this%last_res(1) = fdm_ksp_norm(this%w, coef, n)
    end if
    ksp_results%res_final = this%last_res(1)
    call this%monitor_iter(1, ksp_results%res_final)
    call this%monitor_stop()
    ksp_results%converged = .true.
  end function fdm_ksp_solve

  function fdm_ksp_solve_coupled(this, Ax, x, y, z, fx, fy, fz, n, coef, bc_projector, gs_h, niter) &
       result(ksp_results)
    class(fdm_ksp_t), intent(inout) :: this
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
    type(fdm_pc_t), pointer :: pc
    type(c_ptr) :: fx_d, fy_d, fz_d

    pc => fdm_ksp_pc(this)
    this%ncall = this%ncall + 1
    ksp_results(1)%res_start = fdm_ksp_norm(fx, coef, n)
    ksp_results(2)%res_start = fdm_ksp_norm(fy, coef, n)
    ksp_results(3)%res_start = fdm_ksp_norm(fz, coef, n)
    call this%monitor_start('fdm (coupled)')
    call fdm_apply(pc, x%x, fx, n)
    call fdm_apply(pc, y%x, fy, n)
    call fdm_apply(pc, z%x, fz, n)
    ksp_results%iter = 1
    if (pc%residual .or. this%ncall .le. 10 .or. mod(this%ncall, 100) .eq. 0) then
       call fdm_ksp_work(this, n, .true.)
       call Ax%compute_vector(this%w, this%w2, this%w3, x%x, y%x, z%x, coef, x%msh, x%Xh)
       call gs_h%op(this%w, this%w2, this%w3, n, GS_OP_ADD)
       call bc_projector%apply(this%w, this%w2, this%w3, n)
       if (NEKO_BCKND_DEVICE .eq. 1) then
          fx_d = device_get_ptr(fx)
          fy_d = device_get_ptr(fy)
          fz_d = device_get_ptr(fz)
          call device_sub2(this%w_d, fx_d, n)
          call device_sub2(this%w2_d, fy_d, n)
          call device_sub2(this%w3_d, fz_d, n)
       else
          call sub2(this%w, fx, n)
          call sub2(this%w2, fy, n)
          call sub2(this%w3, fz, n)
       end if
       this%last_res(1) = fdm_ksp_norm(this%w, coef, n)
       this%last_res(2) = fdm_ksp_norm(this%w2, coef, n)
       this%last_res(3) = fdm_ksp_norm(this%w3, coef, n)
    end if
    ksp_results%res_final = this%last_res
    call this%monitor_iter(1, maxval(ksp_results%res_final))
    call this%monitor_stop()
    ksp_results%converged = .true.
  end function fdm_ksp_solve_coupled

  !> Profiling of the solve (option "profile"): synchronizes the stream and
  !! accumulates the wall time since the previous tick into slot k - 1.
  subroutine tick(this, k)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: k
    integer :: ierr
    real(kind=rp), save :: tlast = 0.0_rp
    real(kind=rp) :: t
    if (.not. this%profile) return
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call fdm_dev_sync(this%ctx)
    else
       ! host: a phase ends when every rank has finished it (the time of the
       ! slowest rank; the exchanges then time the communication alone)
       call MPI_Barrier(NEKO_COMM, ierr)
    end if
    t = MPI_Wtime()
    if (k .gt. 1) this%tprof(k - 1) = this%tprof(k - 1) + (t - tlast)
    tlast = t
  end subroutine tick

  subroutine profile_report(this)
    class(fdm_pc_t), intent(inout) :: this
    character(len=LOG_SIZE) :: log_buf
    character(len=12), parameter :: names(9) = [character(len=12) :: 'get_ptr', 'gather', &
         'xchg nodes', 'slab in', 'xchg to pen', 'pencil', 'xchg back', 'slab out+x', 'scatter']
    character(len=12), parameter :: names2(14) = [character(len=12) :: 'get_ptr', 'gather', &
         'xchg nodes', 'A: x, pack', 'xchg A->B', 'B: y, pack', 'xchg B->C', 'C: z, scale', &
         'xchg C->B', 'B: y back', 'xchg B->A', 'A: x back', 'xchg nodes', 'scatter']
    integer :: k, nk
    nk = merge(14, 9, this%twod)
    write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A)') 'fdm phases (', this%nf(1), 'x', this%nf(2), 'x', this%nf(3), &
         '), mean of ', this%nsolve - 1, ' solves [ms]:'
    call neko_log%message(log_buf)
    do k = 1, nk
       if (this%twod) then
          write(log_buf, '(A,A,F9.3)') '  ', names2(k), this%tprof(k) / (this%nsolve - 1) * 1.0e3_rp
       else
          write(log_buf, '(A,A,F9.3)') '  ', names(k), this%tprof(k) / (this%nsolve - 1) * 1.0e3_rp
       end if
       call neko_log%message(log_buf)
    end do
  end subroutine profile_report

  !> A transpose of the 2D scheme between the send slot (5) and the receive
  !! slot (6) inside a group; a device copy when the group is one rank.
  subroutine exchange2(this, pg, scnt, sdsp, rcnt, rdsp, ns, nr, comm)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: pg
    integer, intent(in) :: scnt(:), sdsp(:), rcnt(:), rdsp(:)
    integer(kind=8), intent(in) :: ns, nr
    type(MPI_Comm), intent(in) :: comm
    if (pg .eq. 1) then
       call fdm_dev_copy(this%ctx, 5, 6, ns)
    else
       call exchange(this, 5, 6, scnt, sdsp, rcnt, rdsp, ns, nr, comm)
    end if
  end subroutine exchange2

  !> One chunk of a transpose as a non-blocking all-to-all on the device
  !! buffers `from` and `to` (device MPI).
  subroutine post_chunk(this, from, to, scnt, sdsp, rcnt, rdsp, ns, nr, req)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: from, to
    integer, intent(in) :: scnt(:), sdsp(:), rcnt(:), rdsp(:)
    integer(kind=8), intent(in) :: ns, nr
    type(MPI_Request), intent(out) :: req
    real(kind=rp), pointer :: sb(:), rb(:)
    real(kind=c_float), pointer :: sb4(:), rb4(:)
    integer :: ierr
    if (this%single) then
       call c_f_pointer(fdm_dev_ptr(this%ctx, from), sb4, [max(ns, 1_8)])
       call c_f_pointer(fdm_dev_ptr(this%ctx, to), rb4, [max(nr, 1_8)])
       call MPI_Ialltoallv(sb4, scnt, sdsp, MPI_REAL4, rb4, rcnt, rdsp, MPI_REAL4, NEKO_COMM, req, ierr)
    else
       call c_f_pointer(fdm_dev_ptr(this%ctx, from), sb, [max(ns, 1_8)])
       call c_f_pointer(fdm_dev_ptr(this%ctx, to), rb, [max(nr, 1_8)])
       call MPI_Ialltoallv(sb, scnt, sdsp, MPI_DOUBLE_PRECISION, rb, rcnt, rdsp, MPI_DOUBLE_PRECISION, &
            NEKO_COMM, req, ierr)
    end if
  end subroutine post_chunk

  !> One all-to-all of the solve between the device buffers `from` and `to`
  !! (1 sbuf, 2 rbuf, 3 packed slab, 4 pencil): a copy on one rank, MPI on
  !! the device buffers with device MPI, through the host otherwise.
  subroutine exchange(this, from, to, scnt, sdsp, rcnt, rdsp, ns, nr, comm)
    class(fdm_pc_t), intent(inout) :: this
    integer, intent(in) :: from, to
    integer, intent(in) :: scnt(:), sdsp(:), rcnt(:), rdsp(:)
    integer(kind=8), intent(in) :: ns, nr
    type(MPI_Comm), intent(in), optional :: comm
    type(MPI_Comm) :: cm
    real(kind=rp), pointer :: sb(:), rb(:)
    real(kind=c_float), pointer :: sb4(:), rb4(:)
    integer :: ierr
    if (this%P .eq. 1) return   ! rbuf is sbuf and the pencil buffer is the packed slab
    cm = NEKO_COMM
    if (present(comm)) cm = comm
    if (NEKO_DEVICE_MPI) then
       call fdm_dev_sync(this%ctx)
       if (this%single) then
          call c_f_pointer(fdm_dev_ptr(this%ctx, from), sb4, [max(ns, 1_8)])
          call c_f_pointer(fdm_dev_ptr(this%ctx, to), rb4, [max(nr, 1_8)])
          call MPI_Alltoallv(sb4, scnt, sdsp, MPI_REAL4, rb4, rcnt, rdsp, MPI_REAL4, cm, ierr)
       else
          call c_f_pointer(fdm_dev_ptr(this%ctx, from), sb, [max(ns, 1_8)])
          call c_f_pointer(fdm_dev_ptr(this%ctx, to), rb, [max(nr, 1_8)])
          call MPI_Alltoallv(sb, scnt, sdsp, MPI_DOUBLE_PRECISION, rb, rcnt, rdsp, &
               MPI_DOUBLE_PRECISION, cm, ierr)
       end if
    else
       call fdm_dev_d2h(this%ctx, from, this%hs, ns)
       call MPI_Alltoallv(this%hs, scnt, sdsp, MPI_DOUBLE_PRECISION, this%hr, rcnt, rdsp, &
            MPI_DOUBLE_PRECISION, cm, ierr)
       call fdm_dev_h2d(this%ctx, to, this%hr, nr)
    end if
  end subroutine exchange

  !> The host buffers (the aliases of one rank are not deallocated twice).
  subroutine host_free(this)
    class(fdm_pc_t), intent(inout) :: this
    integer :: d
    if (associated(this%hrb, this%hsb)) nullify(this%hrb)
    if (associated(this%hpb, this%ht)) nullify(this%hpb)
    if (associated(this%hsb)) deallocate(this%hsb)
    if (associated(this%hrb)) deallocate(this%hrb)
    if (associated(this%ha)) deallocate(this%ha)
    if (associated(this%ht)) deallocate(this%ht)
    if (associated(this%hpb)) deallocate(this%hpb)
    if (associated(this%hpc)) deallocate(this%hpc)
    if (associated(this%hs0)) deallocate(this%hs0)
    if (associated(this%hs1)) deallocate(this%hs1)
    if (associated(this%hs2)) deallocate(this%hs2)
    if (associated(this%hs3)) deallocate(this%hs3)
    if (associated(this%hwk)) deallocate(this%hwk)
    nullify(this%hsb, this%hrb, this%ha, this%ht, this%hpb, this%hpc, this%hs0, this%hs1, this%hs2, this%hs3, &
         this%hwk)
    if (allocated(this%x4s)) deallocate(this%x4s, this%x4r)
    if (allocated(this%lxy)) deallocate(this%lxy)
    do d = 1, 3
       if (allocated(this%bl(d)%C)) deallocate(this%bl(d)%C, this%bl(d)%V, this%bl(d)%off, this%bl(d)%sz)
       this%bl(d)%on = .false.
       if (allocated(this%fd(d)%Ve)) deallocate(this%fd(d)%Ve, this%fd(d)%Vo)
       this%fd(d)%on = .false.
    end do
  end subroutine host_free

  subroutine fdm_free(this)
    class(fdm_pc_t), intent(inout) :: this
    integer :: ierr
    if (c_associated(this%ctx)) then
       call fdm_dev_free(this%ctx)
       this%ctx = C_NULL_PTR
    end if
    if (allocated(this%Sx)) deallocate(this%Sx)
    if (allocated(this%Sy)) deallocate(this%Sy)
    if (allocated(this%Sz)) deallocate(this%Sz)
    if (allocated(this%lamx)) deallocate(this%lamx)
    if (allocated(this%lamy)) deallocate(this%lamy)
    if (allocated(this%lamz)) deallocate(this%lamz)
    if (allocated(this%gsend)) deallocate(this%gsend)
    if (allocated(this%rpos)) deallocate(this%rpos)
    if (allocated(this%sidx)) deallocate(this%sidx)
    if (allocated(this%cdest)) deallocate(this%cdest)
    if (allocated(this%cstride)) deallocate(this%cstride)
    if (allocated(this%fs_cnt)) deallocate(this%fs_cnt, this%fs_dsp, this%fr_cnt, this%fr_dsp)
    if (allocated(this%ts_cnt)) deallocate(this%ts_cnt, this%ts_dsp, this%tr_cnt, this%tr_dsp)
    if (allocated(this%hs)) deallocate(this%hs, this%hr)
    if (allocated(this%posA)) deallocate(this%posA, this%tabB, this%posB, this%tabC)
    if (allocated(this%zc)) deallocate(this%zc, this%cs_cnt, this%cs_dsp, this%cr_cnt, this%cr_dsp)
    if (allocated(this%t1s_cnt)) deallocate(this%t1s_cnt, this%t1s_dsp, this%t1r_cnt, this%t1r_dsp)
    if (allocated(this%t2s_cnt)) deallocate(this%t2s_cnt, this%t2s_dsp, this%t2r_cnt, this%t2r_dsp)
    if (this%twod .and. this%ready) then
       call MPI_Comm_free(this%comm_k, ierr)
       call MPI_Comm_free(this%comm_j, ierr)
    end if
    this%twod = .false.
    call host_free(this)
    nullify(this%coef)
    nullify(this%bclst)
    this%ready = .false.
  end subroutine fdm_free

end module box_fdm
