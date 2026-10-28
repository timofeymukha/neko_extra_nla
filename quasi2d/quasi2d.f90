!> quasi2d: direct solves for Neko on quasi-2D meshes (one periodic element
!! across z), injected through Neko's type registry (no change to Neko).
!!
!! Build:  makeneko q2d_spchol.f90 quasi2d.f90 [user.f90]   and use
!!         "preconditioner": {"type": "quasi2d"} inside Neko's own Krylov
!!         solvers (scalar problems), or the solver type "quasi2d" (PCG with
!!         this preconditioner; scalar problems and the coupled velocity of
!!         the full stress formulation).
!!
!! The mesh: a 2D quad mesh (curved elements, periodic directions in the
!! plane, any element orientation) times ONE element along z, periodic in z
!! (Neko's 2D .nmsh thin slab, or genmeshbox with nelz = 1). With the
!! B_z-orthonormal eigenbasis S of the 1D periodic z operators (A_z S =
!! B_z S Lambda, N_z = lx - 1 modes) and coefficients h1(x, y), h2(x, y):
!!
!! - scalar Helmholtz (pressure, velocity components, scalars), exact:
!!     (A_xy^{h1} + lambda_k B_xy^{h1} + B_xy^{h2}) u_k = (S^T f)_k;
!! - full-stress velocity (Neko's ax_helm_full), A^s = sum of 2D blocks
!!   times B_z, A_z and the weak derivative C_z. Mode 0 (z-constant) splits
!!   off exactly into the 2D (u, v) stress operator and the 2D Helmholtz for
!!   w; the higher modes couple only u-w and v-w across modes (through
!!   S^T C_z S). The preconditioner keeps, per mode, the coupled (u, v)
!!   block and the w block (exact for mode 0), so a 2D flow converges in one
!!   PCG iteration and a 3D one in about 8-22 (logbook 2026-10-26).
!!
!! Layout (CPU): every element holds whole z columns, so the z transform is
!! local. The planes are factored and solved by all ranks together
!! (q2d_plane): rank 0 orders each plane (sparse supernodal Cholesky,
!! q2d_spchol, symbolic only) and maps the elimination tree onto the ranks
!! subtree-to-subcube; the subtrees are factored on single ranks, the
!! separators above them on rank groups (dense fronts distributed by rows),
!! and a solve sweeps the tree the same way. The transformed right-hand side goes
!! from Neko's layout to the owners of the plane dofs and back with one
!! all-to-all each way. Each mode is factored on first use, and again when
!! h1 / h2 change. Modes whose
!! right-hand side is below skip_tol (relative) are not solved: their
!! solution is zero up to that tolerance (a 2D flow solves only mode 0).
!!
!! The 2D numbering comes from Neko's own dof ids on the z = 0 plane, so it
!! carries Neko's periodicity in x, y. The output is averaged over Neko's
!! copies of each dof (gs), which keeps it consistent with Neko's
!! connectivity even where Neko's thin slab joins vertical faces mirrored in
!! z (a Neko bug, NEKO_ISSUES.md, 2026-10-26; mode 0 is unaffected).
module quasi2d
  use num_types, only : rp, dp, i8
  use precon, only : pc_t, register_precon, precon_allocate
  use krylov, only : ksp_t, ksp_monitor_t, register_krylov, krylov_allocate
  use ax_product, only : ax_t
  use field, only : field_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use scalar_bc_projector, only : scalar_bc_projector_t
  use vector_bc_projector, only : vector_bc_projector_t
  use math, only : glsc3
  use coefs, only : coef_t
  use bc_list, only : bc_list_t
  use bc, only : bc_t, BC_DIRICHLET, BC_NEUMANN
  use json_module, only : json_file
  use json_utils, only : json_get_or_default
  use comm, only : NEKO_COMM, pe_rank, pe_size
  use mpi_f08, only : MPI_Allreduce, MPI_Allgather, MPI_Allgatherv, MPI_Alltoallv, MPI_Gatherv, MPI_Bcast, &
       MPI_IN_PLACE, MPI_INTEGER, MPI_INTEGER8, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_MAX, MPI_MIN, &
       MPI_LOR, MPI_LOGICAL, MPI_Wtime
  use utils, only : neko_error
  use logger, only : neko_log, LOG_SIZE
  use neko_config, only : NEKO_BCKND_DEVICE
  use q2d_plane, only : plane_t
  implicit none
  private

  public :: quasi2d_register_types

  !> The exchange between Neko's layout (local 2D nodes, components) and the
  !! owners of a plane's columns: every local free node goes to the owner of
  !! its plane dof; the solution comes back the same way.
  type :: xch_t
     integer :: nc = 0                         !< components on Neko's side
     integer, allocatable :: scnt(:), sdsp(:), sl(:), sc(:)        !< Neko side, by destination
     integer, allocatable :: rcnt(:), rdsp(:), rcol(:), rrhs(:)    !< owner side, by source
  end type xch_t

  type, public, extends(pc_t) :: q2d_pc_t
     type(coef_t), pointer :: coef => null()
     type(bc_list_t), pointer :: bclst => null()
     logical :: ready = .false.
     logical :: verbose = .false.
     real(kind=dp) :: skip_tol = 1.0e-6_dp
     character(len=8) :: residual = 'auto'    !< solver type: 'always' or 'auto' (see q2d_ksp_check)
     real(kind=dp) :: refactor_tol = 0.0_dp
     integer :: lx = 0, nz = 0, n = 0, nelv = 0, ne2 = 0
     real(kind=dp) :: Lz = 0.0_dp
     !> local points -> (local 2D node, z index); local nodes -> global node
     integer :: nl = 0
     integer, allocatable :: pl(:), pz(:), lrep(:,:), lglob(:)
     integer, allocatable :: eax(:)            !< extrusion axis (local direction) of every element
     logical, allocatable :: zinc(:)           !< z increases along it
     !> global 2D nodes, Dirichlet, free numbering
     integer :: nn = 0, nfree = 0
     integer, allocatable :: fid(:)
     logical :: singular = .false.
     !> z: S (nz, nz), lam (nz)
     real(kind=dp), allocatable :: S(:,:), lam(:)
     !> the planes (scalar pattern; interleaved (u, v) pattern), their exchanges
     type(plane_t) :: ps, puv
     type(xch_t) :: xs, xuv, xw
     !> rank 0: the free node of every point of the 2D elements of the whole
     !! mesh (lx*lx, ne2; 0: Dirichlet), for the patterns
     integer, allocatable :: enode(:,:)
     !> this rank's elements: free node (lx*lx, nelv), geometry (5, lx*lx,
     !! nelv: rx, sx, ry, sy, W), coefficients (2, lx*lx, nelv: z-averaged h1,
     !! h2) now and those of the factors
     integer, allocatable :: lenode(:,:)
     real(kind=dp), allocatable :: lgeo(:,:,:), leh(:,:,:), lehf(:,:,:)
     !> rank 0: CSR patterns (free nodes; interleaved (u, v) for the stress)
     integer, allocatable :: ia(:), ja(:), ia2(:), ja2(:)
     !> coefficient key (to detect changes) and the coefficients on this rank's elements
     real(kind=dp) :: hkey(4) = -1.0_dp
     real(kind=dp), allocatable :: Dr(:,:), Ds(:,:)
     !> work
     real(kind=dp), allocatable :: C(:,:), F(:,:), sb(:), rb(:)
     real(kind=dp), allocatable :: zt(:)
     integer :: nsolve = 0, nskip = 0, nfact = 0
     real(kind=dp) :: t_ph(5) = 0.0_dp      !< transform, setup/factor, plane solves, -, scatter
     real(kind=dp) :: t_setup = 0.0_dp, t_fact = 0.0_dp, t_solve = 0.0_dp
     real(kind=dp) :: t_asm = 0.0_dp        !< element matrices of the plane factorizations (this rank)
     logical :: zwarn = .false.
   contains
     procedure, pass(this) :: init => q2d_init
     procedure, pass(this) :: solve => q2d_solve
     procedure, pass(this) :: update => q2d_update
     procedure, pass(this) :: free => q2d_free
     procedure, pass(this) :: apply3 => q2d_apply3
  end type q2d_pc_t

  !> PCG with the quasi2d preconditioner (Neko's operator, Neko's norms).
  type, public, extends(ksp_t) :: q2d_ksp_t
     real(kind=rp), allocatable :: r(:,:), z(:,:), p(:,:), w(:,:)
     integer :: n = 0
     integer :: stress = -1          !< coupled solves: -1 unknown, 0 segregated operator, 1 full stress
     integer :: ncall = 0            !< solves so far
     logical :: exact = .true.       !< every checked solve so far converged in one iteration
     real(kind=rp) :: ratio = 0.0_rp !< res_final / res_start of the last checked solve
   contains
     procedure, pass(this) :: init => q2d_ksp_init
     procedure, pass(this) :: solve => q2d_ksp_solve
     procedure, pass(this) :: solve_coupled => q2d_ksp_solve_coupled
     procedure, pass(this) :: free => q2d_ksp_free
  end type q2d_ksp_t

contains

  subroutine quasi2d_register_types()
    procedure(precon_allocate), pointer :: alloc
    procedure(krylov_allocate), pointer :: alloc_ksp
    alloc => q2d_allocate
    call register_precon("quasi2d", alloc)
    alloc_ksp => q2d_ksp_allocate
    call register_krylov("quasi2d", alloc_ksp)
  end subroutine quasi2d_register_types

  subroutine q2d_allocate(obj)
    class(pc_t), allocatable, intent(inout) :: obj
    allocate(q2d_pc_t :: obj)
  end subroutine q2d_allocate

  subroutine q2d_ksp_allocate(obj)
    class(ksp_t), allocatable, intent(inout) :: obj
    allocate(q2d_ksp_t :: obj)
  end subroutine q2d_ksp_allocate

  !> Stores the pointers and options; the setup waits for the first update()
  !! (the boundary masks are not set yet when Neko creates the preconditioner).
  subroutine q2d_init(this, coef, bclst, json)
    class(q2d_pc_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json
    call this%free()
    this%coef => coef
    this%bclst => bclst
    call json_get_or_default(json, 'verbose', this%verbose, .false.)
    call json_get_or_default(json, 'skip_tol', this%skip_tol, 1.0e-6_dp)
    block
      character(len=:), allocatable :: str
      call json_get_or_default(json, 'residual', str, 'auto')
      this%residual = trim(str)
      if (this%residual .ne. 'auto' .and. this%residual .ne. 'always') &
           call neko_error('quasi2d: "residual" is "auto" or "always"')
    end block
    call json_get_or_default(json, 'refactor_tol', this%refactor_tol, 0.0_dp)
    this%ready = .false.
  end subroutine q2d_init

  subroutine q2d_update(this)
    class(q2d_pc_t), intent(inout) :: this
    if (.not. associated(this%coef)) call neko_error('quasi2d: preconditioner never initialized ' // &
         '(Neko develop >= 77243ce needed)')
    if (NEKO_BCKND_DEVICE .eq. 1) call neko_error('quasi2d: CPU backend only (for now)')
    if (.not. this%ready) call q2d_setup(this)
    call q2d_coefficients(this)
  end subroutine q2d_update

  ! ---------------------------------------------------------------------------
  ! setup
  ! ---------------------------------------------------------------------------

  subroutine q2d_setup(this)
    class(q2d_pc_t), intent(inout), target :: this
    type(coef_t), pointer :: coef
    character(len=320) :: log_buf
    integer :: lx, N, nelv, e, i, j, k, a, t, u, v, p, q, l, ierr, r, nzt, ne2, nb, o, m, cnt, iz
    integer, allocatable :: eax(:), cnts(:), dsps(:), ord(:), node0(:), lnode(:), d2i(:)
    integer(kind=i8), allocatable :: key(:), ukey(:), allkey(:)
    real(kind=dp), allocatable :: xf(:), yf(:), zf(:), D(:,:), w(:), Az(:,:), bz(:), sbuf(:), rbuf(:)
    real(kind=dp) :: span, tol, dxy, dz, best, zmin, zmax, t0, xr, xs, yr, ys, jac
    logical, allocatable :: dflag(:), d2(:)
    logical, allocatable :: zinc(:)
    integer :: nbad, nkey

    t0 = MPI_Wtime()
    coef => this%coef
    lx = coef%Xh%lx; N = lx - 1
    nelv = coef%msh%nelv
    this%lx = lx; this%nz = N; this%nelv = nelv
    this%n = lx**3 * nelv
    allocate(xf(this%n), yf(this%n), zf(this%n))
    xf = real(reshape(coef%dof%x%x, [this%n]), dp)
    yf = real(reshape(coef%dof%y%x, [this%n]), dp)
    zf = real(reshape(coef%dof%z%x, [this%n]), dp)
    allocate(D(lx, lx), w(lx))
    D = real(coef%Xh%dx, dp)
    w = real(coef%Xh%wx, dp)

    ! --- 1. the extrusion axis of every element; one layer [zmin, zmax] -----
    span = 0.0_dp
    if (nelv .gt. 0) span = max(maxval(xf) - minval(xf), maxval(yf) - minval(yf), maxval(zf) - minval(zf))
    call MPI_Allreduce(MPI_IN_PLACE, span, 1, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    tol = max(1.0e-7_dp * span, 64.0_dp * real(epsilon(1.0_rp), dp) * span)
    zmin = huge(1.0_dp); zmax = -huge(1.0_dp)
    if (nelv .gt. 0) then
       zmin = minval(zf); zmax = maxval(zf)
    end if
    call MPI_Allreduce(MPI_IN_PLACE, zmin, 1, MPI_DOUBLE_PRECISION, MPI_MIN, NEKO_COMM, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, zmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    this%Lz = zmax - zmin
    allocate(eax(nelv), zinc(nelv))
    nbad = 0
    do e = 1, nelv
       eax(e) = 0
       do a = 1, 3
          dxy = 0.0_dp; dz = 0.0_dp
          do k = 1, lx
             do j = 1, lx
                do i = 1, lx - 1
                   p = fp(a, j, k, i, e); q = fp(a, j, k, i + 1, e)
                   dxy = max(dxy, abs(xf(p) - xf(q)) + abs(yf(p) - yf(q)))
                   dz = max(dz, abs(zf(p) - zf(q)))
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
       ! one layer: the element spans [zmin, zmax]
       best = zf(fp(eax(e), 1, 1, lx, e)) - zf(fp(eax(e), 1, 1, 1, e))
       zinc(e) = best .gt. 0.0_dp
       if (abs(abs(best) - this%Lz) .gt. tol) nbad = nbad + 1
    end do
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    if (nbad .gt. 0) then
       write(log_buf, '(A,I0,A)') 'quasi2d: ', nbad, ' elements are not one full z layer ' // &
            '(the mesh must be one element across z)'
       call neko_error(trim(log_buf))
    end if
    if (.not. z_periodic(coef, zmin, zmax, tol)) &
         call neko_error('quasi2d: z must be periodic (one periodic element across z)')

    ! --- 2. 2D nodes: Neko's dof ids on the z = 0 plane ---------------------
    ! z index of a point: (t - 1) along increasing z, mod N
    allocate(this%pl(this%n), this%pz(this%n), node0(lx * lx * max(nelv, 1)))
    allocate(key(lx * lx * max(nelv, 1)), ord(lx * lx * max(nelv, 1)))
    nkey = 0
    do e = 1, nelv
       do v = 1, lx
          do u = 1, lx
             nkey = nkey + 1
             p = fp(eax(e), u, v, tzero(e), e)
             key(nkey) = coef%dof%dof(mod(p - 1, lx) + 1, mod((p - 1) / lx, lx) + 1, &
                  mod((p - 1) / (lx * lx), lx) + 1, (p - 1) / lx**3 + 1)
             ord(nkey) = nkey
          end do
       end do
    end do
    call sort_key_index(key, ord, nkey)
    ! local unique nodes
    allocate(lnode(max(nkey, 1)), ukey(max(nkey, 1)))
    this%nl = 0
    do i = 1, nkey
       if (i .eq. 1) then
          this%nl = 1
          ukey(1) = key(ord(1))
       else if (key(ord(i)) .ne. key(ord(i - 1))) then
          this%nl = this%nl + 1
          ukey(this%nl) = key(ord(i))
       end if
       lnode(ord(i)) = this%nl
    end do
    ! global numbering: all keys, sorted, unique
    allocate(cnts(0:pe_size - 1), dsps(0:pe_size - 1))
    call MPI_Allgather(this%nl, 1, MPI_INTEGER, cnts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    dsps(0) = 0
    do r = 1, pe_size - 1
       dsps(r) = dsps(r - 1) + cnts(r - 1)
    end do
    nb = dsps(pe_size - 1) + cnts(pe_size - 1)
    allocate(allkey(nb))
    call MPI_Allgatherv(ukey, this%nl, MPI_INTEGER8, allkey, cnts, dsps, MPI_INTEGER8, NEKO_COMM, ierr)
    allocate(d2i(nb))
    do i = 1, nb
       d2i(i) = i
    end do
    call sort_key_index(allkey, d2i, nb)
    ! compact: allkey(d2i(:)) ascending
    block
      integer(kind=i8), allocatable :: gk(:)
      allocate(gk(nb))
      this%nn = 0
      do i = 1, nb
         if (i .eq. 1) then
            this%nn = 1; gk(1) = allkey(d2i(1))
         else if (allkey(d2i(i)) .ne. allkey(d2i(i - 1))) then
            this%nn = this%nn + 1; gk(this%nn) = allkey(d2i(i))
         end if
      end do
      allocate(this%lglob(this%nl))
      do l = 1, this%nl
         this%lglob(l) = find_i8(gk, this%nn, ukey(l))
      end do
    end block
    ! points -> (local node, z index), and a representative point per (z, node)
    allocate(this%lrep(0:N - 1, this%nl))
    this%lrep = 0
    nkey = 0
    do e = 1, nelv
       do v = 1, lx
          do u = 1, lx
             nkey = nkey + 1
             l = lnode(nkey)
             do t = 1, lx
                p = fp(eax(e), u, v, t, e)
                if (zinc(e)) then
                   iz = mod(t - 1, N)
                else
                   iz = mod(lx - t, N)
                end if
                this%pl(p) = l
                this%pz(p) = iz
                if (this%lrep(iz, l) .eq. 0) this%lrep(iz, l) = p
             end do
          end do
       end do
    end do

    ! --- 3. Dirichlet nodes (independent of z), free numbering -------------
    allocate(dflag(this%n), d2(this%nn))
    dflag = .false.
    do i = 1, this%bclst%size()
       call mark_dirichlet(this%bclst%get(i), dflag, this%n)
    end do
    d2 = .false.
    do p = 1, this%n
       if (dflag(p)) d2(this%lglob(this%pl(p))) = .true.
    end do
    call MPI_Allreduce(MPI_IN_PLACE, d2, this%nn, MPI_LOGICAL, MPI_LOR, NEKO_COMM, ierr)
    nbad = 0
    do p = 1, this%n
       if (d2(this%lglob(this%pl(p))) .neqv. dflag(p)) nbad = nbad + 1
    end do
    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM, ierr)
    if (nbad .gt. 0) then
       write(log_buf, '(A,I0,A)') 'quasi2d: the Dirichlet set depends on z (', nbad, ' points)'
       call neko_error(trim(log_buf))
    end if
    allocate(this%fid(this%nn))
    this%nfree = 0
    do i = 1, this%nn
       if (d2(i)) then
          this%fid(i) = 0
       else
          this%nfree = this%nfree + 1
          this%fid(i) = this%nfree
       end if
    end do

    ! --- 4. z: 1D operators of one periodic element, eigenbasis -------------
    allocate(Az(N, N), bz(N), this%S(N, N), this%lam(N))
    Az = 0.0_dp; bz = 0.0_dp
    do j = 1, lx
       q = mod(j - 1, N) + 1
       bz(q) = bz(q) + 0.5_dp * this%Lz * w(j)
       do i = 1, lx
          p = mod(i - 1, N) + 1
          Az(p, q) = Az(p, q) + 2.0_dp / this%Lz * sum(D(:, i) * w * D(:, j))
       end do
    end do
    call gen_eig(Az, bz, N, this%S, this%lam)

    ! --- 5. the 2D elements: kept here, their nodes to rank 0 (patterns) ----
    ! element data: node ids, rx, sx, ry, sy, W per point
    allocate(this%Dr(lx * lx, lx * lx), this%Ds(lx * lx, lx * lx))
    this%Dr = 0.0_dp; this%Ds = 0.0_dp
    do j = 1, lx
       do i = 1, lx
          do k = 1, lx
             this%Dr(i + lx * (j - 1), k + lx * (j - 1)) = D(i, k)
             this%Ds(i + lx * (j - 1), i + lx * (k - 1)) = D(j, k)
          end do
       end do
    end do
    allocate(sbuf(6 * lx * lx * max(nelv, 1)))
    do e = 1, nelv
       do v = 1, lx
          do u = 1, lx
             a = u + lx * (v - 1)
             o = 6 * (a - 1 + lx * lx * (e - 1))
             p = fp(eax(e), u, v, tzero(e), e)
             sbuf(o + 1) = real(this%lglob(this%pl(p)), dp)
             ! derivatives along u (r) and v (s) on the face
             xr = 0.0_dp; xs = 0.0_dp; yr = 0.0_dp; ys = 0.0_dp
             do k = 1, lx
                xr = xr + D(u, k) * xf(fp(eax(e), k, v, tzero(e), e))
                yr = yr + D(u, k) * yf(fp(eax(e), k, v, tzero(e), e))
                xs = xs + D(v, k) * xf(fp(eax(e), u, k, tzero(e), e))
                ys = ys + D(v, k) * yf(fp(eax(e), u, k, tzero(e), e))
             end do
             jac = xr * ys - xs * yr
             sbuf(o + 2) = ys / jac          ! r_x
             sbuf(o + 3) = -yr / jac         ! s_x
             sbuf(o + 4) = -xs / jac         ! r_y
             sbuf(o + 5) = xr / jac          ! s_y
             sbuf(o + 6) = w(u) * w(v) * abs(jac)
          end do
       end do
    end do
    allocate(this%lenode(lx * lx, max(nelv, 1)), this%lgeo(5, lx * lx, max(nelv, 1)))
    do e = 1, nelv
       do a = 1, lx * lx
          o = 6 * (a - 1 + lx * lx * (e - 1))
          this%lenode(a, e) = this%fid(nint(sbuf(o + 1)))
          this%lgeo(:, a, e) = sbuf(o + 2:o + 6)
       end do
    end do
    call MPI_Allgather(nelv, 1, MPI_INTEGER, cnts, 1, MPI_INTEGER, NEKO_COMM, ierr)
    ne2 = sum(cnts)
    this%ne2 = ne2
    nzt = 6 * lx * lx
    dsps(0) = 0
    do r = 1, pe_size - 1
       dsps(r) = dsps(r - 1) + nzt * cnts(r - 1)
    end do
    if (pe_rank .eq. 0) then
       allocate(rbuf(nzt * ne2))
    else
       allocate(rbuf(1))
    end if
    call MPI_Gatherv(sbuf, nzt * nelv, MPI_DOUBLE_PRECISION, rbuf, nzt * cnts, dsps, &
         MPI_DOUBLE_PRECISION, 0, NEKO_COMM, ierr)
    if (pe_rank .eq. 0) then
       allocate(this%enode(lx * lx, ne2))
       do e = 1, ne2
          do a = 1, lx * lx
             o = 6 * (a - 1 + lx * lx * (e - 1))
             this%enode(a, e) = this%fid(nint(rbuf(o + 1)))
          end do
       end do
       call build_patterns(this)
       deallocate(this%enode)
    end if
    deallocate(rbuf)


    ! singular: no Dirichlet node (h2 = 0 is checked per solve)
    this%singular = this%nfree .eq. this%nn

    allocate(this%C(max(this%nl, 1), N), this%F(max(this%nl, 1), N), this%zt(this%n))
    call move_alloc(eax, this%eax)
    call move_alloc(zinc, this%zinc)
    this%t_setup = MPI_Wtime() - t0
    write(log_buf, '(A,I0,A,I0,A,I0,A,I0,A,F7.2,A)') 'quasi2d: 2D mesh ', ne2, ' elements, ', this%nn, &
         ' nodes (', this%nfree, ' free), ', N, ' z modes; setup ', this%t_setup, ' s'
    call neko_log%message(log_buf)
    this%ready = .true.

  contains

    !> the local point at face coordinates (uu, vv) and extrusion index tt
    integer function fp(ax, uu, vv, tt, el)
      integer, intent(in) :: ax, uu, vv, tt, el
      integer :: idx(3)
      select case (ax)
      case (1)
         idx = [tt, uu, vv]
      case (2)
         idx = [uu, tt, vv]
      case default
         idx = [uu, vv, tt]
      end select
      fp = idx(1) + lx * ((idx(2) - 1) + lx * ((idx(3) - 1) + lx * (el - 1)))
    end function fp

    !> the extrusion index of the z = zmin plane of element el
    integer function tzero(el)
      integer, intent(in) :: el
      if (zinc(el)) then
         tzero = 1
      else
         tzero = lx
      end if
    end function tzero

  end subroutine q2d_setup

  !> CSR patterns on the free nodes (scalar; interleaved (u, v)) and the
  !! positions of the diagonals
  subroutine build_patterns(this)
    class(q2d_pc_t), intent(inout) :: this
    integer :: nl2, nf, e, a, b, i, k, row, col, pos, cnt, c
    integer, allocatable :: nep(:), ne(:), mark(:), fill(:)
    nl2 = this%lx * this%lx; nf = this%nfree
    allocate(nep(nf + 1), ne(nl2 * this%ne2), mark(nf), fill(nf))
    nep = 0
    do e = 1, this%ne2
       do a = 1, nl2
          i = this%enode(a, e)
          if (i .gt. 0) nep(i + 1) = nep(i + 1) + 1
       end do
    end do
    nep(1) = 1
    do i = 1, nf
       nep(i + 1) = nep(i + 1) + nep(i)
    end do
    fill = 0
    do e = 1, this%ne2
       do a = 1, nl2
          i = this%enode(a, e)
          if (i .eq. 0) cycle
          ne(nep(i) + fill(i)) = e
          fill(i) = fill(i) + 1
       end do
    end do
    allocate(this%ia(nf + 1))
    mark = 0
    this%ia(1) = 1
    do row = 1, nf
       cnt = 0
       do k = nep(row), nep(row + 1) - 1
          e = ne(k)
          do b = 1, nl2
             col = this%enode(b, e)
             if (col .eq. 0) cycle
             if (mark(col) .ne. row) then
                mark(col) = row
                cnt = cnt + 1
             end if
          end do
       end do
       this%ia(row + 1) = this%ia(row) + cnt
    end do
    allocate(this%ja(this%ia(nf + 1) - 1))
    mark = 0
    do row = 1, nf
       pos = this%ia(row)
       do k = nep(row), nep(row + 1) - 1
          e = ne(k)
          do b = 1, nl2
             col = this%enode(b, e)
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
    ! interleaved (u, v): row 2 (f - 1) + c, columns 2 (g - 1) + 1, + 2
    allocate(this%ia2(2 * nf + 1), this%ja2(2 * (this%ia(nf + 1) - 1) * 2))
    this%ia2(1) = 1
    do row = 1, nf
       do c = 1, 2
          i = 2 * (row - 1) + c
          this%ia2(i + 1) = this%ia2(i) + 2 * (this%ia(row + 1) - this%ia(row))
          pos = this%ia2(i)
          do k = this%ia(row), this%ia(row + 1) - 1
             this%ja2(pos) = 2 * (this%ja(k) - 1) + 1
             this%ja2(pos + 1) = 2 * (this%ja(k) - 1) + 2
             pos = pos + 2
          end do
       end do
    end do
  end subroutine build_patterns

  !> The coefficients h1, h2 of the next solve (Neko sets them before
  !! update()): averaged over z per 2D point (exact when they do not depend on
  !! z), compared with the last ones by a key; on a change every rank keeps
  !! the new values of its elements, and all decide together whether to
  !! refactor.
  subroutine q2d_coefficients(this)
    class(q2d_pc_t), intent(inout) :: this
    integer :: lx, e, u, v, t, p, a, o, ierr, n, refac
    real(kind=dp) :: key(4), hb(2), dev, hmax, diff(2), nrm(2)
    real(kind=dp), allocatable :: sbuf(:), h1(:), h2(:), w(:)
    character(len=320) :: log_buf
    lx = this%lx; n = this%n
    allocate(w(lx), h1(n), h2(n))
    w = real(this%coef%Xh%wx, dp)
    w = w / sum(w)
    h1 = real(reshape(this%coef%h1, [n]), dp)
    h2 = 0.0_dp
    if (this%coef%ifh2) h2 = real(reshape(this%coef%h2, [n]), dp)
    allocate(sbuf(2 * lx * lx * max(this%nelv, 1)))
    sbuf = 0.0_dp
    dev = 0.0_dp; hmax = 0.0_dp
    do e = 1, this%nelv
       do v = 1, lx
          do u = 1, lx
             hb = 0.0_dp
             do t = 1, lx
                p = fpt(lx, this%eax(e), u, v, t, e)
                hb(1) = hb(1) + w(t) * h1(p)
                hb(2) = hb(2) + w(t) * h2(p)
             end do
             do t = 1, lx
                p = fpt(lx, this%eax(e), u, v, t, e)
                dev = max(dev, abs(h1(p) - hb(1)), abs(h2(p) - hb(2)))
                hmax = max(hmax, abs(h1(p)), abs(h2(p)))
             end do
             a = u + lx * (v - 1)
             o = 2 * (a - 1 + lx * lx * (e - 1))
             sbuf(o + 1:o + 2) = hb
          end do
       end do
    end do
    key = 0.0_dp
    if (this%nelv .gt. 0) then
       key(1) = sum(sbuf(1::2)**2); key(2) = sum(sbuf(2::2)**2)
       key(3) = maxval(sbuf(1::2)); key(4) = maxval(sbuf(2::2))
    end if
    call MPI_Allreduce(MPI_IN_PLACE, key(1:2), 2, MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, key(3:4), 2, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    if (all(abs(key - this%hkey) .le. 1.0e-14_dp * abs(key))) return
    this%hkey = key
    call MPI_Allreduce(MPI_IN_PLACE, dev, 1, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, hmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
    if (dev .gt. 1.0e-10_dp * hmax .and. .not. this%zwarn) then
       write(log_buf, '(A,ES9.2,A)') 'quasi2d: h1/h2 depend on z (relative ', dev / hmax, &
            '); their z average is used, the solve is a preconditioner'
       call neko_log%message(log_buf)
       this%zwarn = .true.
    end if
    ! keep them; refactor when they moved by more than refactor_tol (max norm
    ! over all ranks)
    if (.not. allocated(this%leh)) allocate(this%leh(2, lx * lx, max(this%nelv, 1)))
    this%leh = 0.0_dp
    if (this%nelv .gt. 0) this%leh(:, :, 1:this%nelv) = reshape(sbuf, [2, lx * lx, this%nelv])
    refac = 1
    if (allocated(this%lehf)) then
       nrm = 0.0_dp; diff = 0.0_dp
       do a = 1, 2
          if (this%nelv .gt. 0) then
             nrm(a) = maxval(abs(this%lehf(a, :, 1:this%nelv)))
             diff(a) = maxval(abs(this%leh(a, :, 1:this%nelv) - this%lehf(a, :, 1:this%nelv)))
          end if
       end do
       call MPI_Allreduce(MPI_IN_PLACE, nrm, 2, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
       call MPI_Allreduce(MPI_IN_PLACE, diff, 2, MPI_DOUBLE_PRECISION, MPI_MAX, NEKO_COMM, ierr)
       if (this%refactor_tol .gt. 0.0_dp .and. all(diff .le. this%refactor_tol * max(nrm, tiny(1.0_dp)))) &
            refac = 0
    else
       allocate(this%lehf(2, lx * lx, max(this%nelv, 1)))
    end if
    if (refac .eq. 1) then
       this%lehf = this%leh
       if (allocated(this%ps%fresh)) this%ps%fresh = .false.
       if (allocated(this%puv%fresh)) this%puv%fresh = .false.
    end if
  end subroutine q2d_coefficients


  !> the local point at face coordinates (u, v) and index t along axis ax
  pure integer function fpt(lx, ax, u, v, t, el)
    integer, intent(in) :: lx, ax, u, v, t, el
    integer :: idx(3)
    select case (ax)
    case (1)
       idx = [t, u, v]
    case (2)
       idx = [u, t, v]
    case default
       idx = [u, v, t]
    end select
    fpt = idx(1) + lx * ((idx(2) - 1) + lx * ((idx(3) - 1) + lx * (el - 1)))
  end function fpt

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
       call neko_error('quasi2d: only Dirichlet and Neumann conditions are supported')
    end if
  end subroutine mark_dirichlet

  !> z periodic: a periodic facet of the mesh lies at zmin or zmax
  function z_periodic(coef, zmin, zmax, tz) result(per)
    type(coef_t), intent(in) :: coef
    real(kind=dp), intent(in) :: zmin, zmax, tz
    logical :: per
    integer :: m, f, e, lx, i, j, k, i0, i1, j0, j1, k0, k1, flag, ierr
    real(kind=dp) :: lo, hi, zv
    lx = coef%Xh%lx
    flag = 0
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
       if (hi - lo .lt. tz .and. (abs(lo - zmin) .lt. tz .or. abs(lo - zmax) .lt. tz)) flag = 1
    end do
    call MPI_Allreduce(MPI_IN_PLACE, flag, 1, MPI_INTEGER, MPI_MAX, NEKO_COMM, ierr)
    per = flag .eq. 1
  end function z_periodic

  ! ---------------------------------------------------------------------------
  ! the 2D matrices and the factors (owners)
  ! ---------------------------------------------------------------------------

  !> element matrices of the 2D Helmholtz stiffness with weight h, and of the
  !! stress blocks, for this rank's element e
  subroutine element_matrices(this, e, h, lap, mxx, myy, myx, stress)
    class(q2d_pc_t), intent(inout) :: this
    integer, intent(in) :: e
    real(kind=dp), intent(in) :: h(:)
    real(kind=dp), intent(out) :: lap(:,:), mxx(:,:), myy(:,:), myx(:,:)
    logical, intent(in) :: stress
    integer :: nl2, a
    real(kind=dp), allocatable :: gx(:,:), gy(:,:), tx(:,:), ty(:,:)
    nl2 = this%lx * this%lx
    allocate(gx(nl2, nl2), gy(nl2, nl2), tx(nl2, nl2), ty(nl2, nl2))
    do a = 1, nl2
       gx(a, :) = this%lgeo(1, a, e) * this%Dr(a, :) + this%lgeo(2, a, e) * this%Ds(a, :)
       gy(a, :) = this%lgeo(3, a, e) * this%Dr(a, :) + this%lgeo(4, a, e) * this%Ds(a, :)
       tx(a, :) = this%lgeo(5, a, e) * h(a) * gx(a, :)
       ty(a, :) = this%lgeo(5, a, e) * h(a) * gy(a, :)
    end do
    call dgemm('T', 'N', nl2, nl2, nl2, 1.0_dp, gx, nl2, tx, nl2, 0.0_dp, mxx, nl2)
    call dgemm('T', 'N', nl2, nl2, nl2, 1.0_dp, gy, nl2, ty, nl2, 0.0_dp, myy, nl2)
    lap = mxx + myy
    if (stress) call dgemm('T', 'N', nl2, nl2, nl2, 1.0_dp, gy, nl2, tx, nl2, 0.0_dp, myx, nl2)
  end subroutine element_matrices

  !> This rank's contributions to the matrix of plane kp (1: scalar, 2:
  !! interleaved (u, v) stress), element by element: their number nc, their
  !! (row, column) (crow, ccol), or their values for mode m (cw = 1 for scalar problems, 2
  !! for the w block of the stress, whose z term is 2 lambda B). Scalar: the
  !! h1-Laplacian plus (cw lambda h1 + h2) mass on the diagonal; (u, v): the
  !! blocks (row c, col d) = delta_cd Lap + G_d^T W G_c plus the same mass
  !! (cw = 1).
  subroutine contributions(this, kp, nc, crow, ccol, m, cw, vals)
    class(q2d_pc_t), intent(inout) :: this
    integer, intent(in) :: kp
    integer, intent(out) :: nc
    integer, intent(inout), optional :: crow(:), ccol(:)
    integer, intent(in), optional :: m, cw
    real(kind=dp), intent(inout), optional :: vals(:)
    integer :: nl2, e, a, b, row, col
    real(kind=dp) :: lam, ms
    real(kind=dp), allocatable :: lap(:,:), mxx(:,:), myy(:,:), myx(:,:)
    logical :: pat, vls
    nl2 = this%lx * this%lx
    pat = present(crow)
    vls = present(vals)
    if (vls) then
       lam = this%lam(m)
       allocate(lap(nl2, nl2), mxx(nl2, nl2), myy(nl2, nl2), myx(nl2, nl2))
    end if
    nc = 0
    do e = 1, this%nelv
       if (vls) call element_matrices(this, e, this%leh(1, :, e), lap, mxx, myy, myx, kp .eq. 2)
       do b = 1, nl2
          col = this%lenode(b, e)
          if (col .eq. 0) cycle
          do a = 1, nl2
             row = this%lenode(a, e)
             if (row .eq. 0) cycle
             if (kp .eq. 1) then
                nc = nc + 1
                if (pat) then
                   crow(nc) = row; ccol(nc) = col
                else if (vls) then
                   vals(nc) = lap(a, b)
                   if (a .eq. b) vals(nc) = vals(nc) + this%lgeo(5, a, e) * &
                        (real(cw, dp) * lam * this%leh(1, a, e) + this%leh(2, a, e))
                end if
             else
                if (pat) then
                   crow(nc + 1:nc + 4) = [2 * row - 1, 2 * row - 1, 2 * row, 2 * row]
                   ccol(nc + 1:nc + 4) = [2 * col - 1, 2 * col, 2 * col - 1, 2 * col]
                else if (vls) then
                   ms = 0.0_dp
                   if (a .eq. b) ms = this%lgeo(5, a, e) * (lam * this%leh(1, a, e) + this%leh(2, a, e))
                   vals(nc + 1) = lap(a, b) + mxx(a, b) + ms
                   vals(nc + 2) = myx(a, b)
                   vals(nc + 3) = myx(b, a)
                   vals(nc + 4) = lap(a, b) + myy(a, b) + ms
                end if
                nc = nc + 4
             end if
          end do
       end do
    end do
  end subroutine contributions

  !> plane kp set up (ordering and symbolic factorization on rank 0; this
  !! rank's contribution pattern), and mode m factored. Collective.
  subroutine ensure_mode(this, kp, m, cw)
    class(q2d_pc_t), intent(inout), target :: this
    integer, intent(in) :: kp, m, cw
    type(plane_t), pointer :: pl
    integer, allocatable :: crow(:), ccol(:)
    real(kind=dp), allocatable :: vals(:)
    integer :: di(1), n, nc, pin
    real(kind=dp) :: t0, t1
    logical :: dopin
    character(len=320) :: log_buf
    if (kp .eq. 1) then
       pl => this%ps
       n = this%nfree
    else
       pl => this%puv
       n = 2 * this%nfree
    end if
    t0 = MPI_Wtime()
    if (.not. pl%ready) then
       if (pe_rank .eq. 0) then
          if (kp .eq. 1) then
             call pl%setup(n, this%nz, this%ia, this%ja)
          else
             call pl%setup(n, this%nz, this%ia2, this%ja2)
          end if
       else
          call pl%setup(n, this%nz, di, di)
       end if
       call contributions(this, kp, nc)
       allocate(crow(max(nc, 1)), ccol(max(nc, 1)))
       call contributions(this, kp, nc, crow, ccol)
       ! pure Neumann: the last free node may be pinned
       pin = 0
       if (kp .eq. 1 .and. this%singular) pin = this%nfree
       call pl%contrib_setup(nc, crow, ccol, pin)
       write(log_buf, '(A,I0,A,I0,A,I0,A,F8.3,A)') 'quasi2d: plane ', kp, ': ', n, ' unknowns, nnz(L) ', &
            pl%nnz_l, '; ordering and symbolic in ', MPI_Wtime() - t0, ' s'
       call neko_log%message(log_buf)
       t0 = MPI_Wtime()
    end if
    if (pl%fresh(m)) return
    t1 = MPI_Wtime()
    call contributions(this, kp, nc)
    allocate(vals(max(nc, 1)))
    call contributions(this, kp, nc, m = m, cw = cw, vals = vals)
    this%t_asm = this%t_asm + (MPI_Wtime() - t1)
    ! pure Neumann, no mass: the constant is the null vector, pin the last node
    dopin = kp .eq. 1 .and. cw .eq. 1 .and. this%singular .and. this%lam(m) .eq. 0.0_dp .and. &
         this%hkey(4) .eq. 0.0_dp
    call pl%factor(m, nc, vals, dopin)
    this%nfact = this%nfact + 1
    this%t_fact = this%t_fact + (MPI_Wtime() - t0)
    if (this%verbose .or. this%nfact .le. 2 * this%nz) then
       write(log_buf, '(A,I0,A,I0,A,ES10.3,A,F8.3,A)') 'quasi2d: factored plane ', kp, ' mode ', m, &
            ' (lambda ', this%lam(m), ') in ', MPI_Wtime() - t0, ' s'
       call neko_log%message(log_buf, lvl = 1)
    end if
  end subroutine ensure_mode

  !> the exchange of plane kp for nc components starting at cfirst (scalar
  !! plane: nc right-hand sides; (u, v) plane: components 1, 2). Collective.
  subroutine build_xch(this, x, pl, kp, nc, cfirst)
    class(q2d_pc_t), intent(inout) :: this
    type(xch_t), intent(inout) :: x
    type(plane_t), intent(in) :: pl
    integer, intent(in) :: kp, nc, cfirst
    integer :: l, f, c, d, ne, i, ierr, P, nent
    integer, allocatable :: dest(:), el(:), ec(:), eold(:), erhs(:), fill(:), sb(:), rb(:)
    P = pe_size
    if (allocated(x%scnt)) deallocate(x%scnt, x%sdsp, x%sl, x%sc, x%rcnt, x%rdsp, x%rcol, x%rrhs)
    x%nc = nc
    nent = 0
    if (kp .eq. 1) then
       nent = nc * this%nl
    else
       nent = 2 * this%nl
    end if
    allocate(dest(nent), el(nent), ec(nent), eold(nent), erhs(nent))
    ne = 0
    do l = 1, this%nl
       f = this%fid(this%lglob(l))
       if (f .eq. 0) cycle
       if (kp .eq. 1) then
          do c = 1, nc
             ne = ne + 1
             el(ne) = l; ec(ne) = cfirst + c - 1; eold(ne) = f; erhs(ne) = c
          end do
       else
          do c = 1, 2
             ne = ne + 1
             el(ne) = l; ec(ne) = c; eold(ne) = 2 * (f - 1) + c; erhs(ne) = 1
          end do
       end if
    end do
    allocate(x%scnt(0:P - 1), x%sdsp(0:P - 1), x%rcnt(0:P - 1), x%rdsp(0:P - 1))
    x%scnt = 0
    do i = 1, ne
       dest(i) = pl%own_old(eold(i))
       x%scnt(dest(i)) = x%scnt(dest(i)) + 1
    end do
    x%sdsp(0) = 0
    do d = 1, P - 1
       x%sdsp(d) = x%sdsp(d - 1) + x%scnt(d - 1)
    end do
    allocate(x%sl(max(ne, 1)), x%sc(max(ne, 1)), sb(max(2 * ne, 1)), fill(0:P - 1))
    fill = 0
    do i = 1, ne
       d = dest(i)
       fill(d) = fill(d) + 1
       x%sl(x%sdsp(d) + fill(d)) = el(i)
       x%sc(x%sdsp(d) + fill(d)) = ec(i)
       sb(2 * (x%sdsp(d) + fill(d)) - 1) = eold(i)
       sb(2 * (x%sdsp(d) + fill(d))) = erhs(i)
    end do
    call MPI_Alltoall_int(x%scnt, x%rcnt)
    x%rdsp(0) = 0
    do d = 1, P - 1
       x%rdsp(d) = x%rdsp(d - 1) + x%rcnt(d - 1)
    end do
    allocate(rb(max(2 * sum(x%rcnt), 1)), x%rcol(max(sum(x%rcnt), 1)), x%rrhs(max(sum(x%rcnt), 1)))
    call MPI_Alltoallv(sb, 2 * x%scnt, 2 * x%sdsp, MPI_INTEGER, rb, 2 * x%rcnt, 2 * x%rdsp, MPI_INTEGER, &
         NEKO_COMM, ierr)
    do i = 1, sum(x%rcnt)
       x%rcol(i) = pl%local_index(rb(2 * i - 1))
       if (x%rcol(i) .eq. 0) call neko_error('quasi2d: a plane dof sent to a rank that does not own it')
       x%rrhs(i) = rb(2 * i)
    end do
  end subroutine build_xch

  subroutine MPI_Alltoall_int(a, b)
    use mpi_f08, only : MPI_Alltoall
    integer, intent(in) :: a(:)
    integer, intent(out) :: b(:)
    integer :: ierr
    call MPI_Alltoall(a, 1, MPI_INTEGER, b, 1, MPI_INTEGER, NEKO_COMM, ierr)
  end subroutine MPI_Alltoall_int


  ! ---------------------------------------------------------------------------
  ! the solve
  ! ---------------------------------------------------------------------------

  !> z = M r for one scalar field
  subroutine q2d_solve(this, z, r, n)
    class(q2d_pc_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    if (.not. this%ready) call this%update()
    call q2d_apply(this, 1, 1, n, r, z)
  end subroutine q2d_solve

  !> (zx, zy, zz) = M (rx, ry, rz) for the velocity: the full-stress
  !! operator (stress), or three scalar problems solved together
  subroutine q2d_apply3(this, zx, zy, zz, rx, ry, rz, n, stress)
    class(q2d_pc_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: zx, zy, zz
    real(kind=rp), dimension(n), intent(in) :: rx, ry, rz
    logical, intent(in) :: stress
    if (.not. this%ready) call this%update()
    if (stress) then
       call q2d_apply(this, 3, 2, n, rx, zx, ry, zy, rz, zz)
    else
       call q2d_apply(this, 3, 1, n, rx, zx, ry, zy, rz, zz)
    end if
  end subroutine q2d_apply3

  !> nc right-hand sides; kind 1: nc scalar problems with the same matrix,
  !! kind 2: the full-stress velocity (nc = 3)
  subroutine q2d_apply(this, nc, kind, n, r1, z1, r2, z2, r3, z3)
    class(q2d_pc_t), intent(inout), target :: this
    integer, intent(in) :: nc, kind, n
    real(kind=rp), dimension(n), intent(in) :: r1
    real(kind=rp), dimension(n), intent(inout) :: z1
    real(kind=rp), dimension(n), intent(in), optional :: r2, r3
    real(kind=rp), dimension(n), intent(inout), optional :: z2, z3
    integer :: nz, nl, c, l, iz, m, ierr, na, ip, nplanes
    real(kind=dp), allocatable :: Fc(:,:,:), U(:,:,:)
    real(kind=dp) :: nrm(this%nz), t0, t1
    integer :: act(this%nz)
    character(len=320) :: log_buf
    t0 = MPI_Wtime()
    nz = this%nz; nl = this%nl
    allocate(Fc(max(nl, 1), nz, nc), U(max(nl, 1), nz, nc))
    ! 1. z transform of every local column: F = C S
    do c = 1, nc
       do l = 1, nl
          do iz = 0, nz - 1
             select case (c)
             case (1)
                this%C(l, iz + 1) = real(r1(this%lrep(iz, l)), dp)
             case (2)
                this%C(l, iz + 1) = real(r2(this%lrep(iz, l)), dp)
             case default
                this%C(l, iz + 1) = real(r3(this%lrep(iz, l)), dp)
             end select
          end do
       end do
       if (nl .gt. 0) call dgemm('N', 'N', nl, nz, nz, 1.0_dp, this%C, size(this%C, 1), this%S, nz, &
            0.0_dp, Fc(1, 1, c), size(Fc, 1))
    end do
    this%t_ph(1) = this%t_ph(1) + (MPI_Wtime() - t0); t1 = MPI_Wtime()
    ! 2. the modes to solve
    nrm = 0.0_dp
    do m = 1, nz
       do c = 1, nc
          if (nl .gt. 0) nrm(m) = nrm(m) + sum(Fc(1:nl, m, c)**2)
       end do
    end do
    call MPI_Allreduce(MPI_IN_PLACE, nrm, nz, MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM, ierr)
    nrm = sqrt(nrm)
    na = 0
    do m = 1, nz
       if (nrm(m) .gt. this%skip_tol * maxval(nrm)) then
          na = na + 1
          act(na) = m
       end if
    end do
    this%nskip = this%nskip + (nz - na)
    if (this%verbose .and. (this%nsolve .lt. 3 .or. mod(this%nsolve, 100) .eq. 0)) then
       write(log_buf, '(A,I0,A,7ES9.1)') 'quasi2d: solve ', this%nsolve + 1, ' relative mode norms ', &
            nrm(1:min(nz, 7)) / max(maxval(nrm), tiny(1.0_dp))
       call neko_log%message(log_buf)
    end if
    ! 3. planes, exchanges and factors
    if (kind .eq. 1) then
       nplanes = 1
       do m = 1, na
          call ensure_mode(this, 1, act(m), 1)
       end do
       if (this%xs%nc .ne. nc) call build_xch(this, this%xs, this%ps, 1, nc, 1)
    else
       nplanes = 2
       do m = 1, na
          call ensure_mode(this, 2, act(m), 1)
          call ensure_mode(this, 1, act(m), 2)
       end do
       if (this%xuv%nc .ne. 2) call build_xch(this, this%xuv, this%puv, 2, 2, 1)
       if (this%xw%nc .ne. 1) call build_xch(this, this%xw, this%ps, 1, 1, 3)
    end if
    this%t_ph(2) = this%t_ph(2) + (MPI_Wtime() - t1); t1 = MPI_Wtime()
    ! 4. the solves
    U = 0.0_dp
    if (na .gt. 0) then
       if (kind .eq. 1) then
          call plane_solve(this, this%ps, this%xs, nc, .true.)
       else
          call plane_solve(this, this%puv, this%xuv, 1, .false.)
          call plane_solve(this, this%ps, this%xw, 1, .false.)
       end if
    end if
    this%t_ph(3) = this%t_ph(3) + (MPI_Wtime() - t1); t1 = MPI_Wtime()
    ! 5. back: C = U S^T per column, to every point
    do c = 1, nc
       if (nl .gt. 0) call dgemm('N', 'T', nl, nz, nz, 1.0_dp, U(1, 1, c), size(U, 1), this%S, nz, &
            0.0_dp, this%C, size(this%C, 1))
       select case (c)
       case (1)
          call scatter(this, z1, n)
       case (2)
          call scatter(this, z2, n)
       case default
          call scatter(this, z3, n)
       end select
    end do
    ! average over Neko's copies of each dof (one gather-scatter for all)
    if (nc .eq. 1) then
       call this%coef%gs_h%op(z1, n, GS_OP_ADD)
       call mult_flat(n, this%coef%mult, z1)
    else
       call this%coef%gs_h%op(z1, z2, z3, n, GS_OP_ADD)
       call mult_flat(n, this%coef%mult, z1)
       call mult_flat(n, this%coef%mult, z2)
       call mult_flat(n, this%coef%mult, z3)
    end if
    this%t_ph(5) = this%t_ph(5) + (MPI_Wtime() - t1)
    this%nsolve = this%nsolve + 1
    this%t_solve = this%t_solve + (MPI_Wtime() - t0)

  contains

    !> Fc -> owners -> sweep -> U, for the active modes
    subroutine plane_solve(self, pl, x, nr, scalar)
      class(q2d_pc_t), intent(inout) :: self
      type(plane_t), intent(inout) :: pl
      type(xch_t), intent(in) :: x
      integer, intent(in) :: nr
      logical, intent(in) :: scalar
      real(kind=dp), allocatable :: Y(:,:,:), sb(:), rb(:)
      integer :: ld, d, e, a, pos, cc, lp, ns, nrv
      real(kind=dp) :: mean(nr)
      logical :: sing
      ld = max(pl%ncol + pl%next, 1)
      allocate(Y(ld, nr, na))
      Y = 0.0_dp
      ns = sum(x%scnt); nrv = sum(x%rcnt)
      allocate(sb(max(ns * na, 1)), rb(max(nrv * na, 1)))
      pos = 0
      do d = 0, pe_size - 1
         do a = 1, na
            do e = x%sdsp(d) + 1, x%sdsp(d) + x%scnt(d)
               pos = pos + 1
               sb(pos) = Fc(x%sl(e), act(a), x%sc(e))
            end do
         end do
      end do
      call MPI_Alltoallv(sb, na * x%scnt, na * x%sdsp, MPI_DOUBLE_PRECISION, rb, na * x%rcnt, na * x%rdsp, &
           MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
      pos = 0
      do d = 0, pe_size - 1
         do a = 1, na
            do e = x%rdsp(d) + 1, x%rdsp(d) + x%rcnt(d)
               pos = pos + 1
               Y(x%rcol(e), x%rrhs(e), a) = rb(pos)
            end do
         end do
      end do
      ! pure Neumann without mass (mode 0): mean-free right-hand side, the
      ! pinned node zero; the solution mean-free
      lp = 0
      if (scalar .and. self%singular .and. self%hkey(4) .eq. 0.0_dp) lp = pl%local_index(self%nfree)
      do a = 1, na
         sing = scalar .and. self%singular .and. self%hkey(4) .eq. 0.0_dp .and. self%lam(act(a)) .eq. 0.0_dp
         if (.not. sing) cycle
         do cc = 1, nr
            mean(cc) = sum(Y(1:pl%ncol, cc, a))
         end do
         call MPI_Allreduce(MPI_IN_PLACE, mean, nr, MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM, ierr)
         do cc = 1, nr
            Y(1:pl%ncol, cc, a) = Y(1:pl%ncol, cc, a) - mean(cc) / self%nfree
            if (lp .gt. 0) Y(lp, cc, a) = 0.0_dp
         end do
      end do
      call pl%sweep(na, act(1:na), nr, ld, Y)
      do a = 1, na
         sing = scalar .and. self%singular .and. self%hkey(4) .eq. 0.0_dp .and. self%lam(act(a)) .eq. 0.0_dp
         if (.not. sing) cycle
         do cc = 1, nr
            mean(cc) = sum(Y(1:pl%ncol, cc, a))
         end do
         call MPI_Allreduce(MPI_IN_PLACE, mean, nr, MPI_DOUBLE_PRECISION, MPI_SUM, NEKO_COMM, ierr)
         do cc = 1, nr
            Y(1:pl%ncol, cc, a) = Y(1:pl%ncol, cc, a) - mean(cc) / self%nfree
         end do
      end do
      ! back to Neko's side, in the same order
      pos = 0
      do d = 0, pe_size - 1
         do a = 1, na
            do e = x%rdsp(d) + 1, x%rdsp(d) + x%rcnt(d)
               pos = pos + 1
               rb(pos) = Y(x%rcol(e), x%rrhs(e), a)
            end do
         end do
      end do
      call MPI_Alltoallv(rb, na * x%rcnt, na * x%rdsp, MPI_DOUBLE_PRECISION, sb, na * x%scnt, na * x%sdsp, &
           MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
      pos = 0
      do d = 0, pe_size - 1
         do a = 1, na
            do e = x%sdsp(d) + 1, x%sdsp(d) + x%scnt(d)
               pos = pos + 1
               U(x%sl(e), act(a), x%sc(e)) = sb(pos)
            end do
         end do
      end do
    end subroutine plane_solve

    !> z(p) = C(l(p), iz(p) + 1)
    subroutine scatter(self, zz, nn)
      class(q2d_pc_t), intent(inout) :: self
      integer, intent(in) :: nn
      real(kind=rp), intent(inout) :: zz(nn)
      integer :: p
      do p = 1, nn
         zz(p) = real(self%C(self%pl(p), self%pz(p) + 1), rp)
      end do
    end subroutine scatter

  end subroutine q2d_apply

  !> the timers and balance of a plane (rank 0's view)
  subroutine plane_report(pl, name)
    type(plane_t), intent(in) :: pl
    character(len=*), intent(in) :: name
    character(len=320) :: log_buf
    if (.not. pl%ready) return
    write(log_buf, '(3A,4F9.3)') 'quasi2d: ', name, ' plane sweeps on rank 0 (s): local fwd, distributed fwd, '// &
         'distributed bwd, local bwd ', pl%tsw
    call neko_log%message(log_buf)
    write(log_buf, '(3A,3F9.3)') 'quasi2d: ', name, ' plane factorizations on rank 0 (s): values, local fronts, '// &
         'distributed fronts ', pl%tfac
    call neko_log%message(log_buf)
    write(log_buf, '(3A,I0,A,I0,A,I0,A,I0,A,I0)') 'quasi2d: ', name, ' plane: ordering ', pl%ordering_used, &
         '; G entries per rank max ', pl%ld_max, ', mean ', pl%ld_mean, '; distributed fronts per rank max ', &
         pl%nd_max, '; group depth ', pl%dmax
    call neko_log%message(log_buf)
  end subroutine plane_report

  subroutine mult_flat(n, a, z)
    integer, intent(in) :: n
    real(kind=rp), intent(in) :: a(n)
    real(kind=rp), intent(inout) :: z(n)
    z = z * a
  end subroutine mult_flat

  subroutine q2d_free(this)
    class(q2d_pc_t), intent(inout) :: this
    integer :: j
    character(len=320) :: log_buf
    if (this%ready .and. this%nsolve .gt. 0) then
       write(log_buf, '(A,I0,A,F9.3,A,I0,A,F9.3,A,I0)') 'quasi2d: ', this%nsolve, ' solves in ', &
            this%t_solve, ' s; ', this%nfact, ' factorizations in ', this%t_fact, ' s; skipped modes ', &
            this%nskip
       call neko_log%message(log_buf)
       write(log_buf, '(A,5F9.3)') 'quasi2d: phases (s): transform, setup/factor, plane solves, -, scatter ', &
            this%t_ph
       call neko_log%message(log_buf)
       write(log_buf, '(A,F9.3)') 'quasi2d: element matrices for the factorizations on rank 0 (s) ', this%t_asm
       call neko_log%message(log_buf)
       call plane_report(this%ps, 'scalar')
       call plane_report(this%puv, '(u, v)')
    end if
    this%ready = .false.
    call this%ps%free()
    call this%puv%free()
    this%xs%nc = 0; this%xuv%nc = 0; this%xw%nc = 0
    if (allocated(this%pl)) deallocate(this%pl)
    if (allocated(this%pz)) deallocate(this%pz)
    if (allocated(this%lrep)) deallocate(this%lrep)
    if (allocated(this%lglob)) deallocate(this%lglob)
    if (allocated(this%eax)) deallocate(this%eax)
    if (allocated(this%zinc)) deallocate(this%zinc)
    if (allocated(this%fid)) deallocate(this%fid)
    if (allocated(this%S)) deallocate(this%S)
    if (allocated(this%lam)) deallocate(this%lam)
    if (allocated(this%enode)) deallocate(this%enode)
    if (allocated(this%lenode)) deallocate(this%lenode, this%lgeo)
    if (allocated(this%leh)) deallocate(this%leh)
    if (allocated(this%lehf)) deallocate(this%lehf)
    if (allocated(this%ia)) deallocate(this%ia, this%ja, this%ia2, this%ja2)
    if (allocated(this%Dr)) deallocate(this%Dr, this%Ds)
    if (allocated(this%C)) deallocate(this%C, this%F, this%zt)
    this%hkey = -1.0_dp
  end subroutine q2d_free

  ! ---------------------------------------------------------------------------
  ! the solver type: PCG with the quasi2d preconditioner
  ! ---------------------------------------------------------------------------

  subroutine q2d_ksp_init(this, n, max_iter, M, rel_tol, abs_tol, monitor)
    class(q2d_ksp_t), target, intent(inout) :: this
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
  end subroutine q2d_ksp_init

  subroutine q2d_ksp_free(this)
    class(q2d_ksp_t), intent(inout) :: this
    call this%ksp_free()
    if (allocated(this%r)) deallocate(this%r, this%z, this%p, this%w)
  end subroutine q2d_ksp_free

  function q2d_pc(this) result(pc)
    class(q2d_ksp_t), intent(inout) :: this
    type(q2d_pc_t), pointer :: pc
    pc => null()
    if (.not. associated(this%M)) call neko_error('the "quasi2d" solver needs the "quasi2d" preconditioner')
    select type (q => this%M)
    type is (q2d_pc_t)
       pc => q
    class default
       call neko_error('the "quasi2d" solver needs the "quasi2d" preconditioner')
    end select
  end function q2d_pc

  !> PCG on one scalar field (x = 0 initially)
  function q2d_ksp_solve(this, Ax, x, f, n, coef, bc_projector, gs_h, niter) result(ksp_results)
    class(q2d_ksp_t), intent(inout) :: this
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(in) :: f
    type(coef_t), intent(inout) :: coef
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    integer, optional, intent(in) :: niter
    type(ksp_monitor_t) :: ksp_results
    type(q2d_pc_t), pointer :: pc
    integer :: iter, max_iter, i
    real(kind=rp) :: rnorm, rtz1, rtz2, beta, alpha, pap, nfac

    pc => q2d_pc(this)
    max_iter = this%max_iter
    if (present(niter)) max_iter = niter
    call ksp_work(this, n, 1)
    nfac = 1.0_rp / sqrt(coef%volume)
    associate(r => this%r(:, 1), z => this%z(:, 1), p => this%p(:, 1), w => this%w(:, 1))
      x%x = 0.0_rp
      r = f
      p = 0.0_rp
      rnorm = sqrt(glsc3(r, coef%mult, r, n)) * nfac
      ksp_results%res_start = rnorm
      ksp_results%res_final = rnorm
      ksp_results%iter = 0
      ksp_results%converged = .true.
      if (rnorm .eq. 0.0_rp) return
      this%ncall = this%ncall + 1
      if (.not. q2d_ksp_check(this, pc)) then
         ! the solve has been exact (one iteration) at every check: x = M f
         call pc%solve(z, r, n)
         call copy_flat(n, z, x%x)
         ksp_results%iter = 1
         ksp_results%res_final = rnorm * this%ratio
         return
      end if
      call this%monitor_start('quasi2d')
      rtz1 = 1.0_rp
      do iter = 1, max_iter
         call pc%solve(z, r, n)
         rtz2 = rtz1
         rtz1 = glsc3(r, coef%mult, z, n)
         beta = rtz1 / rtz2
         if (iter .eq. 1) beta = 0.0_rp
         p = z + beta * p
         call Ax%compute(w, p, coef, x%msh, x%Xh)
         call gs_h%op(w, n, GS_OP_ADD)
         call bc_projector%apply(w, n)
         pap = glsc3(w, coef%mult, p, n)
         alpha = rtz1 / pap
         call axpy_flat(n, alpha, p, x%x)
         r = r - alpha * w
         rnorm = sqrt(glsc3(r, coef%mult, r, n)) * nfac
         call this%monitor_iter(iter, rnorm)
         ksp_results%iter = iter
         ksp_results%res_final = rnorm
         if (rnorm .lt. this%abs_tol) exit
      end do
      ksp_results%converged = rnorm .lt. this%abs_tol
      call q2d_ksp_record(this, ksp_results%iter, ksp_results%res_start, rnorm)
      call this%monitor_stop()
    end associate
  end function q2d_ksp_solve

  !> PCG on the coupled velocity (full stress formulation), x = 0 initially
  function q2d_ksp_solve_coupled(this, Ax, x, y, z, fx, fy, fz, n, coef, bc_projector, gs_h, niter) &
       result(ksp_results)
    class(q2d_ksp_t), intent(inout) :: this
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
    type(q2d_pc_t), pointer :: pc
    integer :: iter, max_iter, c
    real(kind=rp) :: rnorm, rtz1, rtz2, beta, alpha, pap, nfac

    pc => q2d_pc(this)
    max_iter = this%max_iter
    if (present(niter)) max_iter = niter
    call ksp_work(this, n, 3)
    nfac = 1.0_rp / sqrt(coef%volume)
    associate(r => this%r, zz => this%z, p => this%p, w => this%w)
      if (this%stress .lt. 0) then
         ! does the operator couple the components (full stress)? apply it to
         ! (x y + x^2 / 2, 0, 0): the stress operator gives a nonzero v part
         call copy_flat(n, coef%dof%x%x, p(:, 1))
         call copy_flat(n, coef%dof%y%x, p(:, 2))
         p(:, 1) = p(:, 1) * p(:, 2) + 0.5_rp * p(:, 1)**2
         p(:, 2) = 0.0_rp; p(:, 3) = 0.0_rp
         call Ax%compute_vector(w(:, 1), w(:, 2), w(:, 3), p(:, 1), p(:, 2), p(:, 3), coef, x%msh, x%Xh)
         if (glsc3(w(:, 2), coef%mult, w(:, 2), n) .gt. 1.0e-20_rp * glsc3(w(:, 1), coef%mult, w(:, 1), n)) then
            this%stress = 1
            call neko_log%message('quasi2d: coupled solve with the full-stress operator')
         else
            this%stress = 0
            call neko_log%message('quasi2d: coupled solve with a segregated operator (three scalar problems)')
         end if
      end if
      x%x = 0.0_rp; y%x = 0.0_rp; z%x = 0.0_rp
      r(:, 1) = fx; r(:, 2) = fy; r(:, 3) = fz
      p = 0.0_rp
      rnorm = sqrt(dot3(r, r)) * nfac
      ksp_results%res_start = rnorm
      ksp_results%res_final = rnorm
      ksp_results%iter = 0
      ksp_results%converged = .true.
      if (rnorm .eq. 0.0_rp) return
      this%ncall = this%ncall + 1
      if (.not. q2d_ksp_check(this, pc)) then
         call pc%apply3(zz(:, 1), zz(:, 2), zz(:, 3), r(:, 1), r(:, 2), r(:, 3), n, this%stress .eq. 1)
         call copy_flat(n, zz(:, 1), x%x)
         call copy_flat(n, zz(:, 2), y%x)
         call copy_flat(n, zz(:, 3), z%x)
         ksp_results%iter = 1
         ksp_results%res_final = rnorm * this%ratio
         return
      end if
      call this%monitor_start('quasi2d (coupled)')
      rtz1 = 1.0_rp
      do iter = 1, max_iter
         call pc%apply3(zz(:, 1), zz(:, 2), zz(:, 3), r(:, 1), r(:, 2), r(:, 3), n, this%stress .eq. 1)
         rtz2 = rtz1
         rtz1 = dot3(r, zz)
         beta = rtz1 / rtz2
         if (iter .eq. 1) beta = 0.0_rp
         p = zz + beta * p
         call Ax%compute_vector(w(:, 1), w(:, 2), w(:, 3), p(:, 1), p(:, 2), p(:, 3), coef, x%msh, x%Xh)
         call gs_h%op(w(:, 1), w(:, 2), w(:, 3), n, GS_OP_ADD)
         call bc_projector%apply(w(:, 1), w(:, 2), w(:, 3), n)
         pap = dot3(w, p)
         alpha = rtz1 / pap
         call axpy_flat(n, alpha, p(:, 1), x%x)
         call axpy_flat(n, alpha, p(:, 2), y%x)
         call axpy_flat(n, alpha, p(:, 3), z%x)
         r = r - alpha * w
         rnorm = sqrt(dot3(r, r)) * nfac
         call this%monitor_iter(iter, rnorm)
         ksp_results%iter = iter
         ksp_results%res_final = rnorm
         if (rnorm .lt. this%abs_tol) exit
      end do
      ksp_results%converged = rnorm .lt. this%abs_tol
      call q2d_ksp_record(this, ksp_results(1)%iter, ksp_results(1)%res_start, rnorm)
      call this%monitor_stop()
    end associate

  contains

    real(kind=rp) function dot3(a, b)
      real(kind=rp), intent(in) :: a(:,:), b(:,:)
      integer :: cc
      dot3 = 0.0_rp
      do cc = 1, 3
         dot3 = dot3 + glsc3(a(:, cc), coef%mult, b(:, cc), n)
      end do
    end function dot3

  end function q2d_ksp_solve_coupled

  subroutine copy_flat(n, a, b)
    integer, intent(in) :: n
    real(kind=rp), intent(in) :: a(n)
    real(kind=rp), intent(out) :: b(n)
    b = a
  end subroutine copy_flat

  !> y = y + a x on flat arrays (y: a field's 4D array by sequence association)
  subroutine axpy_flat(n, a, x, y)
    integer, intent(in) :: n
    real(kind=rp), intent(in) :: a, x(n)
    real(kind=rp), intent(inout) :: y(n)
    y = y + a * x
  end subroutine axpy_flat

  !> whether this solve runs PCG with residuals ("residual": "always", the
  !! first ten solves, every 50th, and always once a checked solve needed more
  !! than one iteration); otherwise the solve is x = M f
  logical function q2d_ksp_check(this, pc) result(check)
    class(q2d_ksp_t), intent(inout) :: this
    type(q2d_pc_t), intent(in) :: pc
    check = pc%residual .eq. 'always' .or. .not. this%exact .or. this%ncall .le. 10 .or. &
         mod(this%ncall, 50) .eq. 0
  end function q2d_ksp_check

  subroutine q2d_ksp_record(this, iter, r0, r1)
    class(q2d_ksp_t), intent(inout) :: this
    integer, intent(in) :: iter
    real(kind=rp), intent(in) :: r0, r1
    if (iter .gt. 1) then
       if (this%exact) call neko_log%message('quasi2d: a solve needed more than one iteration; ' // &
            'residuals are checked at every solve from now on')
       this%exact = .false.
    end if
    if (r0 .gt. 0.0_rp) this%ratio = r1 / r0
  end subroutine q2d_ksp_record

  subroutine ksp_work(this, n, nc)
    class(q2d_ksp_t), intent(inout) :: this
    integer, intent(in) :: n, nc
    if (allocated(this%r)) then
       if (size(this%r, 2) .ge. nc .and. size(this%r, 1) .eq. n) return
       deallocate(this%r, this%z, this%p, this%w)
    end if
    allocate(this%r(n, nc), this%z(n, nc), this%p(n, nc), this%w(n, nc))
    this%r = 0.0_rp; this%z = 0.0_rp; this%p = 0.0_rp; this%w = 0.0_rp
  end subroutine ksp_work

  ! ---------------------------------------------------------------------------
  ! helpers
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
    if (info .ne. 0) call neko_error('quasi2d: dsyev failed')
    do j = 1, nz
       do i = 1, nz
          S(i, j) = A(i, j) / sqrt(wz(i))
       end do
    end do
    if (abs(lam(1)) .lt. 1.0e-10_dp * abs(lam(nz))) lam(1) = 0.0_dp
  end subroutine gen_eig

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
    call neko_error('quasi2d: column not in the pattern')
  end function find_col

  integer function find_i8(a, n, v)
    integer(kind=i8), intent(in) :: a(:), v
    integer, intent(in) :: n
    integer :: lo, hi, mid
    lo = 1; hi = n
    do while (lo .le. hi)
       mid = (lo + hi) / 2
       if (a(mid) .lt. v) then
          lo = mid + 1
       else if (a(mid) .gt. v) then
          hi = mid - 1
       else
          find_i8 = mid
          return
       end if
    end do
    find_i8 = 0
    call neko_error('quasi2d: node id not found')
  end function find_i8

  subroutine isort(a)
    integer, intent(inout) :: a(:)
    integer :: i, j, v
    do i = 2, size(a)
       v = a(i)
       j = i - 1
       do while (j .ge. 1)
          if (a(j) .le. v) exit
          a(j + 1) = a(j)
          j = j - 1
       end do
       a(j + 1) = v
    end do
  end subroutine isort

  !> idx such that key(idx) is ascending (heapsort on the index)
  subroutine sort_key_index(key, idx, n)
    integer, intent(in) :: n
    integer(kind=i8), intent(in) :: key(:)
    integer, intent(inout) :: idx(:)
    integer :: i, t
    do i = n / 2, 1, -1
       call sift(i, n)
    end do
    do i = n, 2, -1
       t = idx(1); idx(1) = idx(i); idx(i) = t
       call sift(1, i - 1)
    end do
  contains
    subroutine sift(lo, hi)
      integer, intent(in) :: lo, hi
      integer :: root, child, tt
      root = lo
      do while (2 * root .le. hi)
         child = 2 * root
         if (child .lt. hi) then
            if (key(idx(child)) .lt. key(idx(child + 1))) child = child + 1
         end if
         if (key(idx(root)) .lt. key(idx(child))) then
            tt = idx(root); idx(root) = idx(child); idx(child) = tt
            root = child
         else
            return
         end if
      end do
    end subroutine sift
  end subroutine sort_key_index

end module quasi2d
