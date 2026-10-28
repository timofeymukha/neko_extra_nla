!> A sparse SPD 2D "plane" matrix factored and solved by all ranks together
!! (quasi2d, logbooks 2026-10-26 and 2026-10-27).
!!
!! Rank 0 orders the matrix and does the symbolic factorization (q2d_spchol).
!! The supernodal elimination tree is mapped onto the ranks subtree-to-subcube
!! (proportional mapping by factor storage): a supernode gets a contiguous
!! group of ranks; the children of a supernode split its group in two, in
!! proportion to their weights, recursively; a group of one rank takes the
!! whole subtree ("local" supernodes). A supernode with a group of two or more
!! ranks ("distributed") has its dense front distributed over the group by
!! rows, block-cyclically in blocks of NB rows.
!!
!! Numeric factorization (multifrontal, all ranks): rank 0 scatters the matrix
!! values to the owners of their front positions; every rank factors its local
!! subtrees; the distributed fronts are assembled by an all-to-all of the
!! children's update rows within the group, then factored by a right-looking
!! blocked Cholesky (diagonal block and panel all-gathered), and turned into
!! the G form [L11^{-1}; L21 L11^{-1}] by streaming the rows of L11^{-1}. Every
!! rank keeps the G rows it owns (transposed, k x own rows).
!!
!! Solve (all ranks): forward over the supernodes in postorder (local: the G
!! block times the pivot part; distributed: an all-reduce of the front's
!! right-hand side over the group, then each rank its rows), backward in
!! reverse (distributed: each rank G^T times its rows, all-reduce over the
!! group). The right-hand side arrives, and the solution leaves, in the
!! owners' local column numbering (the caller maps plane dofs with own_old and
!! local_index). Several modes (factorizations with the same pattern) and
!! several right-hand sides per mode are solved in one pass.
module q2d_plane
  use num_types, only : dp, i8
  use q2d_spchol, only : spchol_t
  use comm, only : NEKO_COMM, pe_rank, pe_size
  use mpi_f08, only : MPI_Comm, MPI_Bcast, MPI_Scatter, MPI_Scatterv, MPI_Allreduce, MPI_Alltoall, &
       MPI_Alltoallv, MPI_Allgatherv, MPI_Comm_split, MPI_Comm_free, MPI_Reduce, MPI_INTEGER, &
       MPI_DOUBLE_PRECISION, MPI_SUM, MPI_MAX, MPI_IN_PLACE, MPI_UNDEFINED, MPI_Wtime
  use utils, only : neko_error
  implicit none
  private

  !> rows per block of the distributed fronts
  integer, parameter :: NB = 32
  !> mapping: the largest accepted imbalance of a two-way split
  real(kind=dp), public :: q2d_split_tol = 1.1_dp
  !> mapping: groups of at most this many ranks take their subtrees by LPT,
  !! with at most this imbalance
  integer, public :: q2d_lpt_group = 16
  real(kind=dp), public :: q2d_lpt_tol = 1.05_dp
  !> profiling of the distributed fronts: assembly, factorization, G rows
  real(kind=dp), public :: q2d_tdist(3) = 0.0_dp

  type :: gblk_t
     real(kind=dp), allocatable :: v(:)
  end type gblk_t

  type :: umat_t
     real(kind=dp), allocatable :: u(:,:)
  end type umat_t

  type, public :: plane_t
     integer :: n = 0                          !< order (plane dofs, "old" numbering)
     integer :: nmodes = 0
     logical :: ready = .false.
     integer(kind=i8) :: nnz_l = 0
     !> ordering (q2d_spchol: 0 auto, 1 nested dissection, 2 AMD) and the one used
     integer :: ordering = 1
     integer :: ordering_used = 0
     !> every rank: its supernodes in postorder
     integer :: nsup = 0, ncol = 0, next = 0, dmax = 0
     integer, allocatable :: sk(:), sm(:), sg0(:), sgs(:), sdep(:), spar(:)
     integer, allocatable :: nown(:), nlow(:), c0(:), snb(:)
     integer, allocatable :: roff(:), srow(:)  !< the rows of every front, as local slots
     integer, allocatable :: cmo(:), cmap(:)   !< rows k+1..m of every front: positions in the parent's front
     integer(kind=i8), allocatable :: boff(:)  !< this rank's G rows of every front (nown x k)
     integer, allocatable :: lhead(:), lnext(:) !< children of the local supernodes
     !> the children of the distributed supernodes: k, m, group, the local
     !! index here (0: not held here), positions in the parent's front
     integer, allocatable :: choff(:), chk(:), chm(:), chg0(:), chgs(:), chh(:), chmo(:), chmap(:)
     !> the matrix entries of every front held here (in the columns this rank
     !! owns): position (jc - 1) m + fr; pin: 1 the pinned diagonal, 2 the
     !! rest of its row and column
     integer, allocatable :: aoff(:), apos(:), apin(:)
     !> the contributions (contrib_setup): send position of every own one,
     !! counts and displacements, and the entry of every received one (0: none)
     integer, allocatable :: cspos(:), cscnt(:), csdsp(:), crcnt(:), crdsp(:), rmap(:)
     logical :: contrib = .false.
     integer, allocatable :: cold(:)           !< plane dof of every local column
     integer, allocatable :: csort(:), cpos(:) !< sorted cold and the local index of each
     integer, allocatable :: csup(:), cjc(:)   !< supernode and front column of every local column
     integer, allocatable :: sold(:), ssort(:), sspos(:) !< plane dof of every slot; sorted, and the slot of each
     integer, allocatable :: own_old(:)        !< every rank: owner of every plane dof
     !> one communicator per depth of the rank groups (have: this rank is in one)
     type(MPI_Comm), allocatable :: dcomm(:)
     logical, allocatable :: have(:)
     !> per mode: this rank's G rows; current or not
     type(gblk_t), allocatable :: g(:)
     logical, allocatable :: fresh(:)
     real(kind=dp), allocatable :: t(:,:), zb(:)
     !> timers: sweeps (local fwd, distributed fwd, distributed bwd, local
     !! bwd); factorization (values, local fronts, distributed fronts)
     real(kind=dp) :: tsw(4) = 0.0_dp, tfac(3) = 0.0_dp
     real(kind=dp) :: t_factor = 0.0_dp
     !> balance (rank 0): stored G entries, max and mean over the ranks;
     !! distributed supernodes on the busiest rank
     integer(kind=i8) :: ld_max = 0, ld_mean = 0
     integer :: nd_max = 0
   contains
     procedure, pass(this) :: setup => plane_setup
     procedure, pass(this) :: contrib_setup => plane_contrib_setup
     procedure, pass(this) :: factor => plane_factor
     procedure, pass(this) :: sweep => plane_sweep
     procedure, pass(this) :: local_index => plane_local_index
     procedure, pass(this) :: free => plane_free
     procedure, pass(this) :: dry_map => plane_dry_map
  end type plane_t

contains

  ! ---------------------------------------------------------------------------
  ! the block-cyclic row distribution of a front over a group of g ranks
  ! (g = 1: the identity)
  ! ---------------------------------------------------------------------------

  !> rows per block of a front of m rows distributed over g ranks
  pure integer function rowblk(m, g)
    integer, intent(in) :: m, g
    rowblk = NB
    if (g .gt. 1) rowblk = max(1, min(NB, m / (4 * g)))
  end function rowblk

  !> group rank owning front row r (blocks of nb rows)
  pure integer function rown(r, g, nb)
    integer, intent(in) :: r, g, nb
    rown = mod((r - 1) / nb, g)
  end function rown

  !> local index of front row r on its owner
  pure integer function rloc(r, g, nb)
    integer, intent(in) :: r, g, nb
    rloc = ((r - 1) / nb / g) * nb + mod(r - 1, nb) + 1
  end function rloc

  !> front row of local row l of group rank q
  pure integer function rrow(l, q, g, nb)
    integer, intent(in) :: l, q, g, nb
    rrow = (((l - 1) / nb) * g + q) * nb + mod(l - 1, nb) + 1
  end function rrow

  !> number of front rows <= x owned by group rank q
  pure integer function ncnt(x, q, g, nb)
    integer, intent(in) :: x, q, g, nb
    integer :: nf
    ncnt = 0
    if (x .le. 0) return
    nf = x / nb
    if (q .lt. nf) ncnt = ((nf - 1 - q) / g + 1) * nb
    if (mod(nf, g) .eq. q) ncnt = ncnt + (x - nf * nb)
  end function ncnt

  ! ---------------------------------------------------------------------------
  ! setup
  ! ---------------------------------------------------------------------------

  !> Ordering, symbolic factorization and mapping on rank 0 (ia, ja: the CSR
  !! pattern of the matrix, both triangles, on rank 0 only), then the
  !! structures to every rank and the group communicators. Collective.
  subroutine plane_setup(this, n, nmodes, ia, ja)
    class(plane_t), intent(inout) :: this
    integer, intent(in) :: n, nmodes
    integer, intent(in) :: ia(:), ja(:)
    integer :: P, ierr, cnt, pos, i, j, q, d, color, nch, ncmap, nrecv, nslot, l
    integer, allocatable :: buf(:), scnt(:), sdsp(:), ib(:), colors(:)
    integer(kind=i8) :: ld
    call this%free()
    this%n = n
    this%nmodes = nmodes
    P = pe_size
    allocate(this%own_old(n), scnt(0:P - 1), sdsp(0:P - 1))
    if (pe_rank .eq. 0) then
       call setup_root(this, n, ia, ja, P, buf, scnt, sdsp)
    else
       allocate(buf(1))
       scnt = 0; sdsp = 0
    end if
    call MPI_Bcast(this%own_old, n, MPI_INTEGER, 0, NEKO_COMM, ierr)
    call MPI_Bcast(this%nnz_l, 1, MPI_INTEGER8_(), 0, NEKO_COMM, ierr)
    ! the structures
    call MPI_Scatter(scnt, 1, MPI_INTEGER, nrecv, 1, MPI_INTEGER, 0, NEKO_COMM, ierr)
    allocate(ib(max(nrecv, 1)))
    call MPI_Scatterv(buf, scnt, sdsp, MPI_INTEGER, ib, nrecv, MPI_INTEGER, 0, NEKO_COMM, ierr)
    deallocate(buf)
    ! parse: [nsup, ncol, nslot, dmax, colors(0:dmax),
    !         per supernode (k, m, g0, gs, dep, par, nch),
    !         rows (slots), cmap (supernodes with a parent),
    !         children of the distributed supernodes (k, m, g0, gs, held, cmap),
    !         plane dof of every slot (own columns first)]
    this%nsup = ib(1); this%ncol = ib(2); this%next = ib(3) - ib(2); this%dmax = ib(4)
    nslot = ib(3)
    pos = 4
    allocate(colors(0:this%dmax))
    colors = ib(pos + 1:pos + this%dmax + 1)
    pos = pos + this%dmax + 1
    associate(ns => this%nsup)
      allocate(this%sk(ns), this%sm(ns), this%sg0(ns), this%sgs(ns), this%sdep(ns), this%spar(ns), &
           this%nown(ns), this%nlow(ns), this%c0(ns), this%snb(ns), this%roff(ns + 1), this%cmo(ns + 1), &
           this%boff(ns + 1), this%choff(ns + 1), this%aoff(ns + 1), this%lhead(ns), this%lnext(ns))
      this%roff(1) = 1; this%cmo(1) = 1; this%boff(1) = 0; this%choff(1) = 0; this%aoff = 0
      cnt = 0
      do i = 1, ns
         this%sk(i) = ib(pos + 1); this%sm(i) = ib(pos + 2); this%sg0(i) = ib(pos + 3)
         this%sgs(i) = ib(pos + 4); this%sdep(i) = ib(pos + 5); this%spar(i) = ib(pos + 6)
         nch = ib(pos + 7)
         pos = pos + 7
         q = pe_rank - this%sg0(i)
         this%snb(i) = rowblk(this%sm(i), this%sgs(i))
         this%nown(i) = ncnt(this%sm(i), q, this%sgs(i), this%snb(i))
         this%nlow(i) = ncnt(this%sk(i), q, this%sgs(i), this%snb(i))
         this%c0(i) = cnt + 1
         cnt = cnt + this%nlow(i)
         this%roff(i + 1) = this%roff(i) + this%sm(i)
         ncmap = 0
         if (this%spar(i) .ne. 0) ncmap = this%sm(i) - this%sk(i)
         this%cmo(i + 1) = this%cmo(i) + ncmap
         this%boff(i + 1) = this%boff(i) + int(this%nown(i), i8) * this%sk(i)
         this%choff(i + 1) = this%choff(i) + nch
      end do
      if (cnt .ne. this%ncol) call neko_error('quasi2d: plane setup, own column count mismatch')
      allocate(this%srow(this%roff(ns + 1) - 1))
      this%srow = ib(pos + 1:pos + size(this%srow))
      pos = pos + size(this%srow)
      allocate(this%cmap(max(this%cmo(ns + 1) - 1, 1)))
      this%cmap(1:this%cmo(ns + 1) - 1) = ib(pos + 1:pos + this%cmo(ns + 1) - 1)
      pos = pos + this%cmo(ns + 1) - 1
      j = this%choff(ns + 1)
      allocate(this%chk(j), this%chm(j), this%chg0(j), this%chgs(j), this%chh(j), this%chmo(j + 1))
      this%chmo(1) = 1
      ! children: descriptors and the sizes of their maps, then the maps
      q = pos
      do j = 1, this%choff(ns + 1)
         this%chk(j) = ib(q + 1); this%chm(j) = ib(q + 2); this%chg0(j) = ib(q + 3)
         this%chgs(j) = ib(q + 4); this%chh(j) = ib(q + 5)
         ncmap = this%chm(j) - this%chk(j)
         this%chmo(j + 1) = this%chmo(j) + ncmap
         q = q + 5 + ncmap
      end do
      allocate(this%chmap(max(this%chmo(j) - 1, 1)))
      do j = 1, this%choff(ns + 1)
         ncmap = this%chm(j) - this%chk(j)
         this%chmap(this%chmo(j):this%chmo(j + 1) - 1) = ib(pos + 6:pos + 5 + ncmap)
         pos = pos + 5 + ncmap
      end do
      allocate(this%sold(max(nslot, 1)), this%cold(this%ncol))
      this%sold(1:nslot) = ib(pos + 1:pos + nslot)
      this%cold = this%sold(1:this%ncol)
      pos = pos + nslot
      if (pos .ne. nrecv) call neko_error('quasi2d: plane setup, structure size mismatch')
      ! children lists of the local supernodes (ascending)
      this%lhead = 0; this%lnext = 0
      do i = ns, 1, -1
         if (this%spar(i) .ne. 0) then
            if (this%sgs(this%spar(i)) .eq. 1) then
               this%lnext(i) = this%lhead(this%spar(i)); this%lhead(this%spar(i)) = i
            end if
         end if
      end do
      ! the local supernodes' pivots are their consecutive own columns
      do i = 1, ns
         if (this%sgs(i) .ne. 1) cycle
         do j = 1, this%sk(i)
            if (this%srow(this%roff(i) + j - 1) .ne. this%c0(i) + j - 1) &
                 call neko_error('quasi2d: plane setup, local pivots not consecutive')
         end do
      end do
    end associate
    ! the group communicators, one per depth
    allocate(this%dcomm(0:this%dmax), this%have(0:this%dmax))
    do d = 0, this%dmax
       color = colors(d)
       this%have(d) = color .ge. 0
       if (.not. this%have(d)) color = MPI_UNDEFINED
       call MPI_Comm_split(NEKO_COMM, color, pe_rank, this%dcomm(d), ierr)
    end do
    ! old -> local column, old -> slot; supernode and front column of every local column
    allocate(this%csort(this%ncol), this%cpos(this%ncol))
    do i = 1, this%ncol
       this%cpos(i) = i
    end do
    this%csort = this%cold
    call sort_pair(this%csort, this%cpos, this%ncol)
    allocate(this%ssort(max(nslot, 1)), this%sspos(max(nslot, 1)))
    do i = 1, nslot
       this%sspos(i) = i
    end do
    this%ssort(1:nslot) = this%sold(1:nslot)
    call sort_pair(this%ssort, this%sspos, nslot)
    allocate(this%csup(this%ncol), this%cjc(this%ncol))
    do i = 1, this%nsup
       q = pe_rank - this%sg0(i)
       do l = 1, this%nlow(i)
          this%csup(this%c0(i) + l - 1) = i
          this%cjc(this%c0(i) + l - 1) = rrow(l, q, this%sgs(i), this%snb(i))
       end do
    end do
    allocate(this%g(nmodes), this%fresh(nmodes))
    this%fresh = .false.
    allocate(this%t(1, 1), this%zb(1))
    ! balance
    ld = this%boff(this%nsup + 1)
    call MPI_Reduce(ld, this%ld_max, 1, MPI_INTEGER8_(), MPI_MAX, 0, NEKO_COMM, ierr)
    call MPI_Reduce(ld, this%ld_mean, 1, MPI_INTEGER8_(), MPI_SUM, 0, NEKO_COMM, ierr)
    this%ld_mean = this%ld_mean / P
    cnt = count(this%sgs(1:this%nsup) .gt. 1)
    call MPI_Reduce(cnt, this%nd_max, 1, MPI_INTEGER, MPI_MAX, 0, NEKO_COMM, ierr)
    this%ready = .true.
  end subroutine plane_setup

  !> rank 0: ordering, symbolic factorization, mapping, and every rank's
  !! structure (buf, scnt, sdsp)
  subroutine setup_root(this, n, ia, ja, P, buf, scnt, sdsp, loads)
    class(plane_t), intent(inout) :: this
    integer, intent(in) :: n, P
    integer, intent(in) :: ia(:), ja(:)
    integer, allocatable, intent(inout) :: buf(:)
    integer, intent(inout) :: scnt(0:P - 1), sdsp(0:P - 1)
    integer(kind=i8), intent(out), optional :: loads(0:3 * P - 1)
    type(spchol_t) :: sc
    integer :: ns, s, c, r, i, j, k, q, nroot, fr, nb, nsr, ncol, nslot, ntot
    integer, allocatable :: kk(:), mm(:), nsz(:), head(:), nxt(:), roots(:), cmapg(:), pos(:)
    integer, allocatable :: g0(:), gs(:), dep(:), colr(:,:), rlo(:), rl(:), efill(:), lidx(:), loc(:), sold(:)
    integer(kind=i8), allocatable :: Wt(:)
    integer :: dmax
    ! ordering and symbolic factorization
    sc%ordering = this%ordering
    call sc%analyse(n, ia, ja)
    this%nnz_l = sc%nnz_l
    this%ordering_used = sc%ordering_used
    ns = sc%nsuper
    allocate(kk(ns), mm(ns), Wt(ns), nsz(ns), head(ns), nxt(ns))
    do s = 1, ns
       kk(s) = sc%sfc(s + 1) - sc%sfc(s)
       mm(s) = sc%srp(s + 1) - sc%srp(s)
    end do
    ! subtree weights (factor storage) and sizes (postorder: children first)
    Wt = 0; nsz = 0
    do s = 1, ns
       Wt(s) = Wt(s) + (sc%slp(s + 1) - sc%slp(s))
       nsz(s) = nsz(s) + 1
       if (sc%sparent(s) .ne. 0) then
          Wt(sc%sparent(s)) = Wt(sc%sparent(s)) + Wt(s)
          nsz(sc%sparent(s)) = nsz(sc%sparent(s)) + nsz(s)
       end if
    end do
    ! children lists, ascending
    head = 0; nxt = 0
    do s = ns, 1, -1
       if (sc%sparent(s) .ne. 0) then
          nxt(s) = head(sc%sparent(s)); head(sc%sparent(s)) = s
       end if
    end do
    nroot = count(sc%sparent .eq. 0)
    allocate(roots(nroot))
    nroot = 0
    do s = 1, ns
       if (sc%sparent(s) .eq. 0) then
          nroot = nroot + 1; roots(nroot) = s
       end if
    end do
    ! positions of the rows k+1..m of every front in its parent's front
    allocate(cmapg(max(sc%srp(ns + 1) - 1, 1)), pos(n))
    cmapg = 0; pos = 0
    do s = 1, ns
       do j = 1, mm(s)
          pos(sc%srow(sc%srp(s) + j - 1)) = j
       end do
       c = head(s)
       do while (c .ne. 0)
          do j = sc%srp(c) + kk(c), sc%srp(c + 1) - 1
             cmapg(j) = pos(sc%srow(j))
             if (sc%srow(sc%srp(s) + cmapg(j) - 1) .ne. sc%srow(j)) &
                  call neko_error('quasi2d: plane setup, a child row is not in its parent')
          end do
          c = nxt(c)
       end do
    end do
    ! the mapping
    allocate(g0(ns), gs(ns), dep(ns), colr(0:P - 1, 0:min(P, 100)))
    colr = -1; dmax = 0
    call assign(roots, 0, P, 0)
    this%dmax = dmax
    ! every rank's supernodes (ascending)
    allocate(rlo(0:P))
    rlo = 0
    do s = 1, ns
       do r = g0(s), g0(s) + gs(s) - 1
          rlo(r + 1) = rlo(r + 1) + 1
       end do
    end do
    do r = 1, P
       rlo(r) = rlo(r) + rlo(r - 1)
    end do
    ntot = rlo(P)
    allocate(rl(ntot), efill(0:P - 1))
    efill = 0
    do s = 1, ns
       do r = g0(s), g0(s) + gs(s) - 1
          efill(r) = efill(r) + 1
          rl(rlo(r) + efill(r)) = s
       end do
    end do
    ! every rank's structure
    allocate(buf(1024), lidx(ns), loc(n), sold(n))
    nb = 0
    lidx = 0; loc = 0
    do r = 0, P - 1
       sdsp(r) = nb
       nsr = rlo(r + 1) - rlo(r)
       do i = 1, nsr
          lidx(rl(rlo(r) + i)) = i
       end do
       ! own columns: local pivots, own pivot rows of the distributed fronts
       ncol = 0
       do i = 1, nsr
          s = rl(rlo(r) + i)
          q = r - g0(s)
          do fr = 1, kk(s)
             if (rown(fr, gs(s), rowblk(mm(s), gs(s))) .ne. q) cycle
             j = sc%srow(sc%srp(s) + fr - 1)
             ncol = ncol + 1
             loc(j) = ncol
             sold(ncol) = sc%perm(j)
             this%own_old(sc%perm(j)) = r
          end do
       end do
       nslot = ncol
       do i = 1, nsr
          s = rl(rlo(r) + i)
          do j = sc%srp(s), sc%srp(s + 1) - 1
             if (loc(sc%srow(j)) .eq. 0) then
                nslot = nslot + 1
                loc(sc%srow(j)) = nslot
                sold(nslot) = sc%perm(sc%srow(j))
             end if
          end do
       end do
       if (present(loads)) then
          loads(r) = 0; loads(P + r) = 0; loads(2 * P + r) = 0
          do i = 1, nsr
             s = rl(rlo(r) + i)
             if (gs(s) .eq. 1) loads(r) = loads(r) + int(mm(s), i8) * kk(s)
             if (gs(s) .gt. 1) loads(2 * P + r) = loads(2 * P + r) + 1
             if (gs(s) .gt. 1) loads(P + r) = loads(P + r) + &
                  int(ncnt(mm(s), r - g0(s), gs(s), rowblk(mm(s), gs(s))), i8) * kk(s)
          end do
       end if
       call push([nsr, ncol, nslot, dmax])
       call push(colr(r, 0:dmax))
       do i = 1, nsr
          s = rl(rlo(r) + i)
          k = 0
          if (sc%sparent(s) .ne. 0) then
             k = lidx(sc%sparent(s))
             if (k .eq. 0) call neko_error('quasi2d: plane setup, a parent is not on the rank')
          end if
          c = 0
          if (gs(s) .gt. 1) then
             j = head(s)
             do while (j .ne. 0)
                c = c + 1; j = nxt(j)
             end do
          end if
          call push([kk(s), mm(s), g0(s), gs(s), dep(s), k, c])
       end do
       do i = 1, nsr
          s = rl(rlo(r) + i)
          call push(loc(sc%srow(sc%srp(s):sc%srp(s + 1) - 1)))
       end do
       do i = 1, nsr
          s = rl(rlo(r) + i)
          if (sc%sparent(s) .ne. 0) call push(cmapg(sc%srp(s) + kk(s):sc%srp(s + 1) - 1))
       end do
       do i = 1, nsr
          s = rl(rlo(r) + i)
          if (gs(s) .eq. 1) cycle
          c = head(s)
          do while (c .ne. 0)
             call push([kk(c), mm(c), g0(c), gs(c), lidx(c)])
             call push(cmapg(sc%srp(c) + kk(c):sc%srp(c + 1) - 1))
             c = nxt(c)
          end do
       end do
       call push(sold(1:nslot))
       scnt(r) = nb - sdsp(r)
       ! reset
       do i = 1, nsr
          s = rl(rlo(r) + i)
          lidx(s) = 0
          do j = sc%srp(s), sc%srp(s + 1) - 1
             loc(sc%srow(j)) = 0
          end do
       end do
    end do
    call sc%free()

  contains

    !> append v to buf
    subroutine push(v)
      integer, intent(in) :: v(:)
      integer, allocatable :: tmp(:)
      if (nb + size(v) .gt. size(buf)) then
         allocate(tmp(max(2 * size(buf), nb + size(v))))
         tmp(1:nb) = buf(1:nb)
         call move_alloc(tmp, buf)
      end if
      buf(nb + 1:nb + size(v)) = v
      nb = nb + size(v)
    end subroutine push

    !> proportional mapping of the subtrees of C onto ranks a0 .. a0 + gg - 1.
    !! One subtree root takes the whole group, and its children are mapped the
    !! same way. In a group of more than q2d_lpt_group ranks, several are
    !! split into two sets of about equal weight, with the group split in
    !! proportion; if the rounding to whole ranks leaves the halves more than
    !! q2d_split_tol apart, the heaviest root takes the whole group (its
    !! children join the candidates) and the split is tried again. In a
    !! smaller group the candidates go to single ranks, largest first to the
    !! least loaded (LPT), after expanding the heaviest the same way until the
    !! LPT loads are within q2d_lpt_tol (or 8 candidates per rank).
    recursive subroutine assign(C0, a0, gg, d)
      integer, intent(in) :: C0(:), a0, gg, d
      integer, allocatable :: C(:), c1(:), c2(:), ord(:)
      integer(kind=i8), allocatable :: rl_(:)
      integer :: ii, ss, n1, n2, g1, tt, ih, rr
      integer(kind=i8) :: w1, w2, wc
      real(kind=dp) :: imb
      if (size(C0) .eq. 0) return
      if (gg .eq. 1) then
         call to_rank(C0, a0, d)
         return
      end if
      if (d .gt. ubound(colr, 2)) call neko_error('quasi2d: plane setup, rank groups nested too deep')
      if (any(colr(a0:a0 + gg - 1, d) .ne. -1 .and. colr(a0:a0 + gg - 1, d) .ne. a0)) &
           call neko_error('quasi2d: plane setup, overlapping rank groups')
      colr(a0:a0 + gg - 1, d) = a0
      dmax = max(dmax, d)
      C = C0
      do
         if (size(C) .eq. 0) return
         ih = maxloc(Wt(C), 1)
         if (size(C) .gt. 1) then
            if (allocated(ord)) deallocate(ord)
            if (allocated(c1)) deallocate(c1)
            if (allocated(c2)) deallocate(c2)
            allocate(ord(size(C)), c1(size(C)), c2(size(C)))
            do ii = 1, size(C)
               ord(ii) = ii
            end do
            call sort_desc(Wt, C, ord, size(C))
            wc = sum(Wt(C))
            if (gg .le. q2d_lpt_group) then
               ! small group: LPT; expand while its imbalance exceeds q2d_lpt_tol
               if (allocated(rl_)) deallocate(rl_)
               allocate(rl_(0:gg - 1))
               rl_ = 0
               do ii = 1, size(C)
                  rr = minloc(rl_, 1) - 1
                  rl_(rr) = rl_(rr) + Wt(C(ord(ii)))
               end do
               imb = real(maxval(rl_), dp) * gg / max(real(wc, dp), 1.0_dp)
               if (imb .le. q2d_lpt_tol .or. size(C) .ge. 8 * gg .or. head(C(ih)) .eq. 0) then
                  rl_ = 0
                  do ii = 1, size(C)
                     ss = C(ord(ii))
                     rr = minloc(rl_, 1) - 1
                     rl_(rr) = rl_(rr) + Wt(ss)
                     call to_rank([ss], a0 + rr, d + 1)
                  end do
                  return
               end if
            else
               ! two sets of about equal weight (largest first to the lighter set)
               n1 = 0; n2 = 0; w1 = 0; w2 = 0
               do ii = 1, size(C)
                  ss = C(ord(ii))
                  if (w1 .le. w2) then
                     n1 = n1 + 1; c1(n1) = ss; w1 = w1 + Wt(ss)
                  else
                     n2 = n2 + 1; c2(n2) = ss; w2 = w2 + Wt(ss)
                  end if
               end do
               g1 = nint(real(gg, dp) * real(w1, dp) / max(real(w1 + w2, dp), 1.0_dp))
               g1 = max(1, min(gg - 1, g1))
               imb = max(real(w1, dp) / g1, real(w2, dp) / (gg - g1)) * gg / max(real(w1 + w2, dp), 1.0_dp)
               if (imb .le. q2d_split_tol .or. head(C(ih)) .eq. 0) then
                  call assign(c1(1:n1), a0, g1, d + 1)
                  call assign(c2(1:n2), a0 + g1, gg - g1, d + 1)
                  return
               end if
            end if
         end if
         ! the heaviest root takes the whole group, its children join
         ss = C(ih)
         g0(ss) = a0; gs(ss) = gg; dep(ss) = d
         c1 = [C(1:ih - 1), C(ih + 1:size(C))]
         tt = head(ss)
         do while (tt .ne. 0)
            c1 = [c1, tt]; tt = nxt(tt)
         end do
         call move_alloc(c1, C)
      end do
    end subroutine assign

    !> the whole subtrees of C to rank r0
    subroutine to_rank(C, r0, d)
      integer, intent(in) :: C(:), r0, d
      integer :: ii, ss
      do ii = 1, size(C)
         ss = C(ii)
         g0(ss - nsz(ss) + 1:ss) = r0
         gs(ss - nsz(ss) + 1:ss) = 1
         dep(ss - nsz(ss) + 1:ss) = d
      end do
    end subroutine to_rank

  end subroutine setup_root

  !> rank-local test of the mapping for P ranks: G entries per rank
  subroutine plane_dry_map(this, n, ia, ja, P, loads)
    class(plane_t), intent(inout) :: this
    integer, intent(in) :: n, P
    integer, intent(in) :: ia(:), ja(:)
    integer(kind=i8), intent(out) :: loads(0:3 * P - 1)
    integer, allocatable :: buf(:)
    integer :: scnt(0:P - 1), sdsp(0:P - 1)
    call this%free()
    allocate(this%own_old(n))
    call setup_root(this, n, ia, ja, P, buf, scnt, sdsp, loads)
    call this%free()
  end subroutine plane_dry_map

  function MPI_INTEGER8_() result(t)
    use mpi_f08, only : MPI_Datatype, MPI_INTEGER8
    type(MPI_Datatype) :: t
    t = MPI_INTEGER8
  end function MPI_INTEGER8_

  ! ---------------------------------------------------------------------------
  ! numeric factorization
  ! ---------------------------------------------------------------------------

  !> The pattern of the matrix contributions of this rank: the i-th one adds
  !! to entry (crow(i), ccol(i)) (plane dofs; any number of contributions to
  !! an entry, from any ranks; both triangles, each triangle entry is used
  !! from its lower-triangle copy). Every contribution goes to the owner of
  !! its column, which keeps the entries of its fronts' columns. pin: a plane
  !! dof whose row and column factor() may replace by the identity (0: none).
  !! Collective.
  subroutine plane_contrib_setup(this, nc, crow, ccol, pin)
    class(plane_t), intent(inout) :: this
    integer, intent(in) :: nc, pin
    integer, intent(in) :: crow(nc), ccol(nc)
    integer :: P, i, j, d, l, s, q, fr, m, nrv, ierr, nk, nu
    integer, allocatable :: sb(:), rb(:), fill(:), slot(:), rs(:), rjc(:), code(:), cnt(:), ord(:), posw(:), &
         key(:), kpos(:)
    P = pe_size
    if (allocated(this%cspos)) deallocate(this%cspos, this%cscnt, this%csdsp, this%crcnt, this%crdsp, this%rmap)
    allocate(this%cspos(max(nc, 1)), this%cscnt(0:P - 1), this%csdsp(0:P - 1), this%crcnt(0:P - 1), &
         this%crdsp(0:P - 1), fill(0:P - 1))
    this%cscnt = 0
    do i = 1, nc
       d = this%own_old(ccol(i))
       this%cscnt(d) = this%cscnt(d) + 1
    end do
    this%csdsp(0) = 0
    do d = 1, P - 1
       this%csdsp(d) = this%csdsp(d - 1) + this%cscnt(d - 1)
    end do
    allocate(sb(max(2 * nc, 1)))
    fill = 0
    do i = 1, nc
       d = this%own_old(ccol(i))
       fill(d) = fill(d) + 1
       this%cspos(i) = this%csdsp(d) + fill(d)
       sb(2 * this%cspos(i) - 1) = crow(i)
       sb(2 * this%cspos(i)) = ccol(i)
    end do
    call MPI_Alltoall(this%cscnt, 1, MPI_INTEGER, this%crcnt, 1, MPI_INTEGER, NEKO_COMM, ierr)
    this%crdsp(0) = 0
    do d = 1, P - 1
       this%crdsp(d) = this%crdsp(d - 1) + this%crcnt(d - 1)
    end do
    nrv = sum(this%crcnt)
    allocate(rb(max(2 * nrv, 1)))
    call MPI_Alltoallv(sb, 2 * this%cscnt, 2 * this%csdsp, MPI_INTEGER, rb, 2 * this%crcnt, 2 * this%crdsp, &
         MPI_INTEGER, NEKO_COMM, ierr)
    deallocate(sb)
    ! every received contribution: supernode, front column, slot of its row
    allocate(slot(max(nrv, 1)), rs(max(nrv, 1)), rjc(max(nrv, 1)), code(max(nrv, 1)), this%rmap(max(nrv, 1)))
    do j = 1, nrv
       l = this%local_index(rb(2 * j))
       if (l .eq. 0) call neko_error('quasi2d: a matrix contribution sent to a rank that does not own its column')
       rs(j) = this%csup(l); rjc(j) = this%cjc(l)
       slot(j) = slot_of(rb(2 * j - 1))
    end do
    ! front rows by supernode: position (jc - 1) m + fr, 0 for upper-triangle copies
    allocate(cnt(this%nsup + 1), ord(max(nrv, 1)), posw(this%ncol + this%next))
    cnt = 0
    do j = 1, nrv
       cnt(rs(j) + 1) = cnt(rs(j) + 1) + 1
    end do
    cnt(1) = 0
    do s = 1, this%nsup
       cnt(s + 1) = cnt(s + 1) + cnt(s)
    end do
    do j = 1, nrv
       cnt(rs(j)) = cnt(rs(j)) + 1
       ord(cnt(rs(j))) = j
    end do
    ! (cnt(s) is now the end of supernode s's range; the start is cnt(s - 1) + 1)
    posw = 0
    this%aoff(1) = 0
    allocate(key(max(nrv, 1)), kpos(max(nrv, 1)))
    if (allocated(this%apos)) deallocate(this%apos)
    allocate(this%apos(max(nrv, 1)))
    nk = 0
    do s = 1, this%nsup
       m = this%sm(s)
       do q = 1, m
          posw(this%srow(this%roff(s) + q - 1)) = q
       end do
       nu = 0
       do i = merge(1, cnt(max(s - 1, 1)) + 1, s .eq. 1), cnt(s)
          j = ord(i)
          fr = 0
          if (slot(j) .gt. 0) fr = posw(slot(j))
          if (fr .lt. rjc(j)) then
             code(j) = 0
          else
             code(j) = (rjc(j) - 1) * m + fr
             nu = nu + 1
             key(nu) = code(j); kpos(nu) = j
          end if
       end do
       do q = 1, m
          posw(this%srow(this%roff(s) + q - 1)) = 0
       end do
       ! unique positions of supernode s, in order
       call sort_pair(key, kpos, nu)
       d = 0
       do i = 1, nu
          if (i .eq. 1) then
             d = d + 1
          else if (key(i) .ne. key(i - 1)) then
             d = d + 1
          end if
          code(kpos(i)) = -(nk + d)
          key(d) = key(i)
       end do
       this%apos(nk + 1:nk + d) = key(1:d)
       nk = nk + d
       this%aoff(s + 1) = nk
    end do
    do j = 1, nrv
       this%rmap(j) = max(-code(j), 0)
    end do
    ! the pinned row and column
    if (allocated(this%apin)) deallocate(this%apin)
    allocate(this%apin(max(nk, 1)))
    this%apin = 0
    if (pin .gt. 0) then
       do j = 1, nrv
          if (this%rmap(j) .eq. 0) cycle
          if (rb(2 * j - 1) .eq. pin .and. rb(2 * j) .eq. pin) then
             this%apin(this%rmap(j)) = 1
          else if (rb(2 * j - 1) .eq. pin .or. rb(2 * j) .eq. pin) then
             this%apin(this%rmap(j)) = 2
          end if
       end do
    end if
    this%contrib = .true.

  contains

    !> the slot of plane dof o (0: none)
    integer function slot_of(o)
      integer, intent(in) :: o
      integer :: lo, hi, mid
      lo = 1; hi = this%ncol + this%next
      slot_of = 0
      do while (lo .le. hi)
         mid = (lo + hi) / 2
         if (this%ssort(mid) .lt. o) then
            lo = mid + 1
         else if (this%ssort(mid) .gt. o) then
            hi = mid - 1
         else
            slot_of = this%sspos(mid)
            return
         end if
      end do
    end function slot_of

  end subroutine plane_contrib_setup

  !> the numeric factorization of mode md from this rank's contributions
  !! cval (in the order of contrib_setup); pin: replace the pinned row and
  !! column by the identity. Collective.
  subroutine plane_factor(this, md, nc, cval, pin)
    class(plane_t), intent(inout) :: this
    integer, intent(in) :: md, nc
    real(kind=dp), intent(in) :: cval(nc)
    logical, intent(in) :: pin
    real(kind=dp), allocatable :: sb(:), rb(:), av(:), fw(:)
    type(umat_t), allocatable :: upd(:)
    integer :: ierr, na, i, e, k, m, c, mu, ii, jj, info, j
    real(kind=dp) :: t0, t1, ts
    if (.not. this%contrib) call neko_error('quasi2d: plane factor before contrib_setup')
    ts = MPI_Wtime(); t0 = ts
    ! 1. the contributions to the owners of their columns, summed per entry
    allocate(sb(max(nc, 1)), rb(max(sum(this%crcnt), 1)))
    do i = 1, nc
       sb(this%cspos(i)) = cval(i)
    end do
    call MPI_Alltoallv(sb, this%cscnt, this%csdsp, MPI_DOUBLE_PRECISION, rb, this%crcnt, this%crdsp, &
         MPI_DOUBLE_PRECISION, NEKO_COMM, ierr)
    na = this%aoff(this%nsup + 1)
    allocate(av(max(na, 1)))
    av = 0.0_dp
    do j = 1, sum(this%crcnt)
       if (this%rmap(j) .gt. 0) av(this%rmap(j)) = av(this%rmap(j)) + rb(j)
    end do
    deallocate(sb, rb)
    if (pin) then
       do e = 1, na
          if (this%apin(e) .eq. 1) av(e) = 1.0_dp
          if (this%apin(e) .eq. 2) av(e) = 0.0_dp
       end do
    end if
    t1 = MPI_Wtime(); this%tfac(1) = this%tfac(1) + (t1 - t0); t0 = t1
    if (.not. allocated(this%g(md)%v)) allocate(this%g(md)%v(max(this%boff(this%nsup + 1), 1_i8)))
    allocate(upd(this%nsup))
    ! 2. the fronts in postorder
    do i = 1, this%nsup
       k = this%sk(i); m = this%sm(i)
       if (this%sgs(i) .eq. 1) then
          if (allocated(fw)) then
             if (size(fw) .lt. m * m) deallocate(fw)
          end if
          if (.not. allocated(fw)) allocate(fw(max(m * m, 1)))
          fw(1:m * m) = 0.0_dp
          do e = this%aoff(i) + 1, this%aoff(i + 1)
             fw(this%apos(e)) = fw(this%apos(e)) + av(e)
          end do
          c = this%lhead(i)
          do while (c .ne. 0)
             mu = this%sm(c) - this%sk(c)
             do jj = 1, mu
                do ii = jj, mu
                   e = this%cmap(this%cmo(c) + ii - 1) + (this%cmap(this%cmo(c) + jj - 1) - 1) * m
                   fw(e) = fw(e) + upd(c)%u(ii, jj)
                end do
             end do
             if (allocated(upd(c)%u)) deallocate(upd(c)%u)
             c = this%lnext(c)
          end do
          call dpotrf('L', k, fw, m, info)
          if (info .ne. 0) call neko_error('quasi2d: a 2D factorization failed (not positive definite)')
          if (m .gt. k) then
             call dtrsm('R', 'L', 'T', 'N', m - k, k, 1.0_dp, fw, m, fw(k + 1), m)
             if (this%spar(i) .ne. 0) then
                ! lower for a local parent; upper (rows as columns) for a
                ! distributed one, which takes it by rows
                mu = m - k
                allocate(upd(i)%u(mu, mu))
                if (this%sgs(this%spar(i)) .eq. 1) then
                   do jj = 1, mu
                      do ii = jj, mu
                         upd(i)%u(ii, jj) = fw(k + ii + (k + jj - 1) * m)
                      end do
                   end do
                   call dsyrk('L', 'N', mu, k, -1.0_dp, fw(k + 1), m, 1.0_dp, upd(i)%u, mu)
                else
                   do jj = 1, mu
                      do ii = jj, mu
                         upd(i)%u(jj, ii) = fw(k + ii + (k + jj - 1) * m)
                      end do
                   end do
                   call dsyrk('U', 'N', mu, k, -1.0_dp, fw(k + 1), m, 1.0_dp, upd(i)%u, mu)
                end if
             end if
          end if
          call dtrtri('L', 'N', k, fw, m, info)
          if (info .ne. 0) call neko_error('quasi2d: dtrtri failed')
          if (m .gt. k) call dtrmm('R', 'L', 'N', 'N', m - k, k, 1.0_dp, fw, m, fw(k + 1), m)
          this%g(md)%v(this%boff(i) + 1:this%boff(i) + int(m, i8) * k) = fw(1:m * k)
          t1 = MPI_Wtime(); this%tfac(2) = this%tfac(2) + (t1 - t0); t0 = t1
       else
          call dist_front(this, i, md, av, upd)
          t1 = MPI_Wtime(); this%tfac(3) = this%tfac(3) + (t1 - t0); t0 = t1
       end if
    end do
    this%fresh(md) = .true.
    this%t_factor = this%t_factor + (MPI_Wtime() - ts)
  end subroutine plane_factor

  !> the distributed front i: assembly (values, children's update rows by an
  !! all-to-all in the group), partial Cholesky, G rows, update rows. The
  !! front is stored transposed, FT(m, nown): column l is this rank's l-th row.
  subroutine dist_front(this, i, md, av, upd)
    class(plane_t), intent(inout), target :: this
    integer, intent(in) :: i, md
    real(kind=dp), intent(in) :: av(:)
    type(umat_t), intent(inout) :: upd(:)
    integer :: k, m, g, me, nown, nlow, e, j, ch, c, mu, ii, jj, fr, pr, d, ierr, q, src, lr, lu, nb, nbc
    integer :: scnt(0:this%sgs(i) - 1), sdsp(0:this%sgs(i) - 1), rcnt(0:this%sgs(i) - 1), &
         rdsp(0:this%sgs(i) - 1), fill(0:this%sgs(i) - 1)
    real(kind=dp), allocatable :: FT(:,:), sb(:), rb(:)
    real(kind=dp) :: tq
    type(MPI_Comm) :: cm
    tq = MPI_Wtime()
    k = this%sk(i); m = this%sm(i); g = this%sgs(i)
    me = pe_rank - this%sg0(i)
    nown = this%nown(i); nlow = this%nlow(i); nb = this%snb(i)
    cm = this%dcomm(this%sdep(i))
    allocate(FT(m, max(nown, 1)))
    FT = 0.0_dp
    ! to every rank of the group: [number of matrix entries, (position in its
    ! FT, value) for each, the children's update rows owned there]. The
    ! matrix entries held here are those of the columns owned here; a row of
    ! an update matrix is a column of its storage.
    scnt = 1
    do e = this%aoff(i) + 1, this%aoff(i + 1)
       d = rown(mod(this%apos(e) - 1, m) + 1, g, nb)
       scnt(d) = scnt(d) + 2
    end do
    do ch = this%choff(i) + 1, this%choff(i + 1)
       c = this%chh(ch)
       if (c .eq. 0) cycle
       mu = this%chm(ch) - this%chk(ch)
       q = pe_rank - this%chg0(ch)
       nbc = rowblk(this%chm(ch), this%chgs(ch))
       do ii = 1, mu
          fr = this%chk(ch) + ii
          if (rown(fr, this%chgs(ch), nbc) .ne. q) cycle
          d = rown(this%chmap(this%chmo(ch) + ii - 1), g, nb)
          scnt(d) = scnt(d) + ii
       end do
    end do
    sdsp(0) = 0
    do d = 1, g - 1
       sdsp(d) = sdsp(d - 1) + scnt(d - 1)
    end do
    allocate(sb(max(sum(scnt), 1)))
    fill = 1
    do e = this%aoff(i) + 1, this%aoff(i + 1)
       fr = mod(this%apos(e) - 1, m) + 1
       d = rown(fr, g, nb)
       sb(sdsp(d) + fill(d) + 1) = real((rloc(fr, g, nb) - 1) * m + (this%apos(e) - 1) / m + 1, dp)
       sb(sdsp(d) + fill(d) + 2) = av(e)
       fill(d) = fill(d) + 2
    end do
    do d = 0, g - 1
       sb(sdsp(d) + 1) = real((fill(d) - 1) / 2, dp)
    end do
    do ch = this%choff(i) + 1, this%choff(i + 1)
       c = this%chh(ch)
       if (c .eq. 0) cycle
       mu = this%chm(ch) - this%chk(ch)
       q = pe_rank - this%chg0(ch)
       nbc = rowblk(this%chm(ch), this%chgs(ch))
       do ii = 1, mu
          fr = this%chk(ch) + ii
          if (rown(fr, this%chgs(ch), nbc) .ne. q) cycle
          d = rown(this%chmap(this%chmo(ch) + ii - 1), g, nb)
          lu = rloc(fr, this%chgs(ch), nbc) - this%nlow(c)
          sb(sdsp(d) + fill(d) + 1:sdsp(d) + fill(d) + ii) = upd(c)%u(1:ii, lu)
          fill(d) = fill(d) + ii
       end do
       if (allocated(upd(c)%u)) deallocate(upd(c)%u)
    end do
    call MPI_Alltoall(scnt, 1, MPI_INTEGER, rcnt, 1, MPI_INTEGER, cm, ierr)
    rdsp(0) = 0
    do d = 1, g - 1
       rdsp(d) = rdsp(d - 1) + rcnt(d - 1)
    end do
    allocate(rb(max(sum(rcnt), 1)))
    call MPI_Alltoallv(sb, scnt, sdsp, MPI_DOUBLE_PRECISION, rb, rcnt, rdsp, MPI_DOUBLE_PRECISION, cm, ierr)
    deallocate(sb)
    do d = 0, g - 1
       do e = 1, nint(rb(rdsp(d) + 1))
          j = nint(rb(rdsp(d) + 2 * e))
          FT(mod(j - 1, m) + 1, (j - 1) / m + 1) = FT(mod(j - 1, m) + 1, (j - 1) / m + 1) + rb(rdsp(d) + 2 * e + 1)
       end do
       fill(d) = rdsp(d) + 1 + 2 * nint(rb(rdsp(d) + 1))
    end do
    do ch = this%choff(i) + 1, this%choff(i + 1)
       mu = this%chm(ch) - this%chk(ch)
       nbc = rowblk(this%chm(ch), this%chgs(ch))
       do ii = 1, mu
          pr = this%chmap(this%chmo(ch) + ii - 1)
          if (rown(pr, g, nb) .ne. me) cycle
          src = this%chg0(ch) + rown(this%chk(ch) + ii, this%chgs(ch), nbc) - this%sg0(i)
          lr = rloc(pr, g, nb)
          do jj = 1, ii
             FT(this%chmap(this%chmo(ch) + jj - 1), lr) = FT(this%chmap(this%chmo(ch) + jj - 1), lr) + &
                  rb(fill(src) + jj)
          end do
          fill(src) = fill(src) + ii
       end do
    end do
    deallocate(rb)
    if (any(fill .ne. rdsp + rcnt)) call neko_error('quasi2d: extend-add count mismatch')
    q2d_tdist(1) = q2d_tdist(1) + (MPI_Wtime() - tq)
    ! partial factorization and the G rows (transposed: GT(k, nown))
    call dist_chol(FT, m, nown, k, g, me, nb, cm, this%g(md)%v(this%boff(i) + 1:this%boff(i + 1)))
    if (this%spar(i) .ne. 0 .and. m .gt. k) then
       allocate(upd(i)%u(m - k, max(nown - nlow, 1)))
       if (nown .gt. nlow) upd(i)%u(:, 1:nown - nlow) = FT(k + 1:m, nlow + 1:nown)
    end if
  end subroutine dist_front

  !> right-looking blocked partial Cholesky of the front (m x m, k pivots;
  !! this rank's rows, in blocks of nb rows cyclic over g ranks, stored
  !! transposed as FT(m, nown); panels of NB columns), then the G rows
  !! G = [L11; L21] L11^{-1} of this rank, transposed into GT(k, nown). Every
  !! rank factors the gathered diagonal blocks (kept as U = L_JJ^T).
  subroutine dist_chol(FT, m, nown, k, g, me, nb, cm, GT)
    integer, intent(in) :: m, nown, k, g, me, nb
    real(kind=dp), intent(inout) :: FT(m, *)
    type(MPI_Comm), intent(in) :: cm
    real(kind=dp), intent(out) :: GT(max(k, 1), *)
    integer, parameter :: CH = 64
    real(kind=dp), allocatable :: Ud(:,:), LpT(:,:), X(:,:), WT(:,:)
    integer :: j0, je, b, la, lb, lo, nrem, nl, l1, l2, r2, c, info, ip
    real(kind=dp) :: tq
    tq = MPI_Wtime()
    allocate(Ud(NB, max(k, 1)), X(NB, max(k, 1)))
    do j0 = 1, k, NB
       b = min(NB, k - j0 + 1); je = j0 + b - 1
       ! the diagonal block to every rank (as its transpose), factored by every rank
       la = ncnt(j0 - 1, me, g, nb) + 1; lb = ncnt(je, me, g, nb)
       call gather_t(FT, m, j0, b, la, lb, j0, je, Ud(:, j0:je), NB)
       call dpotrf('U', b, Ud(1, j0), NB, info)
       if (info .ne. 0) call neko_error('quasi2d: a 2D factorization failed (not positive definite)')
       ! own rows below it: L(r, J) = F(r, J) L_JJ^{-T}, i.e. FT(J, r) = U^{-T} FT(J, r)
       lo = lb + 1
       nrem = nown - lo + 1
       if (nrem .gt. 0) call dtrsm('L', 'U', 'T', 'N', b, nrem, 1.0_dp, Ud(1, j0), NB, FT(j0, lo), m)
       nl = m - je
       if (nl .eq. 0) cycle
       ! all rows below it to every rank: LpT(:, r - je) = L(r, J)
       allocate(LpT(b, nl))
       call gather_t(FT, m, j0, b, lo, nown, je + 1, m, LpT, b)
       ! trailing update, CH own rows at a time (rows up to their last own row)
       l1 = lo
       do while (l1 .le. nown)
          l2 = min(nown, l1 + CH - 1)
          r2 = rrow(l2, me, g, nb)
          call dgemm('T', 'N', r2 - je, l2 - l1 + 1, b, -1.0_dp, LpT, b, FT(j0, l1), m, 1.0_dp, &
               FT(je + 1, l1), m)
          l1 = l2 + 1
       end do
       deallocate(LpT)
    end do
    q2d_tdist(2) = q2d_tdist(2) + (MPI_Wtime() - tq); tq = MPI_Wtime()
    ! G rows: rows <= k of L11^{-1} (streamed in panels), rows > k L21 L11^{-1}
    if (nown .gt. 0) GT(1:k, 1:nown) = 0.0_dp
    if (k .gt. 1) allocate(WT(k, NB))
    do j0 = 1, k, NB
       b = min(NB, k - j0 + 1); je = j0 + b - 1
       la = ncnt(j0 - 1, me, g, nb) + 1; lb = ncnt(je, me, g, nb)
       ! X = L_JJ^{-1} ([0 I] - W(J, :)), W(J, :) gathered from the owners
       if (j0 .gt. 1) then
          call gather_t(GT, max(k, 1), 1, j0 - 1, la, lb, j0, je, WT, k)
          X(1:b, 1:j0 - 1) = -transpose(WT(1:j0 - 1, 1:b))
       end if
       X(1:b, j0:je) = 0.0_dp
       do c = 1, b
          X(c, j0 + c - 1) = 1.0_dp
       end do
       call dtrsm('L', 'U', 'T', 'N', b, je, 1.0_dp, Ud(1, j0), NB, X, NB)
       do ip = la, lb
          GT(1:je, ip) = X(rrow(ip, me, g, nb) - j0 + 1, 1:je)
       end do
       lo = lb + 1
       if (nown .ge. lo) call dgemm('T', 'N', je, nown - lo + 1, b, 1.0_dp, X, NB, FT(j0, lo), m, 1.0_dp, &
            GT(1, lo), max(k, 1))
    end do
    q2d_tdist(3) = q2d_tdist(3) + (MPI_Wtime() - tq)

  contains

    !> rows r0..r1 of the front, columns c0..c0+ncl-1 (this rank's local rows
    !! l0..l1 of A, stored transposed with leading dimension lda), gathered
    !! from all ranks: OutT(:, r - r0 + 1) = A(c0:c0+ncl-1, row r)
    subroutine gather_t(A, lda, c0, ncl, l0, l1_, r0, r1, OutT, ldo)
      integer, intent(in) :: lda, c0, ncl, l0, l1_, r0, r1, ldo
      real(kind=dp), intent(in) :: A(lda, *)
      real(kind=dp), intent(inout) :: OutT(ldo, *)
      real(kind=dp), allocatable :: sb(:), rb(:)
      integer :: q, r, p, nmine, ierr, l
      integer :: cq(0:g - 1), dq(0:g - 1), nb0(0:g - 1)
      do q = 0, g - 1
         nb0(q) = ncnt(r0 - 1, q, g, nb)
         cq(q) = (ncnt(r1, q, g, nb) - nb0(q)) * ncl
      end do
      dq(0) = 0
      do q = 1, g - 1
         dq(q) = dq(q - 1) + cq(q - 1)
      end do
      nmine = max(l1_ - l0 + 1, 0)
      allocate(sb(max(nmine * ncl, 1)), rb(max(sum(cq), 1)))
      do l = 1, nmine
         sb((l - 1) * ncl + 1:l * ncl) = A(c0:c0 + ncl - 1, l0 + l - 1)
      end do
      call MPI_Allgatherv(sb, nmine * ncl, MPI_DOUBLE_PRECISION, rb, cq, dq, MPI_DOUBLE_PRECISION, cm, ierr)
      do r = r0, r1
         q = rown(r, g, nb)
         p = rloc(r, g, nb) - nb0(q)
         OutT(1:ncl, r - r0 + 1) = rb(dq(q) + (p - 1) * ncl + 1:dq(q) + p * ncl)
      end do
    end subroutine gather_t

  end subroutine dist_chol

  ! ---------------------------------------------------------------------------
  ! solve
  ! ---------------------------------------------------------------------------

  !> Y(:, :, a) <- A_{modes(a)}^{-1} Y(:, :, a) for na modes, nr right-hand
  !! sides each, in the local numbering (rows 1..ncol: this rank's columns;
  !! ncol+1..ncol+next: the other slots, zero on entry). Collective.
  subroutine plane_sweep(this, na, modes, nr, ld, Y)
    class(plane_t), intent(inout) :: this
    integer, intent(in) :: na, nr, ld
    integer, intent(in) :: modes(na)
    real(kind=dp), intent(inout) :: Y(ld, nr, na)
    integer :: a, i, ierr, k, m, g, me, nown, q, l, fr, c, mx, nz, nb
    real(kind=dp) :: tt0
    mx = 1
    do i = 1, this%nsup
       mx = max(mx, this%sm(i), this%nown(i))
    end do
    if (size(this%t, 1) .lt. mx .or. size(this%t, 2) .ne. nr) then
       deallocate(this%t)
       allocate(this%t(mx, nr))
    end if
    if (size(this%zb) .lt. mx * nr * na) then
       deallocate(this%zb)
       allocate(this%zb(mx * nr * na))
    end if
    ! forward
    do i = 1, this%nsup
       tt0 = MPI_Wtime()
       k = this%sk(i); m = this%sm(i); g = this%sgs(i)
       if (g .eq. 1) then
          do a = 1, na
             call fwd(i, modes(a), Y(:, :, a))
          end do
          this%tsw(1) = this%tsw(1) + (MPI_Wtime() - tt0)
          cycle
       end if
       me = pe_rank - this%sg0(i); nown = this%nown(i); nb = this%snb(i)
       nz = m * nr * na
       do a = 1, na
          do c = 1, nr
             do q = 1, m
                this%zb(q + m * (c - 1) + m * nr * (a - 1)) = Y(this%srow(this%roff(i) + q - 1), c, a)
                Y(this%srow(this%roff(i) + q - 1), c, a) = 0.0_dp
             end do
          end do
       end do
       call MPI_Allreduce(MPI_IN_PLACE, this%zb(1:nz), nz, MPI_DOUBLE_PRECISION, MPI_SUM, &
            this%dcomm(this%sdep(i)), ierr)
       if (nown .gt. 0) then
          do a = 1, na
             call dgemm('T', 'N', nown, nr, k, 1.0_dp, this%g(modes(a))%v(this%boff(i) + 1), k, &
                  this%zb(1 + m * nr * (a - 1)), m, 0.0_dp, this%t, size(this%t, 1))
             do l = 1, nown
                fr = rrow(l, me, g, nb)
                do c = 1, nr
                   if (fr .le. k) then
                      Y(this%srow(this%roff(i) + fr - 1), c, a) = this%t(l, c)
                   else
                      Y(this%srow(this%roff(i) + fr - 1), c, a) = this%zb(fr + m * (c - 1) + m * nr * (a - 1)) &
                           - this%t(l, c)
                   end if
                end do
             end do
          end do
       end if
       this%tsw(2) = this%tsw(2) + (MPI_Wtime() - tt0)
    end do
    ! backward
    do i = this%nsup, 1, -1
       tt0 = MPI_Wtime()
       k = this%sk(i); m = this%sm(i); g = this%sgs(i)
       if (g .eq. 1) then
          do a = 1, na
             call bwd(i, modes(a), Y(:, :, a))
          end do
          this%tsw(4) = this%tsw(4) + (MPI_Wtime() - tt0)
          cycle
       end if
       me = pe_rank - this%sg0(i); nown = this%nown(i); nb = this%snb(i)
       nz = k * nr * na
       this%zb(1:nz) = 0.0_dp
       if (nown .gt. 0) then
          do a = 1, na
             do l = 1, nown
                fr = rrow(l, me, g, nb)
                do c = 1, nr
                   if (fr .le. k) then
                      this%t(l, c) = Y(this%srow(this%roff(i) + fr - 1), c, a)
                   else
                      this%t(l, c) = -Y(this%srow(this%roff(i) + fr - 1), c, a)
                   end if
                end do
             end do
             call dgemm('N', 'N', k, nr, nown, 1.0_dp, this%g(modes(a))%v(this%boff(i) + 1), k, &
                  this%t, size(this%t, 1), 0.0_dp, this%zb(1 + k * nr * (a - 1)), k)
          end do
       end if
       call MPI_Allreduce(MPI_IN_PLACE, this%zb(1:nz), nz, MPI_DOUBLE_PRECISION, MPI_SUM, &
            this%dcomm(this%sdep(i)), ierr)
       do a = 1, na
          do c = 1, nr
             do q = 1, k
                Y(this%srow(this%roff(i) + q - 1), c, a) = this%zb(q + k * (c - 1) + k * nr * (a - 1))
             end do
          end do
       end do
       this%tsw(3) = this%tsw(3) + (MPI_Wtime() - tt0)
    end do

  contains

    !> t = G y_f; y_f = t(1:k); y(rows) -= t(k+1:m)
    subroutine fwd(ii, md, Ya)
      integer, intent(in) :: ii, md
      real(kind=dp), intent(inout) :: Ya(ld, nr)
      integer :: kk, mm, f, qq, cc
      kk = this%sk(ii); mm = this%sm(ii); f = this%c0(ii)
      if (nr .eq. 1) then
         call dgemv('N', mm, kk, 1.0_dp, this%g(md)%v(this%boff(ii) + 1), mm, Ya(f, 1), 1, 0.0_dp, &
              this%t(1, 1), 1)
      else
         call dgemm('N', 'N', mm, nr, kk, 1.0_dp, this%g(md)%v(this%boff(ii) + 1), mm, Ya(f, 1), ld, 0.0_dp, &
              this%t(1, 1), size(this%t, 1))
      end if
      do cc = 1, nr
         Ya(f:f + kk - 1, cc) = this%t(1:kk, cc)
         do qq = kk + 1, mm
            Ya(this%srow(this%roff(ii) + qq - 1), cc) = Ya(this%srow(this%roff(ii) + qq - 1), cc) - this%t(qq, cc)
         end do
      end do
    end subroutine fwd

    !> x_f = G^T [y_f; -x(rows)]
    subroutine bwd(ii, md, Ya)
      integer, intent(in) :: ii, md
      real(kind=dp), intent(inout) :: Ya(ld, nr)
      integer :: kk, mm, f, qq, cc
      kk = this%sk(ii); mm = this%sm(ii); f = this%c0(ii)
      do cc = 1, nr
         this%t(1:kk, cc) = Ya(f:f + kk - 1, cc)
         do qq = kk + 1, mm
            this%t(qq, cc) = -Ya(this%srow(this%roff(ii) + qq - 1), cc)
         end do
      end do
      if (nr .eq. 1) then
         call dgemv('T', mm, kk, 1.0_dp, this%g(md)%v(this%boff(ii) + 1), mm, this%t(1, 1), 1, 0.0_dp, &
              Ya(f, 1), 1)
      else
         call dgemm('T', 'N', kk, nr, mm, 1.0_dp, this%g(md)%v(this%boff(ii) + 1), mm, this%t(1, 1), &
              size(this%t, 1), 0.0_dp, Ya(f, 1), ld)
      end if
    end subroutine bwd

  end subroutine plane_sweep

  !> local column of plane dof old on this rank (0: not owned here)
  integer function plane_local_index(this, old) result(l)
    class(plane_t), intent(in) :: this
    integer, intent(in) :: old
    integer :: lo, hi, mid
    lo = 1; hi = this%ncol
    l = 0
    do while (lo .le. hi)
       mid = (lo + hi) / 2
       if (this%csort(mid) .lt. old) then
          lo = mid + 1
       else if (this%csort(mid) .gt. old) then
          hi = mid - 1
       else
          l = this%cpos(mid)
          return
       end if
    end do
  end function plane_local_index

  subroutine plane_free(this)
    class(plane_t), intent(inout) :: this
    integer :: i, ierr
    if (allocated(this%g)) then
       do i = 1, size(this%g)
          if (allocated(this%g(i)%v)) deallocate(this%g(i)%v)
       end do
       deallocate(this%g)
    end if
    if (allocated(this%dcomm)) then
       do i = 0, this%dmax
          if (this%have(i)) call MPI_Comm_free(this%dcomm(i), ierr)
       end do
       deallocate(this%dcomm, this%have)
    end if
    if (allocated(this%fresh)) deallocate(this%fresh)
    if (allocated(this%apin)) deallocate(this%apin)
    if (allocated(this%cspos)) deallocate(this%cspos, this%cscnt, this%csdsp, this%crcnt, this%crdsp, this%rmap)
    if (allocated(this%csup)) deallocate(this%csup, this%cjc)
    if (allocated(this%sold)) deallocate(this%sold, this%ssort, this%sspos)
    this%contrib = .false.
    if (allocated(this%sk)) deallocate(this%sk, this%sm, this%sg0, this%sgs, this%sdep, this%spar, &
         this%nown, this%nlow, this%c0, this%snb, this%roff, this%cmo, this%boff, this%choff, this%aoff, &
         this%lhead, this%lnext)
    if (allocated(this%srow)) deallocate(this%srow)
    if (allocated(this%cmap)) deallocate(this%cmap)
    if (allocated(this%chk)) deallocate(this%chk, this%chm, this%chg0, this%chgs, this%chh, this%chmo)
    if (allocated(this%chmap)) deallocate(this%chmap)
    if (allocated(this%apos)) deallocate(this%apos)
    if (allocated(this%cold)) deallocate(this%cold)
    if (allocated(this%csort)) deallocate(this%csort, this%cpos)
    if (allocated(this%own_old)) deallocate(this%own_old)
    if (allocated(this%t)) deallocate(this%t)
    if (allocated(this%zb)) deallocate(this%zb)
    this%ready = .false.
    this%nsup = 0; this%ncol = 0; this%next = 0; this%dmax = 0
  end subroutine plane_free

  !> ord such that W(cand(ord)) is descending (insertion sort; few candidates)
  subroutine sort_desc(W, cand, ord, n)
    integer(kind=i8), intent(in) :: W(:)
    integer, intent(in) :: cand(:), n
    integer, intent(inout) :: ord(:)
    integer :: i, j, v
    do i = 2, n
       v = ord(i)
       j = i - 1
       do while (j .ge. 1)
          if (W(cand(ord(j))) .ge. W(cand(v))) exit
          ord(j + 1) = ord(j)
          j = j - 1
       end do
       ord(j + 1) = v
    end do
  end subroutine sort_desc

  !> sort a ascending, carrying b (heapsort)
  subroutine sort_pair(a, b, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(:), b(:)
    integer :: i, t
    do i = n / 2, 1, -1
       call sift(i, n)
    end do
    do i = n, 2, -1
       t = a(1); a(1) = a(i); a(i) = t
       t = b(1); b(1) = b(i); b(i) = t
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
            if (a(child) .lt. a(child + 1)) child = child + 1
         end if
         if (a(root) .lt. a(child)) then
            tt = a(root); a(root) = a(child); a(child) = tt
            tt = b(root); b(root) = b(child); b(child) = tt
            root = child
         else
            return
         end if
      end do
    end subroutine sift
  end subroutine sort_pair

end module q2d_plane
