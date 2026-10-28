! Copied from neko-extruded/ext_spchol.f90 (box_solvers, f64444e), itself a copy of
! robust_solvers/src/neko_types/rs_spchol.f90 (d54cabc); module renamed to q2d_spchol.
!> Sparse Cholesky factorization A = L L^T of a symmetric positive definite
!! matrix given in CSR (both triangles, 1-based), for coarse-grid solves that
!! are replicated on every rank. Self-contained apart from BLAS / LAPACK.
!!
!! - ordering: nested dissection with breadth-first level-set separators
!!   (recursive, connected components separately, separator refined so that
!!   every separator node touches both halves), leaves in their BFS order;
!! - symbolic: elimination tree (Liu), postorder, column counts by row-subtree
!!   traversal, supernodes = chains of the elimination tree with relaxed
!!   amalgamation, supernodal row structures;
!! - numeric: multifrontal; each supernode assembles a dense front from A and
!!   the update matrices of its children (extend-add) and does dpotrf / dtrsm /
!!   dsyrk; the factor is stored as one dense m x k block per supernode;
!! - solve: supernodal forward / backward substitution (dtrsv, dgemv).
!!
!! Double precision only (real(dp)), as the LAPACK routines it calls.
module q2d_spchol
  use num_types, only : dp, i8
  implicit none
  private

  type, public :: spchol_t
     integer :: n = 0
     integer, allocatable :: perm(:)          !< new -> old
     integer, allocatable :: iperm(:)         !< old -> new
     integer :: nsuper = 0
     integer, allocatable :: sfc(:)           !< first column of each supernode (nsuper + 1)
     integer, allocatable :: sparent(:)       !< parent supernode (0: root)
     integer, allocatable :: srp(:)           !< row-structure pointer (nsuper + 1)
     integer, allocatable :: srow(:)          !< row indices (new numbering), first k are the columns
     integer(kind=i8), allocatable :: slp(:)  !< offset of the dense m x k factor block (nsuper + 1)
     real(kind=dp), allocatable :: lx(:)      !< factor blocks, column-major
     integer :: maxm = 0
     !> lower triangle of the permuted matrix, CSC (new numbering)
     integer, allocatable :: acp(:), aci(:)
     real(kind=dp), allocatable :: acx(:)
     integer(kind=i8) :: nnz_l = 0            !< stored entries of L (incl. amalgamation zeros)
     real(kind=dp) :: flops = 0.0_dp
     real(kind=dp) :: t_order = 0.0_dp, t_symbolic = 0.0_dp, t_numeric = 0.0_dp
     integer :: nd_leaf = 32                  !< nested dissection stops at this size
     integer :: nrelax = 16                   !< supernodes of up to this width are amalgamated freely
     !> 0 auto (the one of nested dissection / AMD with fewer predicted
     !! nonzeros of L), 1 nested dissection, 2 approximate minimum degree
     integer :: ordering = 0
     integer :: ordering_used = 0
     integer(kind=i8) :: nnz_nd = 0, nnz_amd = 0
     real(kind=dp), allocatable :: work(:), tbuf(:)
     !> quasi2d: store every supernode block as G = [L11^{-1}; L21 L11^{-1}]
     !! after the factorization, so that both sweeps are matrix-vector (or
     !! matrix-matrix, solve_many) products
     logical :: gform = .false.
     logical :: is_g = .false.
     real(kind=dp), allocatable :: wm(:,:), tm(:,:)
   contains
     procedure, pass(this) :: init => spchol_init
     procedure, pass(this) :: analyse => spchol_analyse
     procedure, pass(this) :: factor => spchol_factor
     procedure, pass(this) :: solve => spchol_solve
     procedure, pass(this) :: solve_many => spchol_solve_many
     procedure, pass(this) :: free => spchol_free
  end type spchol_t

contains

  !> Ordering and symbolic analysis of the pattern of A, then the numeric
  !! factorization. ia, ja: CSR of the full symmetric matrix, 1-based.
  !! info = 0 on success, > 0 if a pivot is not positive (supernode index).
  subroutine spchol_init(this, n, ia, ja, a, info)
    class(spchol_t), intent(inout) :: this
    integer, intent(in) :: n
    integer, intent(in) :: ia(n + 1), ja(*)
    real(kind=dp), intent(in) :: a(*)
    integer, intent(out) :: info

    info = 0
    call this%analyse(n, ia, ja)
    if (n .le. 0) return
    call this%factor(ia, ja, a, info)
  end subroutine spchol_init

  !> Ordering and symbolic analysis only (no numeric factor, no storage for it).
  subroutine spchol_analyse(this, n, ia, ja)
    class(spchol_t), intent(inout) :: this
    integer, intent(in) :: n
    integer, intent(in) :: ia(n + 1), ja(*)
    integer :: t0, t1, t2, rate

    call this%free()
    this%n = n
    if (n .le. 0) return
    call system_clock(t0, rate)
    select case (this%ordering)
    case (1)
       call nd_order(n, ia, ja, this%nd_leaf, this%perm)
       this%ordering_used = 1
    case (2)
       call amd_order(n, ia, ja, this%perm)
       this%ordering_used = 2
    case default
       block
         integer, allocatable :: p1(:), p2(:)
         call nd_order(n, ia, ja, this%nd_leaf, p1)
         call amd_order(n, ia, ja, p2)
         this%nnz_nd = predict_nnz(n, ia, ja, p1)
         this%nnz_amd = predict_nnz(n, ia, ja, p2)
         if (this%nnz_amd .lt. this%nnz_nd) then
            call move_alloc(p2, this%perm); this%ordering_used = 2
         else
            call move_alloc(p1, this%perm); this%ordering_used = 1
         end if
       end block
    end select
    allocate(this%iperm(n))
    call system_clock(t1)
    call symbolic(this, ia, ja)
    call system_clock(t2)
    this%t_order = real(t1 - t0, dp) / rate
    this%t_symbolic = real(t2 - t1, dp) / rate
  end subroutine spchol_analyse

  !> Numeric factorization (same pattern as at init).
  subroutine spchol_factor(this, ia, ja, a, info)
    class(spchol_t), intent(inout) :: this
    integer, intent(in) :: ia(this%n + 1), ja(*)
    real(kind=dp), intent(in) :: a(*)
    integer, intent(out) :: info
    integer :: t0, t1, rate

    call system_clock(t0, rate)
    if (.not. allocated(this%lx)) allocate(this%lx(this%nnz_l))
    call fill_lower(this, ia, ja, a)
    call multifrontal(this, info)
    this%is_g = .false.
    if (info .eq. 0 .and. this%gform) call to_gform(this)
    call system_clock(t1)
    this%t_numeric = real(t1 - t0, dp) / rate
  end subroutine spchol_factor

  !> x = A^{-1} b
  subroutine spchol_solve(this, b, x)
    class(spchol_t), intent(inout) :: this
    real(kind=dp), intent(in) :: b(this%n)
    real(kind=dp), intent(out) :: x(this%n)
    integer :: s, f, k, m, i, r0
    integer(kind=i8) :: o

    if (this%is_g) then
       call solve_g(this, b, x)
       return
    end if
    associate(y => this%work)
      do i = 1, this%n
         y(i) = b(this%perm(i))
      end do
      ! forward: L y = Pb
      do s = 1, this%nsuper
         f = this%sfc(s); k = this%sfc(s + 1) - f
         r0 = this%srp(s); m = this%srp(s + 1) - r0
         o = this%slp(s)
         call dtrsv('L', 'N', 'N', k, this%lx(o + 1), m, y(f), 1)
         if (m .gt. k) then
            call gemv_scatter(m - k, k, this%lx(o + k + 1), m, y(f), this%srow(r0 + k), y, this%tbuf)
         end if
      end do
      ! backward: L^T x = y
      do s = this%nsuper, 1, -1
         f = this%sfc(s); k = this%sfc(s + 1) - f
         r0 = this%srp(s); m = this%srp(s + 1) - r0
         o = this%slp(s)
         if (m .gt. k) then
            call gemvt_gather(m - k, k, this%lx(o + k + 1), m, this%srow(r0 + k), y, y(f), this%tbuf)
         end if
         call dtrsv('L', 'T', 'N', k, this%lx(o + 1), m, y(f), 1)
      end do
      do i = 1, this%n
         x(this%perm(i)) = y(i)
      end do
    end associate
  end subroutine spchol_solve

  !> L11 -> L11^{-1}, L21 -> L21 L11^{-1} in every supernode block
  subroutine to_gform(this)
    class(spchol_t), intent(inout) :: this
    integer :: s, k, m, info
    integer(kind=i8) :: o
    do s = 1, this%nsuper
       k = this%sfc(s + 1) - this%sfc(s)
       m = this%srp(s + 1) - this%srp(s)
       o = this%slp(s)
       call dtrtri('L', 'N', k, this%lx(o + 1), m, info)
       if (info .ne. 0) error stop 'q2d_spchol: dtrtri failed'
       if (m .gt. k) call dtrmm('R', 'L', 'N', 'N', m - k, k, 1.0_dp, this%lx(o + 1), m, &
            this%lx(o + k + 1), m)
    end do
    this%is_g = .true.
  end subroutine to_gform

  !> x = A^{-1} b with the G blocks: forward t = G y_f, y_f = t(1:k),
  !! y(rows) -= t(k+1:m); backward x_f = G^T [y_f; -x(rows)]
  subroutine solve_g(this, b, x)
    class(spchol_t), intent(inout) :: this
    real(kind=dp), intent(in) :: b(this%n)
    real(kind=dp), intent(out) :: x(this%n)
    integer :: s, f, k, m, i, r0
    integer(kind=i8) :: o
    associate(y => this%work, t => this%tbuf)
      do i = 1, this%n
         y(i) = b(this%perm(i))
      end do
      do s = 1, this%nsuper
         f = this%sfc(s); k = this%sfc(s + 1) - f
         r0 = this%srp(s); m = this%srp(s + 1) - r0
         o = this%slp(s)
         call dgemv('N', m, k, 1.0_dp, this%lx(o + 1), m, y(f), 1, 0.0_dp, t, 1)
         y(f:f + k - 1) = t(1:k)
         do i = k + 1, m
            y(this%srow(r0 + i - 1)) = y(this%srow(r0 + i - 1)) - t(i)
         end do
      end do
      do s = this%nsuper, 1, -1
         f = this%sfc(s); k = this%sfc(s + 1) - f
         r0 = this%srp(s); m = this%srp(s + 1) - r0
         o = this%slp(s)
         t(1:k) = y(f:f + k - 1)
         do i = k + 1, m
            t(i) = -y(this%srow(r0 + i - 1))
         end do
         call dgemv('T', m, k, 1.0_dp, this%lx(o + 1), m, t, 1, 0.0_dp, y(f), 1)
      end do
      do i = 1, this%n
         x(this%perm(i)) = y(i)
      end do
    end associate
  end subroutine solve_g

  !> X = A^{-1} B for nrhs right-hand sides (G form only), B(n, nrhs)
  subroutine spchol_solve_many(this, nrhs, b, x)
    class(spchol_t), intent(inout) :: this
    integer, intent(in) :: nrhs
    real(kind=dp), intent(in) :: b(this%n, nrhs)
    real(kind=dp), intent(out) :: x(this%n, nrhs)
    integer :: s, f, k, m, i, r0, c, rr
    integer(kind=i8) :: o
    if (.not. this%is_g) error stop 'q2d_spchol: solve_many needs the G form'
    if (allocated(this%wm)) then
       if (size(this%wm, 2) .ne. nrhs) deallocate(this%wm, this%tm)
    end if
    if (.not. allocated(this%wm)) allocate(this%wm(this%n, nrhs), this%tm(this%maxm, nrhs))
    associate(y => this%wm, t => this%tm)
      do c = 1, nrhs
         do i = 1, this%n
            y(i, c) = b(this%perm(i), c)
         end do
      end do
      do s = 1, this%nsuper
         f = this%sfc(s); k = this%sfc(s + 1) - f
         r0 = this%srp(s); m = this%srp(s + 1) - r0
         o = this%slp(s)
         call dgemm('N', 'N', m, nrhs, k, 1.0_dp, this%lx(o + 1), m, y(f, 1), this%n, 0.0_dp, t, &
              this%maxm)
         do c = 1, nrhs
            y(f:f + k - 1, c) = t(1:k, c)
            do i = k + 1, m
               rr = this%srow(r0 + i - 1)
               y(rr, c) = y(rr, c) - t(i, c)
            end do
         end do
      end do
      do s = this%nsuper, 1, -1
         f = this%sfc(s); k = this%sfc(s + 1) - f
         r0 = this%srp(s); m = this%srp(s + 1) - r0
         o = this%slp(s)
         do c = 1, nrhs
            t(1:k, c) = y(f:f + k - 1, c)
            do i = k + 1, m
               t(i, c) = -y(this%srow(r0 + i - 1), c)
            end do
         end do
         call dgemm('T', 'N', k, nrhs, m, 1.0_dp, this%lx(o + 1), m, t, this%maxm, 0.0_dp, y(f, 1), this%n)
      end do
      do c = 1, nrhs
         do i = 1, this%n
            x(this%perm(i), c) = y(i, c)
         end do
      end do
    end associate
  end subroutine spchol_solve_many

  !> y(rows(i)) -= sum_j L21(i, j) * xs(j)   (dgemv into a contiguous buffer)
  subroutine gemv_scatter(mm, k, l21, ld, xs, rows, y, t)
    integer, intent(in) :: mm, k, ld
    real(kind=dp), intent(in) :: l21(ld, *), xs(k)
    integer, intent(in) :: rows(mm)
    real(kind=dp), intent(inout) :: y(*), t(*)
    integer :: i
    call dgemv('N', mm, k, 1.0_dp, l21, ld, xs, 1, 0.0_dp, t, 1)
    do i = 1, mm
       y(rows(i)) = y(rows(i)) - t(i)
    end do
  end subroutine gemv_scatter

  !> xs(j) -= sum_i L21(i, j) * y(rows(i))   (gather, then dgemv)
  subroutine gemvt_gather(mm, k, l21, ld, rows, y, xs, t)
    integer, intent(in) :: mm, k, ld
    real(kind=dp), intent(in) :: l21(ld, *)
    integer, intent(in) :: rows(mm)
    real(kind=dp), intent(in) :: y(*)
    real(kind=dp), intent(inout) :: xs(k), t(*)
    integer :: i
    do i = 1, mm
       t(i) = y(rows(i))
    end do
    call dgemv('T', mm, k, -1.0_dp, l21, ld, t, 1, 1.0_dp, xs, 1)
  end subroutine gemvt_gather

  subroutine spchol_free(this)
    class(spchol_t), intent(inout) :: this
    if (allocated(this%perm)) deallocate(this%perm)
    if (allocated(this%iperm)) deallocate(this%iperm)
    if (allocated(this%sfc)) deallocate(this%sfc)
    if (allocated(this%sparent)) deallocate(this%sparent)
    if (allocated(this%srp)) deallocate(this%srp)
    if (allocated(this%srow)) deallocate(this%srow)
    if (allocated(this%slp)) deallocate(this%slp)
    if (allocated(this%lx)) deallocate(this%lx)
    if (allocated(this%acp)) deallocate(this%acp)
    if (allocated(this%aci)) deallocate(this%aci)
    if (allocated(this%acx)) deallocate(this%acx)
    if (allocated(this%work)) deallocate(this%work)
    if (allocated(this%tbuf)) deallocate(this%tbuf)
    if (allocated(this%wm)) deallocate(this%wm)
    if (allocated(this%tm)) deallocate(this%tm)
    this%is_g = .false.
    this%n = 0; this%nsuper = 0; this%maxm = 0; this%nnz_l = 0; this%flops = 0.0_dp
  end subroutine spchol_free

  ! ---------------------------------------------------------------------------
  ! ordering: nested dissection with level-set separators
  ! ---------------------------------------------------------------------------

  !> perm(new) = old. Separators are numbered after the two halves they
  !! separate, recursively; small parts in their BFS order.
  subroutine nd_order(n, ia, ja, leaf, perm)
    integer, intent(in) :: n, ia(n + 1), ja(*), leaf
    integer, allocatable, intent(out) :: perm(:)
    integer, allocatable :: tag(:), lvl(:), queue(:), sub(:), stk_off(:), stk_len(:), side(:)
    integer, allocatable :: part(:)
    integer :: nstk, nout, ntag, off, len, i, j, k, v, w, nq, ncomp, cstart
    integer :: root, nlev, ecc, it, lsep, cnt, half, na, nb, ns, hi

    allocate(perm(n), tag(n), lvl(n), queue(n), sub(n), side(n), part(n))
    ! work stack of subsets (offset, length into a subset buffer); output is
    ! filled from the end: separators get the highest numbers of their part
    allocate(stk_off(2 * n + 2), stk_len(2 * n + 2))
    tag = 0; ntag = 0
    do i = 1, n
       sub(i) = i
    end do
    nstk = 1; stk_off(1) = 1; stk_len(1) = n
    nout = n                       ! next position from the end
    do while (nstk .gt. 0)
       off = stk_off(nstk); len = stk_len(nstk); nstk = nstk - 1
       if (len .le. 0) cycle
       ! mark the subset
       ntag = ntag + 1
       do i = off, off + len - 1
          tag(sub(i)) = ntag
       end do
       ! connected components (BFS); if more than one, push them separately
       ncomp = 0; nq = 0
       ntag = ntag + 1
       do i = off, off + len - 1
          v = sub(i)
          if (tag(v) .ne. ntag - 1) cycle
          ncomp = ncomp + 1
          cstart = nq + 1
          nq = nq + 1; queue(nq) = v; tag(v) = ntag
          k = cstart
          do while (k .le. nq)
             w = queue(k); k = k + 1
             do j = ia(w), ia(w + 1) - 1
                if (tag(ja(j)) .eq. ntag - 1) then
                   nq = nq + 1; queue(nq) = ja(j); tag(ja(j)) = ntag
                end if
             end do
          end do
          part(ncomp) = nq - cstart + 1
       end do
       if (ncomp .gt. 1) then
          ! components in BFS order back into the buffer, one stack entry each
          sub(off:off + len - 1) = queue(1:len)
          k = off
          do i = 1, ncomp
             nstk = nstk + 1; stk_off(nstk) = k; stk_len(nstk) = part(i)
             k = k + part(i)
          end do
          cycle
       end if
       if (len .le. leaf) then
          ! leaf: number in BFS order (queue holds the component in BFS order)
          do i = len, 1, -1
             perm(nout) = queue(i); nout = nout - 1
          end do
          cycle
       end if
       ! pseudo-peripheral root: repeated BFS from a far node of minimum degree
       root = queue(1)
       do i = 1, len
          v = queue(i)
          if (ia(v + 1) - ia(v) .lt. ia(root + 1) - ia(root)) root = v
       end do
       ecc = -1
       do it = 1, 6
          lvl(sub(off:off + len - 1)) = 0
          call bfs_levels(root, ntag, tag, ia, ja, lvl, queue, nq, nlev)
          if (nlev .le. ecc) exit
          ecc = nlev
          ! node of minimum degree in the last level
          hi = queue(nq)
          do i = nq, 1, -1
             v = queue(i)
             if (lvl(v) .ne. nlev) exit
             if (ia(v + 1) - ia(v) .lt. ia(hi + 1) - ia(hi)) hi = v
          end do
          if (it .lt. 6) root = hi
       end do
       lvl(sub(off:off + len - 1)) = 0
       call bfs_levels(root, ntag, tag, ia, ja, lvl, queue, nq, nlev)
       if (nq .ne. len) error stop 'q2d_spchol: BFS did not reach the whole component'
       if (nlev .lt. 3) then
          ! (nearly) a clique: no useful separator, treat as a leaf
          do i = len, 1, -1
             perm(nout) = queue(i); nout = nout - 1
          end do
          cycle
       end if
       ! separator level: where the cumulative count crosses one half
       half = len / 2; cnt = 0; lsep = 2
       do i = 1, nq
          if (cnt .ge. half) exit
          cnt = cnt + 1
          lsep = lvl(queue(i))
       end do
       lsep = max(2, min(lsep, nlev - 1))
       ! sides: 1 below, 2 above, 3 separator; separator nodes without a
       ! neighbour above go below
       do i = 1, nq
          v = queue(i)
          if (lvl(v) .lt. lsep) then
             side(v) = 1
          else if (lvl(v) .gt. lsep) then
             side(v) = 2
          else
             side(v) = 3
          end if
       end do
       do i = 1, nq
          v = queue(i)
          if (side(v) .ne. 3) cycle
          k = 0
          do j = ia(v), ia(v + 1) - 1
             w = ja(j)
             if (tag(w) .eq. ntag) then
                if (side(w) .eq. 2) then
                   k = 1; exit
                end if
             end if
          end do
          if (k .eq. 0) side(v) = 1
       end do
       na = 0; nb = 0; ns = 0
       do i = 1, nq
          select case (side(queue(i)))
          case (1); na = na + 1
          case (2); nb = nb + 1
          case default; ns = ns + 1
          end select
       end do
       ! separator: highest numbers of this part
       do i = nq, 1, -1
          v = queue(i)
          if (side(v) .eq. 3) then
             perm(nout) = v; nout = nout - 1
          end if
       end do
       ! halves back into the buffer: [below | above]
       k = off; j = off + na
       do i = 1, nq
          v = queue(i)
          if (side(v) .eq. 1) then
             sub(k) = v; k = k + 1
          else if (side(v) .eq. 2) then
             sub(j) = v; j = j + 1
          end if
       end do
       nstk = nstk + 1; stk_off(nstk) = off; stk_len(nstk) = na
       nstk = nstk + 1; stk_off(nstk) = off + na; stk_len(nstk) = nb
    end do
    if (nout .ne. 0) error stop 'q2d_spchol: nested dissection lost nodes'
    deallocate(tag, lvl, queue, sub, side, part, stk_off, stk_len)
  end subroutine nd_order

  !> BFS from root within the nodes tagged ntag whose lvl is 0 on entry
  !! (the caller resets lvl over the subset): queue(1:nq) in BFS order,
  !! lvl(v) = level (1 = root), nlev levels.
  subroutine bfs_levels(root, ntag, tag, ia, ja, lvl, queue, nq, nlev)
    integer, intent(in) :: root, ntag, tag(:), ia(:), ja(*)
    integer, intent(inout) :: lvl(:), queue(:)
    integer, intent(out) :: nq, nlev
    integer :: k, w, j, u
    nq = 1; queue(1) = root; lvl(root) = 1; nlev = 1
    k = 1
    do while (k .le. nq)
       w = queue(k); k = k + 1
       do j = ia(w), ia(w + 1) - 1
          u = ja(j)
          if (tag(u) .ne. ntag) cycle
          if (lvl(u) .ne. 0) cycle
          nq = nq + 1; queue(nq) = u; lvl(u) = lvl(w) + 1
          nlev = max(nlev, lvl(u))
       end do
    end do
  end subroutine bfs_levels


  ! ---------------------------------------------------------------------------
  ! ordering: approximate minimum degree (Amestoy, Davis, Duff, SIAM J. Matrix
  ! Anal. Appl. 17, 1996), on the quotient graph: variables and elements,
  ! approximate external degrees from |Le \ Lp|, aggressive element
  ! absorption, supervariables of indistinguishable variables (hashing).
  ! Written from the paper; simple per-node list storage.
  ! ---------------------------------------------------------------------------

  !> AMD with dense rows set aside (review 6): rows with more than
  !! max(16, 10 sqrt(n)) off-diagonal entries are removed from the graph and
  !! numbered last, as in the reference AMD.
  subroutine amd_order(n, ia, ja, perm)
    integer, intent(in) :: n, ia(n + 1), ja(*)
    integer, allocatable, intent(out) :: perm(:)
    integer, allocatable :: newid(:), ib(:), jb(:), pb(:)
    integer :: i, k, ns, nd, dthr, nz
    dthr = max(16, int(10.0_dp * sqrt(real(n, dp))))
    allocate(newid(n))
    ns = 0
    do i = 1, n
       if (ia(i + 1) - ia(i) - 1 .gt. dthr) then
          newid(i) = 0
       else
          ns = ns + 1; newid(i) = ns
       end if
    end do
    if (ns .eq. n) then
       call amd_core(n, ia, ja, perm)
       return
    end if
    ! the graph of the sparse rows only
    allocate(ib(ns + 1))
    ib(1) = 1; nz = 0
    do i = 1, n
       if (newid(i) .eq. 0) cycle
       do k = ia(i), ia(i + 1) - 1
          if (newid(ja(k)) .gt. 0) nz = nz + 1
       end do
       ib(newid(i) + 1) = nz + 1
    end do
    allocate(jb(max(nz, 1)))
    nz = 0
    do i = 1, n
       if (newid(i) .eq. 0) cycle
       do k = ia(i), ia(i + 1) - 1
          if (newid(ja(k)) .gt. 0) then
             nz = nz + 1; jb(nz) = newid(ja(k))
          end if
       end do
    end do
    allocate(perm(n))
    if (ns .gt. 0) then
       call amd_core(ns, ib, jb, pb)
       ! back to the original numbering
       block
         integer, allocatable :: oldid(:)
         allocate(oldid(ns))
         do i = 1, n
            if (newid(i) .gt. 0) oldid(newid(i)) = i
         end do
         do k = 1, ns
            perm(k) = oldid(pb(k))
         end do
       end block
    end if
    nd = ns
    do i = 1, n
       if (newid(i) .eq. 0) then
          nd = nd + 1; perm(nd) = i
       end if
    end do
  end subroutine amd_order

  subroutine amd_core(n, ia, ja, perm)
    integer, intent(in) :: n, ia(n + 1), ja(*)
    integer, allocatable, intent(out) :: perm(:)
    type :: ilist_t
       integer, allocatable :: v(:)
       integer :: n = 0
    end type ilist_t
    type(ilist_t), allocatable :: el(:), al(:), le(:)
    integer, allocatable :: nv(:), deg(:), st(:), w(:), wflg(:), mark(:), esize(:)
    integer, allocatable :: bhead(:), bnext(:), bprev(:), lp(:), hsh(:), hhead(:), hnext(:)
    integer, allocatable :: mnext(:), mlast(:)
    integer :: i, j, k, e, p, nel, mindeg, stamp, nlp, degp, dsum, asum, d, iq, jq, nout, h
    logical :: same

    allocate(perm(n), el(n), al(n), le(n), nv(n), deg(n), st(n), w(n), wflg(n), mark(n), &
         esize(n), bhead(0:n), bnext(n), bprev(n), lp(n), hsh(n), hhead(0:n), hnext(n), &
         mnext(n), mlast(n))
    ! st: 0 principal variable, 1 element, 2 absorbed element, 3 non-principal variable
    st = 0; nv = 1; w = 0; wflg = 0; mark = 0; esize = 0; bhead = 0
    hhead = 0
    do i = 1, n
       mnext(i) = 0; mlast(i) = i
       k = 0
       do j = ia(i), ia(i + 1) - 1
          if (ja(j) .ne. i) k = k + 1
       end do
       allocate(al(i)%v(max(k, 1))); al(i)%n = 0
       do j = ia(i), ia(i + 1) - 1
          if (ja(j) .ne. i) then
             al(i)%n = al(i)%n + 1; al(i)%v(al(i)%n) = ja(j)
          end if
       end do
       allocate(el(i)%v(4)); el(i)%n = 0
       deg(i) = al(i)%n
       call bins(i)
    end do
    nel = 0; mindeg = 0; stamp = 0; nout = 0
    do while (nel .lt. n)
       ! pivot of minimum approximate degree
       do while (bhead(mindeg) .eq. 0)
          mindeg = mindeg + 1
       end do
       p = bhead(mindeg)
       call bdel(p)
       ! Lp = (A_p U union of Le over E_p) \ p, principal variables only
       stamp = stamp + 1
       mark(p) = stamp
       nlp = 0; degp = 0
       do k = 1, el(p)%n
          e = el(p)%v(k)
          if (st(e) .ne. 1) cycle
          do j = 1, le(e)%n
             i = le(e)%v(j)
             if (st(i) .eq. 0 .and. mark(i) .ne. stamp) then
                mark(i) = stamp; nlp = nlp + 1; lp(nlp) = i; degp = degp + nv(i)
             end if
          end do
          st(e) = 2
          if (allocated(le(e)%v)) deallocate(le(e)%v)
          le(e)%n = 0
       end do
       do k = 1, al(p)%n
          i = al(p)%v(k)
          if (st(i) .eq. 0 .and. mark(i) .ne. stamp) then
             mark(i) = stamp; nlp = nlp + 1; lp(nlp) = i; degp = degp + nv(i)
          end if
       end do
       deallocate(el(p)%v, al(p)%v); el(p)%n = 0; al(p)%n = 0
       st(p) = 1
       allocate(le(p)%v(max(nlp, 1))); le(p)%n = nlp; le(p)%v(1:nlp) = lp(1:nlp)
       esize(p) = degp
       nel = nel + nv(p)
       ! output p and the variables merged into it
       i = p
       do while (i .ne. 0)
          nout = nout + 1; perm(nout) = i
          i = mnext(i)
       end do
       ! w(e) = |Le \ Lp| for the elements adjacent to Lp
       do k = 1, nlp
          i = lp(k)
          call bdel(i)
          do j = 1, el(i)%n
             e = el(i)%v(j)
             if (st(e) .ne. 1) cycle
             if (wflg(e) .ne. stamp) then
                wflg(e) = stamp; w(e) = esize(e) - nv(i)
             else
                w(e) = w(e) - nv(i)
             end if
          end do
       end do
       ! update the lists and approximate degrees of the variables in Lp
       do k = 1, nlp
          i = lp(k)
          dsum = 0; jq = 0
          do j = 1, el(i)%n
             e = el(i)%v(j)
             if (st(e) .ne. 1 .or. e .eq. p) cycle
             if (w(e) .eq. 0) then
                st(e) = 2                       ! aggressive absorption: Le inside Lp
                if (allocated(le(e)%v)) deallocate(le(e)%v)
                le(e)%n = 0
                cycle
             end if
             dsum = dsum + w(e)
             jq = jq + 1; el(i)%v(jq) = e
          end do
          if (jq + 1 .gt. size(el(i)%v)) call grow(el(i), jq + 1)
          jq = jq + 1; el(i)%v(jq) = p; el(i)%n = jq
          asum = 0; iq = 0
          do j = 1, al(i)%n
             h = al(i)%v(j)
             if (st(h) .ne. 0 .or. mark(h) .eq. stamp .or. h .eq. i) cycle
             asum = asum + nv(h)
             iq = iq + 1; al(i)%v(iq) = h
          end do
          al(i)%n = iq
          d = min(deg(i) + degp - nv(i), asum + dsum + degp - nv(i))
          deg(i) = max(0, min(d, n - nel - nv(i)))
          ! hash of the (element, variable) lists for supervariable detection
          h = 0
          do j = 1, el(i)%n
             h = mod(h + el(i)%v(j), n)
          end do
          do j = 1, al(i)%n
             h = mod(h + al(i)%v(j), n)
          end do
          hsh(i) = h
          hnext(i) = hhead(h); hhead(h) = i
       end do
       ! supervariables: identical element and variable lists
       do k = 1, nlp
          i = lp(k)
          if (st(i) .ne. 0) cycle
          h = hsh(i)
          if (hhead(h) .eq. 0) cycle
          j = hnext(i)
          do while (j .ne. 0)
             if (st(j) .eq. 0 .and. hsh(j) .eq. h) then
                call compare(i, j, same)
                if (same) then
                   nv(i) = nv(i) + nv(j); deg(i) = max(0, deg(i) - nv(j))
                   nv(j) = 0; st(j) = 3
                   mnext(mlast(i)) = j; mlast(i) = mlast(j)
                   deallocate(el(j)%v, al(j)%v); el(j)%n = 0; al(j)%n = 0
                end if
             end if
             j = hnext(j)
          end do
       end do
       do k = 1, nlp
          hhead(hsh(lp(k))) = 0
       end do
       ! back into the degree lists
       do k = 1, nlp
          i = lp(k)
          if (st(i) .ne. 0) cycle
          call bins(i)
          mindeg = min(mindeg, deg(i))
       end do
       ! drop merged variables from Le(p)
       iq = 0
       do j = 1, le(p)%n
          if (st(le(p)%v(j)) .eq. 0) then
             iq = iq + 1; le(p)%v(iq) = le(p)%v(j)
          end if
       end do
       le(p)%n = iq
    end do
    if (nout .ne. n) error stop 'q2d_spchol: amd lost variables'
  contains
    subroutine bins(v)
      integer, intent(in) :: v
      bprev(v) = 0; bnext(v) = bhead(deg(v))
      if (bhead(deg(v)) .ne. 0) bprev(bhead(deg(v))) = v
      bhead(deg(v)) = v
    end subroutine bins
    subroutine bdel(v)
      integer, intent(in) :: v
      if (bprev(v) .ne. 0) then
         bnext(bprev(v)) = bnext(v)
      else
         if (bhead(deg(v)) .eq. v) bhead(deg(v)) = bnext(v)
      end if
      if (bnext(v) .ne. 0) bprev(bnext(v)) = bprev(v)
      bnext(v) = 0; bprev(v) = 0
    end subroutine bdel
    subroutine grow(l, m)
      type(ilist_t), intent(inout) :: l
      integer, intent(in) :: m
      integer, allocatable :: t(:)
      allocate(t(max(m, 2 * size(l%v))))
      t(1:size(l%v)) = l%v
      call move_alloc(t, l%v)
    end subroutine grow
    !> identical element and variable lists of a and b (as sets)
    subroutine compare(a, b, eq)
      integer, intent(in) :: a, b
      logical, intent(out) :: eq
      integer :: q
      eq = .false.
      if (el(a)%n .ne. el(b)%n .or. al(a)%n .ne. al(b)%n) return
      stamp = stamp + 1
      do q = 1, el(a)%n
         wflg(el(a)%v(q)) = -stamp
      end do
      do q = 1, el(b)%n
         if (wflg(el(b)%v(q)) .ne. -stamp) return
      end do
      do q = 1, al(a)%n
         mark(al(a)%v(q)) = -stamp
      end do
      do q = 1, al(b)%n
         if (mark(al(b)%v(q)) .ne. -stamp) return
      end do
      eq = .true.
    end subroutine compare
  end subroutine amd_core

  ! ---------------------------------------------------------------------------
  ! symbolic analysis
  ! ---------------------------------------------------------------------------

  subroutine symbolic(this, ia, ja)
    class(spchol_t), intent(inout) :: this
    integer, intent(in) :: ia(this%n + 1), ja(*)
    integer :: n, i, j, k, r, t, p, s, f, l, m, nch, ns
    integer, allocatable :: parent(:), anc(:), head(:), next(:), post(:), stk(:)
    integer, allocatable :: cc(:), mark(:), pp(:), pj(:), nchild(:), col2s(:)
    integer, allocatable :: sp(:), sc(:)
    integer(kind=i8) :: actual, stored
    real(kind=dp) :: z

    n = this%n
    do i = 1, n
       this%iperm(this%perm(i)) = i
    end do
    ! permuted adjacency (new numbering)
    call permuted_pattern(n, ia, ja, this%iperm, pp, pj)
    ! elimination tree
    allocate(parent(n), anc(n))
    parent = 0; anc = 0
    do i = 1, n
       do k = pp(i), pp(i + 1) - 1
          r = pj(k)
          if (r .ge. i) cycle
          do while (anc(r) .ne. 0 .and. anc(r) .ne. i)
             t = anc(r); anc(r) = i; r = t
          end do
          if (anc(r) .eq. 0) then
             anc(r) = i; parent(r) = i
          end if
       end do
    end do
    ! postorder (children lists, iterative DFS)
    allocate(head(n), next(n), post(n), stk(n))
    head = 0
    do j = n, 1, -1
       if (parent(j) .ne. 0) then
          next(j) = head(parent(j)); head(parent(j)) = j
       end if
    end do
    k = 0
    do j = 1, n
       if (parent(j) .ne. 0) cycle
       t = 1; stk(1) = j
       do while (t .gt. 0)
          p = stk(t)
          if (head(p) .ne. 0) then
             i = head(p); head(p) = next(i)
             t = t + 1; stk(t) = i
          else
             t = t - 1; k = k + 1; post(k) = p
          end if
       end do
    end do
    ! compose: perm_final(new) = perm(post(new))
    do i = 1, n
       stk(i) = this%perm(post(i))
    end do
    this%perm = stk
    do i = 1, n
       this%iperm(this%perm(i)) = i
    end do
    deallocate(pp, pj)
    call permuted_pattern(n, ia, ja, this%iperm, pp, pj)
    ! etree again in the final numbering (cheap, keeps it simple)
    parent = 0; anc = 0
    do i = 1, n
       do k = pp(i), pp(i + 1) - 1
          r = pj(k)
          if (r .ge. i) cycle
          do while (anc(r) .ne. 0 .and. anc(r) .ne. i)
             t = anc(r); anc(r) = i; r = t
          end do
          if (anc(r) .eq. 0) then
             anc(r) = i; parent(r) = i
          end if
       end do
    end do
    ! column counts by row-subtree traversal
    allocate(cc(n), mark(n), nchild(n))
    cc = 1; mark = 0; nchild = 0
    do j = 1, n
       if (parent(j) .ne. 0) nchild(parent(j)) = nchild(parent(j)) + 1
    end do
    do i = 1, n
       mark(i) = i
       do k = pp(i), pp(i + 1) - 1
          r = pj(k)
          if (r .ge. i) cycle
          do while (mark(r) .ne. i)
             cc(r) = cc(r) + 1; mark(r) = i
             r = parent(r)
          end do
       end do
    end do
    ! supernodes: chains of the etree, fundamental or relaxed
    allocate(sp(n + 1))
    ns = 1; sp(1) = 1
    actual = cc(1)
    do j = 2, n
       f = sp(ns)
       if (parent(j - 1) .eq. j) then
          if (nchild(j) .eq. 1 .and. cc(j - 1) .eq. cc(j) + 1) then   ! fundamental
             actual = actual + cc(j)
             cycle
          end if
          ! relaxed: stored entries of the merged block vs. actual
          m = (j - f + 1) + cc(j) - 1
          stored = int(j - f + 1, i8) * int(m, i8) - int(j - f + 1, i8) * int(j - f, i8) / 2
          z = real(stored - (actual + cc(j)), dp) / real(stored, dp)
          if ((j - f + 1 .le. this%nrelax .and. z .le. 0.5_dp) .or. &
               (j - f + 1 .le. 256 .and. z .le. 0.05_dp)) then
             actual = actual + cc(j)
             cycle
          end if
       end if
       ns = ns + 1; sp(ns) = j; actual = cc(j)
    end do
    sp(ns + 1) = n + 1
    this%nsuper = ns
    allocate(this%sfc(ns + 1), this%sparent(ns), col2s(n))
    this%sfc = sp(1:ns + 1)
    do s = 1, ns
       do j = this%sfc(s), this%sfc(s + 1) - 1
          col2s(j) = s
       end do
    end do
    do s = 1, ns
       l = this%sfc(s + 1) - 1
       if (parent(l) .eq. 0) then
          this%sparent(s) = 0
       else
          this%sparent(s) = col2s(parent(l))
       end if
    end do
    ! row structures: columns of s, then rows > l from A and the children
    allocate(this%srp(ns + 1), sc(ns))
    this%srp(1) = 1
    do s = 1, ns
       l = this%sfc(s + 1) - 1
       this%srp(s + 1) = this%srp(s) + (l - this%sfc(s) + 1) + cc(l) - 1
    end do
    allocate(this%srow(this%srp(ns + 1) - 1))
    ! children lists of supernodes
    head(1:ns) = 0
    do s = ns, 1, -1
       if (this%sparent(s) .ne. 0) then
          next(s) = head(this%sparent(s)); head(this%sparent(s)) = s
       end if
    end do
    mark = 0
    do s = 1, ns
       f = this%sfc(s); l = this%sfc(s + 1) - 1
       p = this%srp(s)
       do j = f, l
          this%srow(p) = j; p = p + 1
       end do
       t = p
       ! rows of A in the columns of s below l (lower part of the symmetric pattern)
       do j = f, l
          do k = pp(j), pp(j + 1) - 1
             r = pj(k)
             if (r .gt. l .and. mark(r) .ne. s) then
                mark(r) = s; this%srow(p) = r; p = p + 1
             end if
          end do
       end do
       ! children's rows below l
       i = head(s)
       do while (i .ne. 0)
          do k = this%srp(i) + (this%sfc(i + 1) - this%sfc(i)), this%srp(i + 1) - 1
             r = this%srow(k)
             if (r .gt. l .and. mark(r) .ne. s) then
                mark(r) = s; this%srow(p) = r; p = p + 1
             end if
          end do
          i = next(i)
       end do
       if (p .ne. this%srp(s + 1)) error stop 'q2d_spchol: row structure count mismatch'
       call isort(this%srow(t:p - 1))
    end do
    ! factor storage
    allocate(this%slp(ns + 1))
    this%slp(1) = 0
    this%maxm = 0
    do s = 1, ns
       m = this%srp(s + 1) - this%srp(s)
       k = this%sfc(s + 1) - this%sfc(s)
       this%slp(s + 1) = this%slp(s) + int(m, i8) * int(k, i8)
       this%maxm = max(this%maxm, m)
    end do
    this%nnz_l = this%slp(ns + 1)
    allocate(this%work(n), this%tbuf(max(this%maxm, 1)))
    deallocate(parent, anc, head, next, post, stk, cc, mark, nchild, col2s, sp, sc, pp, pj)
    nch = 0
  end subroutine symbolic

  !> Nonzeros of L (column counts incl. the diagonal) for the ordering
  !! perm(new) = old, without supernodal padding.
  function predict_nnz(n, ia, ja, perm) result(nnz)
    integer, intent(in) :: n, ia(n + 1), ja(*), perm(n)
    integer(kind=i8) :: nnz
    integer, allocatable :: iperm(:), pp(:), pj(:), parent(:), anc(:), mark(:)
    integer :: i, k, r, t
    allocate(iperm(n), parent(n), anc(n), mark(n))
    do i = 1, n
       iperm(perm(i)) = i
    end do
    call permuted_pattern(n, ia, ja, iperm, pp, pj)
    parent = 0; anc = 0
    do i = 1, n
       do k = pp(i), pp(i + 1) - 1
          r = pj(k)
          if (r .ge. i) cycle
          do while (anc(r) .ne. 0 .and. anc(r) .ne. i)
             t = anc(r); anc(r) = i; r = t
          end do
          if (anc(r) .eq. 0) then
             anc(r) = i; parent(r) = i
          end if
       end do
    end do
    nnz = n; mark = 0
    do i = 1, n
       mark(i) = i
       do k = pp(i), pp(i + 1) - 1
          r = pj(k)
          if (r .ge. i) cycle
          do while (mark(r) .ne. i)
             nnz = nnz + 1; mark(r) = i
             r = parent(r)
          end do
       end do
    end do
  end function predict_nnz

  !> Pattern of P A P^T (both triangles, diagonal excluded), new numbering.
  subroutine permuted_pattern(n, ia, ja, iperm, pp, pj)
    integer, intent(in) :: n, ia(n + 1), ja(*), iperm(n)
    integer, allocatable, intent(out) :: pp(:), pj(:)
    integer :: i, k, ni, nj
    integer, allocatable :: fill(:)
    allocate(pp(n + 1), fill(n))
    pp = 0
    do i = 1, n
       ni = iperm(i)
       do k = ia(i), ia(i + 1) - 1
          if (ja(k) .ne. i) pp(ni + 1) = pp(ni + 1) + 1
       end do
    end do
    pp(1) = 1
    do i = 1, n
       pp(i + 1) = pp(i + 1) + pp(i)
    end do
    allocate(pj(max(pp(n + 1) - 1, 1)))
    fill = pp(1:n)
    do i = 1, n
       ni = iperm(i)
       do k = ia(i), ia(i + 1) - 1
          if (ja(k) .eq. i) cycle
          nj = iperm(ja(k))
          pj(fill(ni)) = nj; fill(ni) = fill(ni) + 1
       end do
    end do
    deallocate(fill)
  end subroutine permuted_pattern

  !> Lower triangle of P A P^T as CSC (new numbering).
  subroutine fill_lower(this, ia, ja, a)
    class(spchol_t), intent(inout) :: this
    integer, intent(in) :: ia(this%n + 1), ja(*)
    real(kind=dp), intent(in) :: a(*)
    integer :: n, i, k, ni, nj, c
    integer, allocatable :: fill(:)
    n = this%n
    if (allocated(this%acp)) deallocate(this%acp, this%aci, this%acx)
    allocate(this%acp(n + 1), fill(n))
    this%acp = 0
    do i = 1, n
       ni = this%iperm(i)
       do k = ia(i), ia(i + 1) - 1
          nj = this%iperm(ja(k))
          if (ni .ge. nj) this%acp(nj + 1) = this%acp(nj + 1) + 1
       end do
    end do
    this%acp(1) = 1
    do i = 1, n
       this%acp(i + 1) = this%acp(i + 1) + this%acp(i)
    end do
    allocate(this%aci(this%acp(n + 1) - 1), this%acx(this%acp(n + 1) - 1))
    fill = this%acp(1:n)
    do i = 1, n
       ni = this%iperm(i)
       do k = ia(i), ia(i + 1) - 1
          nj = this%iperm(ja(k))
          if (ni .ge. nj) then
             c = nj
             this%aci(fill(c)) = ni; this%acx(fill(c)) = a(k); fill(c) = fill(c) + 1
          end if
       end do
    end do
    deallocate(fill)
  end subroutine fill_lower

  ! ---------------------------------------------------------------------------
  ! numeric factorization (multifrontal)
  ! ---------------------------------------------------------------------------

  subroutine multifrontal(this, info)
    class(spchol_t), intent(inout) :: this
    integer, intent(out) :: info
    type :: upd_t
       real(kind=dp), allocatable :: u(:,:)
    end type upd_t
    type(upd_t), allocatable :: upd(:)
    real(kind=dp), allocatable :: fr(:,:)
    integer, allocatable :: pos(:), head(:), next(:)
    integer :: s, c, f, k, m, mc, kc, i, j, r0, rc0, ii, jj, ns
    integer(kind=i8) :: o

    info = 0
    ns = this%nsuper
    allocate(upd(ns), pos(this%n), head(ns), next(ns))
    allocate(fr(this%maxm, this%maxm))
    head = 0
    do s = ns, 1, -1
       if (this%sparent(s) .ne. 0) then
          next(s) = head(this%sparent(s)); head(this%sparent(s)) = s
       end if
    end do
    this%flops = 0.0_dp
    do s = 1, ns
       f = this%sfc(s); k = this%sfc(s + 1) - f
       r0 = this%srp(s); m = this%srp(s + 1) - r0
       do i = 1, m
          pos(this%srow(r0 + i - 1)) = i
       end do
       fr(1:m, 1:m) = 0.0_dp
       ! original entries of the columns of s (lower triangle)
       do j = 1, k
          do ii = this%acp(f + j - 1), this%acp(f + j) - 1
             fr(pos(this%aci(ii)), j) = fr(pos(this%aci(ii)), j) + this%acx(ii)
          end do
       end do
       ! extend-add of the children's update matrices
       c = head(s)
       do while (c .ne. 0)
          kc = this%sfc(c + 1) - this%sfc(c)
          rc0 = this%srp(c) + kc
          mc = this%srp(c + 1) - rc0
          do jj = 1, mc
             j = pos(this%srow(rc0 + jj - 1))
             do ii = jj, mc
                i = pos(this%srow(rc0 + ii - 1))
                fr(i, j) = fr(i, j) + upd(c)%u(ii, jj)
             end do
          end do
          deallocate(upd(c)%u)
          c = next(c)
       end do
       ! partial dense factorization of the front
       call dpotrf('L', k, fr, this%maxm, info)
       if (info .ne. 0) then
          info = s
          return
       end if
       if (m .gt. k) then
          call dtrsm('R', 'L', 'T', 'N', m - k, k, 1.0_dp, fr, this%maxm, fr(k + 1, 1), this%maxm)
          allocate(upd(s)%u(m - k, m - k))
          upd(s)%u = 0.0_dp
          do j = 1, m - k
             do i = j, m - k
                upd(s)%u(i, j) = fr(k + i, k + j)
             end do
          end do
          call dsyrk('L', 'N', m - k, k, -1.0_dp, fr(k + 1, 1), this%maxm, 1.0_dp, upd(s)%u, m - k)
       end if
       o = this%slp(s)
       do j = 1, k
          this%lx(o + int(j - 1, i8) * m + 1 : o + int(j, i8) * m) = fr(1:m, j)
       end do
       this%flops = this%flops + real(k, dp)**3 / 3.0_dp + real(m - k, dp) * real(k, dp)**2 &
            + real(m - k, dp)**2 * real(k, dp)
    end do
    deallocate(upd, pos, head, next, fr)
  end subroutine multifrontal

  !> Insertion / shell sort of a short integer list.
  subroutine isort(a)
    integer, intent(inout) :: a(:)
    integer :: i, j, gap, t
    gap = size(a) / 2
    do while (gap .gt. 0)
       do i = gap + 1, size(a)
          t = a(i); j = i
          do while (j .gt. gap)
             if (a(j - gap) .le. t) exit
             a(j) = a(j - gap); j = j - gap
          end do
          a(j) = t
       end do
       gap = gap / 2
    end do
  end subroutine isort

end module q2d_spchol
