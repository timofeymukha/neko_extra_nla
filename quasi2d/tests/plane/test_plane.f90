!> Standalone test of q2d_plane: a 2D SEM-like SPD pattern (E x E elements,
!! lx = 8, or interleaved pairs with -2), nm modes (diagonal shifts), nr
!! right-hand sides; residual of the distributed solve on rank 0, timings.
!! usage: mpirun -n P test_plane E [nr] [nm] [-2]
program test_plane
  use mpi_f08
  use comm, only : NEKO_COMM, pe_rank, pe_size
  use num_types, only : dp, i8
  use q2d_plane, only : plane_t, q2d_split_tol, q2d_lpt_group, q2d_tdist
  implicit none
  integer :: E, nr, nm, lx, np, nx, n, nn, i, j, k, ii, jj, ierr, md, c, l, bs, bi, bj
  integer, allocatable :: ia(:), ja(:), cnt(:), modes(:), rc(:), rd(:), ib(:), crow(:), ccol(:), cidx(:)
  real(kind=dp), allocatable :: cval(:)
  integer :: pin, nc
  real(kind=dp), allocatable :: av(:), a(:), Y(:,:,:), x(:,:,:), sb(:), rb(:)
  real(kind=dp) :: t0, t1, t2, t3, res, bmax, shift(4), v
  type(plane_t) :: pl
  character(len=32) :: arg
  call MPI_Init(ierr)
  NEKO_COMM = MPI_COMM_WORLD
  call MPI_Comm_rank(NEKO_COMM, pe_rank, ierr)
  call MPI_Comm_size(NEKO_COMM, pe_size, ierr)
  call get_command_argument(1, arg); read(arg, *) E
  nr = 1; nm = 1; bs = 1
  call get_command_argument(2, arg); if (len_trim(arg) .gt. 0) read(arg, *) nr
  call get_command_argument(3, arg); if (len_trim(arg) .gt. 0) read(arg, *) nm
  call get_command_argument(4, arg); if (trim(arg) .eq. '-2') bs = 2
  call get_environment_variable('Q2D_ORDERING', arg)
  if (len_trim(arg) .gt. 0) read(arg, *) pl%ordering
  shift = [1.0e-3_dp, 1.0_dp, 10.0_dp, 100.0_dp]
  lx = 8; np = lx - 1; nx = E * np + 1; nn = nx * nx; n = bs * nn
  ! pattern on every rank (rank 0 needs it; the others only n)
  allocate(cnt(n), ia(n + 1))
  do j = 1, nx
     do i = 1, nx
        do bi = 1, bs
           cnt(bs * (i + nx * (j - 1) - 1) + bi) = bs * span(i) * span(j)
        end do
     end do
  end do
  ia(1) = 1
  do i = 1, n
     ia(i + 1) = ia(i) + cnt(i)
  end do
  allocate(ja(ia(n + 1) - 1), av(ia(n + 1) - 1))
  do j = 1, nx
     do i = 1, nx
        do bi = 1, bs
           k = ia(bs * (i + nx * (j - 1) - 1) + bi)
           do jj = lo(j), hi(j)
              do ii = lo(i), hi(i)
                 do bj = 1, bs
                    ja(k) = bs * (ii + nx * (jj - 1) - 1) + bj
                    av(k) = -0.1_dp * (1.0_dp + 0.5_dp * sin(0.37_dp * (min(ja(k), bs * (i + nx * (j - 1) - 1) + bi) &
                         + 3 * max(ja(k), bs * (i + nx * (j - 1) - 1) + bi))))
                    k = k + 1
                 end do
              end do
           end do
        end do
     end do
  end do
  ! diagonal: sum of the off-diagonal magnitudes (shift per mode added later)
  do i = 1, n
     v = 0.0_dp
     do k = ia(i), ia(i + 1) - 1
        if (ja(k) .ne. i) v = v + abs(av(k))
     end do
     do k = ia(i), ia(i + 1) - 1
        if (ja(k) .eq. i) av(k) = v
     end do
  end do
  call get_environment_variable('Q2D_TOL', arg)
  if (len_trim(arg) .gt. 0) read(arg, *) q2d_split_tol
  call get_environment_variable('Q2D_LPT', arg)
  if (len_trim(arg) .gt. 0) read(arg, *) q2d_lpt_group
  call get_environment_variable('Q2D_DRY', arg)
  if (len_trim(arg) .gt. 0) then
     ! mapping only, for the rank counts given (comma separated)
     block
       integer :: pv(16), npv, ip
       integer(kind=i8), allocatable :: loads(:)
       pv = 0
       read(arg, *, end = 10) pv
10     npv = count(pv .gt. 0)
       do ip = 1, npv
          allocate(loads(0:3 * pv(ip) - 1))
          t0 = MPI_Wtime()
          call pl%dry_map(n, ia, ja, pv(ip), loads)
          associate(lt => loads(0:pv(ip) - 1) + loads(pv(ip):), ll => loads(0:pv(ip) - 1), &
               ldd => loads(pv(ip):2 * pv(ip) - 1), lnd => loads(2 * pv(ip):))
            print '(A,I0,A,I0,A,2F6.3,A,2F6.3,A,F6.3,A,I0,A,I0,A,F7.2,A)', 'dry P ', pv(ip), ' n ', n, &
                 '  G entries max/mean, min/mean ', real(maxval(lt), dp) / (real(sum(lt), dp) / pv(ip)), &
                 real(minval(lt), dp) / (real(sum(lt), dp) / pv(ip)), '; local only ', &
                 real(maxval(ll), dp) / (real(sum(ll), dp) / pv(ip)), &
                 real(minval(ll), dp) / (real(sum(ll), dp) / pv(ip)), '; distributed share ', &
                 real(sum(ldd), dp) / real(sum(lt), dp), '; dist. fronts per rank max ', maxval(lnd), &
                 ' min ', minval(lnd), '  (', MPI_Wtime() - t0, ' s)'
          end associate
          deallocate(loads)
       end do
     end block
     call MPI_Finalize(ierr)
     stop
  end if
  call get_environment_variable('Q2D_PIN', arg)
  pin = 0
  if (len_trim(arg) .gt. 0) pin = n
  t0 = MPI_Wtime()
  call pl%setup(n, nm, ia, ja)
  ! contributions: every entry in two halves, from the ranks i mod P and (i + 1) mod P
  nc = 0
  do i = 1, n
     if (mod(i, pe_size) .eq. pe_rank .or. mod(i + 1, pe_size) .eq. pe_rank) nc = nc + ia(i + 1) - ia(i)
  end do
  allocate(crow(nc), ccol(nc), cval(nc), cidx(nc))
  nc = 0
  do i = 1, n
     if (mod(i, pe_size) .ne. pe_rank .and. mod(i + 1, pe_size) .ne. pe_rank) cycle
     do k = ia(i), ia(i + 1) - 1
        nc = nc + 1
        crow(nc) = i; ccol(nc) = ja(k); cidx(nc) = k
     end do
  end do
  call pl%contrib_setup(nc, crow, ccol, pin)
  t1 = MPI_Wtime()
  do md = 1, nm
     do l = 1, nc
        k = cidx(l)
        cval(l) = av(k)
        if (ja(k) .eq. crow(l)) cval(l) = cval(l) + shift(md)
        if (pe_size .gt. 1) cval(l) = 0.5_dp * cval(l)
     end do
     call pl%factor(md, nc, cval, pin .gt. 0)
  end do
  t2 = MPI_Wtime()
  ! right-hand sides at the own columns
  allocate(Y(max(pl%ncol + pl%next, 1), nr, nm), modes(nm))
  Y = 0.0_dp
  do md = 1, nm
     modes(md) = md
     do c = 1, nr
        do l = 1, pl%ncol
           Y(l, c, md) = rhs(pl%cold(l), c, md)
        end do
     end do
  end do
  call MPI_Barrier(NEKO_COMM, ierr)
  t3 = MPI_Wtime()
  call pl%sweep(nm, modes, nr, size(Y, 1), Y)
  call MPI_Barrier(NEKO_COMM, ierr)
  t3 = MPI_Wtime() - t3
  ! gather (cold, x) to rank 0
  allocate(rc(0:pe_size - 1), rd(0:pe_size - 1))
  call MPI_Gather(pl%ncol, 1, MPI_INTEGER, rc, 1, MPI_INTEGER, 0, NEKO_COMM, ierr)
  rd(0) = 0
  do i = 1, pe_size - 1
     rd(i) = rd(i - 1) + rc(i - 1)
  end do
  allocate(ib(max(n, 1)), sb(max(pl%ncol * nr * nm, 1)), rb(max(n * nr * nm, 1)))
  call MPI_Gatherv(pl%cold, pl%ncol, MPI_INTEGER, ib, rc, rd, MPI_INTEGER, 0, NEKO_COMM, ierr)
  do md = 1, nm
     do c = 1, nr
        sb(1 + pl%ncol * ((c - 1) + nr * (md - 1)):pl%ncol * (c + nr * (md - 1))) = Y(1:pl%ncol, c, md)
     end do
  end do
  call MPI_Gatherv(sb, pl%ncol * nr * nm, MPI_DOUBLE_PRECISION, rb, rc * nr * nm, rd * nr * nm, &
       MPI_DOUBLE_PRECISION, 0, NEKO_COMM, ierr)
  if (pe_rank .eq. 0) then
     allocate(x(n, nr, nm))
     do i = 0, pe_size - 1
        do md = 1, nm
           do c = 1, nr
              do l = 1, rc(i)
                 x(ib(rd(i) + l), c, md) = rb(rd(i) * nr * nm + l + rc(i) * ((c - 1) + nr * (md - 1)))
              end do
           end do
        end do
     end do
     res = 0.0_dp; bmax = 0.0_dp
     do md = 1, nm
        do c = 1, nr
           do i = 1, n
              v = rhs(i, c, md)
              bmax = max(bmax, abs(v))
              if (i .eq. pin) then
                 v = v - x(i, c, md)
              else
                 do k = ia(i), ia(i + 1) - 1
                    if (ja(k) .eq. pin) cycle
                    v = v - av(k) * x(ja(k), c, md)
                    if (ja(k) .eq. i) v = v - shift(md) * x(i, c, md)
                 end do
              end if
              res = max(res, abs(v))
           end do
        end do
     end do
     print '(A,I0,A,I0,A,I0,A,I0,A,I0,A,ES9.2)', 'P ', pe_size, ' n ', n, ' nnz(L) ', pl%nnz_l, ' nr ', nr, &
          ' nm ', nm, '  max residual / max rhs ', res / bmax
     print '(A,I0)', '  ordering ', pl%ordering_used
     print '(A,F8.3,A,F8.3,A,F8.4,A)', '  setup (incl. contributions) ', t1 - t0, ' s, factor ', t2 - t1, &
          ' s, sweep ', t3, ' s'
     print '(A,4F8.4,A,3F8.3)', '  sweep (rank 0): local fwd, dist fwd, dist bwd, local bwd', pl%tsw, &
          '; factor: values, local, distributed', pl%tfac
     print '(A,I0,A,I0,A,I0,A,I0)', '  G entries per rank: max ', pl%ld_max, ' mean ', pl%ld_mean, &
          '; distributed fronts (max per rank) ', pl%nd_max, '; depth ', pl%dmax
  end if
  call get_environment_variable('Q2D_ALLTIMES', arg)
  if (len_trim(arg) .gt. 0) then
     do i = 0, pe_size - 1
        call MPI_Barrier(NEKO_COMM, ierr)
        if (i .eq. pe_rank) print '(A,I4,A,4F8.4,A,3F8.3,A,I0,A,3F8.3)', '  rank ', pe_rank, ' sweep', pl%tsw, &
             '  factor', pl%tfac, '  G entries ', pl%boff(pl%nsup + 1), &
             '  dist: asm, chol, G', q2d_tdist
     end do
  end if
  call pl%free()
  call MPI_Finalize(ierr)
contains
  real(kind=dp) function rhs(i, c, md)
    integer, intent(in) :: i, c, md
    rhs = sin(0.013_dp * i * c + md) + 0.1_dp * c
  end function rhs
  integer function lo(i)
    integer, intent(in) :: i
    lo = max(1, ((i - 2) / np) * np + 1)
    if (mod(i - 1, np) .eq. 0 .and. i .gt. 1) lo = i - np
  end function lo
  integer function hi(i)
    integer, intent(in) :: i
    hi = min(nx, ((i - 1) / np + 1) * np + 1)
    if (mod(i - 1, np) .eq. 0 .and. i .lt. nx) hi = i + np
    if (mod(i - 1, np) .eq. 0 .and. i .eq. nx) hi = nx
  end function hi
  integer function span(i)
    integer, intent(in) :: i
    span = hi(i) - lo(i) + 1
  end function span
end program test_plane
