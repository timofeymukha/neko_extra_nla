!> Diagnostic CG for the user-type injection path of `ksp_t`.
!!
!! Textbook preconditioned CG that records the CG coefficients alpha_j and
!! beta_j and, optionally, the M-orthonormal Lanczos vectors
!! v_j = z_j / sqrt(r_j^T z_j). After the solve it assembles the Lanczos
!! tridiagonal T_k (Saad, Iterative Methods for Sparse Linear Systems,
!! sec. 6.7.3), computes its eigenpairs with LAPACK dstev and reports the
!! Ritz values of M^{-1} A (log line "LANCZOS {...}", file <prefix>_ritz.txt),
!! the slowest Ritz pairs with an accuracy check and their variation along
!! the three reference directions inside the elements (log lines
!! "RITZ {...}"), and writes the slowest Ritz vectors to <prefix>.fld.
!!
!! The `ksp_t` init interface receives no JSON, so the options come from
!! environment variables:
!!   RS_LANCZOS_PREFIX    output prefix                       (lanczos)
!!   RS_LANCZOS_NRITZ     number of slowest Ritz vectors kept (4, 0 = off)
!!   RS_LANCZOS_MAXSTORE  maximum number of Lanczos vectors stored (800)
!!   RS_LANCZOS_FLD       1 = write the Ritz vectors to <prefix>.fld (1)
!!
!! The residual recurrence is the same as in Neko's cpu `cg_t`, so the
!! iteration counts are identical. CPU backend only; dstev needs rp = dp.
module rs_cg_lanczos
  use num_types, only : rp, xp, dp
  use krylov, only : ksp_t, ksp_monitor_t, krylov_allocate, register_krylov
  use precon, only : pc_t
  use ax_product, only : ax_t
  use field, only : field_t
  use coefs, only : coef_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use scalar_bc_projector, only : scalar_bc_projector_t
  use vector_bc_projector, only : vector_bc_projector_t, &
       vector_bc_projector_components
  use math, only : glsc3, abscmp
  use comm, only : pe_rank, NEKO_COMM, MPI_REAL_PRECISION, MPI_EXTRA_PRECISION
  use mpi_f08, only : MPI_Allreduce, MPI_IN_PLACE, MPI_SUM, &
       MPI_DOUBLE_PRECISION
  use file, only : file_t
  use space, only : space_t, GLL
  use interpolation, only : interpolator_t
  use tensor, only : tnsr3d
  use logger, only : neko_log, LOG_SIZE
  use utils, only : neko_error
  use neko_config, only : NEKO_BCKND_DEVICE
  implicit none
  private

  type, public, extends(ksp_t) :: rs_cg_lanczos_t
     real(kind=rp), allocatable :: w(:), r(:), p(:), z(:)
     real(kind=rp), allocatable :: alpha(:), beta(:)
     !> Lanczos vectors, (n, max_store)
     real(kind=rp), allocatable :: v(:,:)
     integer :: max_store = 800
     integer :: n_ritz = 4
     logical :: write_fld = .true.
     character(len=256) :: prefix = 'lanczos'
     integer :: n_solve = 0
     !> Indefiniteness counters of the last solve (r^T z < 0, alpha < 0)
     integer :: n_neg_rtz = 0
     integer :: n_neg_alpha = 0
   contains
     procedure, pass(this) :: init => rs_cg_lanczos_init
     procedure, pass(this) :: free => rs_cg_lanczos_free
     procedure, pass(this) :: solve => rs_cg_lanczos_solve
     procedure, pass(this) :: solve_coupled => rs_cg_lanczos_solve_coupled
  end type rs_cg_lanczos_t

  public :: rs_cg_lanczos_register_types

contains

  subroutine rs_cg_lanczos_init(this, n, max_iter, M, rel_tol, abs_tol, monitor)
    class(rs_cg_lanczos_t), intent(inout), target :: this
    integer, intent(in) :: max_iter
    class(pc_t), optional, intent(in), target :: M
    integer, intent(in) :: n
    real(kind=rp), optional, intent(in) :: rel_tol
    real(kind=rp), optional, intent(in) :: abs_tol
    logical, optional, intent(in) :: monitor
    character(len=256) :: buf
    character(len=LOG_SIZE) :: log_buf
    integer :: length, status, ival

    if (NEKO_BCKND_DEVICE .eq. 1) then
       call neko_error('rs_cg_lanczos: CPU backend only')
    end if
    if (rp .ne. dp) then
       call neko_error('rs_cg_lanczos: needs a double precision build (dstev)')
    end if

    call this%free()
    if (present(M)) this%M => M
    call this%ksp_init(max_iter, rel_tol = rel_tol, abs_tol = abs_tol, &
         monitor = monitor)

    call get_environment_variable('RS_LANCZOS_PREFIX', buf, length, status)
    if (status .eq. 0 .and. length .gt. 0) this%prefix = buf(1:length)
    call get_environment_variable('RS_LANCZOS_NRITZ', buf, length, status)
    if (status .eq. 0 .and. length .gt. 0) then
       read(buf(1:length), *) ival
       this%n_ritz = max(0, ival)
    end if
    call get_environment_variable('RS_LANCZOS_MAXSTORE', buf, length, status)
    if (status .eq. 0 .and. length .gt. 0) then
       read(buf(1:length), *) ival
       this%max_store = max(1, ival)
    end if
    call get_environment_variable('RS_LANCZOS_FLD', buf, length, status)
    if (status .eq. 0 .and. length .gt. 0) then
       read(buf(1:length), *) ival
       this%write_fld = ival .ne. 0
    end if

    call neko_log%section('rs_cg_lanczos')
    write(log_buf, '(A,A)') 'prefix    : ', trim(this%prefix)
    call neko_log%message(log_buf)
    write(log_buf, '(A,I0,A,I0,A,L1)') 'n_ritz    : ', this%n_ritz, &
         ', max_store: ', this%max_store, ', fld: ', this%write_fld
    call neko_log%message(log_buf)
    write(log_buf, '(A,I0,A,I0)') 'n (local) : ', n, ', max_iter: ', max_iter
    call neko_log%message(log_buf)
    call neko_log%end_section()
  end subroutine rs_cg_lanczos_init

  subroutine rs_cg_lanczos_free(this)
    class(rs_cg_lanczos_t), intent(inout) :: this

    call this%ksp_free()
    if (allocated(this%w)) deallocate(this%w)
    if (allocated(this%r)) deallocate(this%r)
    if (allocated(this%p)) deallocate(this%p)
    if (allocated(this%z)) deallocate(this%z)
    if (allocated(this%alpha)) deallocate(this%alpha)
    if (allocated(this%beta)) deallocate(this%beta)
    if (allocated(this%v)) deallocate(this%v)
    nullify(this%M)
  end subroutine rs_cg_lanczos_free

  function rs_cg_lanczos_solve(this, Ax, x, f, n, coef, bc_projector, gs_h, &
       niter) result(ksp_results)
    class(rs_cg_lanczos_t), intent(inout) :: this
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(in) :: f
    type(coef_t), intent(inout) :: coef
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    integer, optional, intent(in) :: niter
    type(ksp_monitor_t) :: ksp_results
    integer :: iter, max_iter, i, k, m, ierr, nstore
    real(kind=rp) :: rnorm, rtr, rtz1, rtz2, beta, pap, alpha, norm_fac
    real(kind=xp) :: tmp

    if (present(niter)) then
       max_iter = niter
    else
       max_iter = this%max_iter
    end if
    norm_fac = 1.0_rp / sqrt(coef%volume)

    if (allocated(this%w)) then
       if (size(this%w) .ne. n) deallocate(this%w, this%r, this%p, this%z)
    end if
    if (.not. allocated(this%w)) then
       allocate(this%w(n), this%r(n), this%p(n), this%z(n))
    end if
    if (allocated(this%alpha)) deallocate(this%alpha, this%beta)
    allocate(this%alpha(max_iter), this%beta(max_iter))
    nstore = 0
    if (this%n_ritz .gt. 0) then
       nstore = min(this%max_store, max_iter)
       if (allocated(this%v)) deallocate(this%v)
       allocate(this%v(n, nstore))
    end if

    k = 0
    m = 0
    this%n_neg_rtz = 0
    this%n_neg_alpha = 0
    associate(w => this%w, r => this%r, p => this%p, z => this%z)
      rtz1 = 1.0_rp
      rtr = 0.0_rp
      do i = 1, n
         x%x(i,1,1,1) = 0.0_rp
         p(i) = 0.0_rp
         r(i) = f(i)
         rtr = rtr + r(i) * coef%mult(i,1,1,1) * r(i)
      end do
      call MPI_Allreduce(MPI_IN_PLACE, rtr, 1, MPI_REAL_PRECISION, MPI_SUM, &
           NEKO_COMM, ierr)
      rnorm = sqrt(rtr) * norm_fac
      ksp_results%res_start = rnorm
      ksp_results%res_final = rnorm
      ksp_results%iter = 0
      if (abscmp(rnorm, 0.0_rp)) then
         ksp_results%converged = .true.
         return
      end if

      call this%monitor_start('CG')
      do iter = 1, max_iter
         call this%M%solve(z, r, n)
         rtz2 = rtz1
         rtz1 = glsc3(r, coef%mult, z, n)
         if (rtz1 .lt. 0.0_rp) this%n_neg_rtz = this%n_neg_rtz + 1
         beta = rtz1 / rtz2
         if (iter .eq. 1) beta = 0.0_rp
         do i = 1, n
            p(i) = z(i) + beta * p(i)
         end do

         call Ax%compute(w, p, coef, x%msh, x%Xh)
         call gs_h%op(w, n, GS_OP_ADD)
         call bc_projector%apply(w, n)

         pap = glsc3(w, coef%mult, p, n)
         alpha = rtz1 / pap
         if (alpha .lt. 0.0_rp) this%n_neg_alpha = this%n_neg_alpha + 1

         tmp = 0.0_xp
         do i = 1, n
            x%x(i,1,1,1) = x%x(i,1,1,1) + alpha * p(i)
            r(i) = r(i) - alpha * w(i)
            tmp = tmp + r(i) * r(i) * coef%mult(i,1,1,1)
         end do
         call MPI_Allreduce(MPI_IN_PLACE, tmp, 1, MPI_EXTRA_PRECISION, &
              MPI_SUM, NEKO_COMM, ierr)
         rtr = tmp

         k = iter
         this%alpha(iter) = alpha
         this%beta(iter) = beta
         if (iter .le. nstore) then
            m = iter
            do i = 1, n
               this%v(i, iter) = z(i) / sqrt(rtz1)
            end do
         end if

         rnorm = sqrt(rtr) * norm_fac
         call this%monitor_iter(iter, rnorm)
         if (rnorm .lt. this%abs_tol) exit
      end do
    end associate
    call this%monitor_stop()
    ksp_results%res_final = rnorm
    ksp_results%iter = iter
    ksp_results%converged = this%is_converged(iter, rnorm)
    this%n_solve = this%n_solve + 1

    call lanczos_analysis(this, k, m, Ax, x, n, coef, bc_projector, gs_h)
  end function rs_cg_lanczos_solve

  !> Build the Lanczos tridiagonal T_k from the CG coefficients.
  !! d_j = 1/alpha_j + beta_j/alpha_{j-1} (beta_1 = 0),
  !! e_j = -sqrt(beta_{j+1}) / alpha_j (the sign matters for the vectors).
  subroutine build_tridiag(alpha, beta, k, d, e)
    integer, intent(in) :: k
    real(kind=rp), intent(in) :: alpha(:), beta(:)
    real(kind=dp), intent(out) :: d(k), e(*)
    integer :: j

    do j = 1, k
       d(j) = 1.0_dp / alpha(j)
       if (j .gt. 1) d(j) = d(j) + beta(j) / alpha(j-1)
    end do
    do j = 1, k - 1
       e(j) = -sqrt(max(beta(j+1), 0.0_rp)) / alpha(j)
    end do
  end subroutine build_tridiag

  !> Per-element variation of y along the three reference directions.
  subroutine directional_variation(y, lx, ly, lz, nelv, var)
    integer, intent(in) :: lx, ly, lz, nelv
    real(kind=rp), intent(in) :: y(lx, ly, lz, nelv)
    real(kind=dp), intent(out) :: var(3)
    integer :: i, j, l, e, ierr

    var = 0.0_dp
    do e = 1, nelv
       do l = 1, lz
          do j = 1, ly
             do i = 1, lx - 1
                var(1) = var(1) + (y(i+1,j,l,e) - y(i,j,l,e))**2
             end do
          end do
       end do
       do l = 1, lz
          do j = 1, ly - 1
             do i = 1, lx
                var(2) = var(2) + (y(i,j+1,l,e) - y(i,j,l,e))**2
             end do
          end do
       end do
       do l = 1, lz - 1
          do j = 1, ly
             do i = 1, lx
                var(3) = var(3) + (y(i,j,l+1,e) - y(i,j,l,e))**2
             end do
          end do
       end do
    end do
    call MPI_Allreduce(MPI_IN_PLACE, var, 3, MPI_DOUBLE_PRECISION, MPI_SUM, &
         NEKO_COMM, ierr)
  end subroutine directional_variation

  !> Ritz values of T_k, Ritz vectors of T_m (m stored Lanczos vectors).
  subroutine lanczos_analysis(this, k, m, Ax, x, n, coef, bc_projector, gs_h)
    class(rs_cg_lanczos_t), intent(inout), target :: this
    integer, intent(in) :: k, m, n
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    type(coef_t), intent(inout) :: coef
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    real(kind=dp), allocatable :: d(:), e(:), work(:), s(:,:), dm(:), em(:)
    real(kind=dp) :: zdum(1,1), theta, var(3), vsum, kappa
    real(kind=rp), allocatable :: y(:)
    real(kind=rp) :: num, den, yay, wtz, resm, e1, e3
    real(kind=rp), allocatable :: yc(:), yt(:)
    type(space_t) :: Xh1, Xh3
    type(interpolator_t) :: ip1, ip3
    type(field_t) :: fld
    type(file_t) :: out
    character(len=LOG_SIZE) :: log_buf
    integer :: info, i, j, l, nr, unit, nsmall

    if (k .lt. 1) return

    ! Ritz values of the full T_k
    allocate(d(k), e(max(1, k-1)), work(max(1, 2*k-2)))
    call build_tridiag(this%alpha, this%beta, k, d, e)
    call dstev('N', k, d, e, zdum, 1, work, info)
    if (info .ne. 0) then
       write(log_buf, '(A,I0)') 'rs_cg_lanczos: dstev failed, info = ', info
       call neko_log%message(log_buf)
       return
    end if
    kappa = d(k) / d(1)
    nsmall = count(d .lt. 1.0e-2_dp * d(k))

    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,I0,A,ES14.6,A,ES14.6,A,ES14.6,A,I0,A,I0,A,I0,A)') &
            'LANCZOS {"k": ', k, ', "m": ', m, ', "theta_min": ', d(1), &
            ', "theta_max": ', d(k), ', "kappa": ', kappa, &
            ', "n_below_1e-2": ', nsmall, ', "n_neg_rtz": ', this%n_neg_rtz, &
            ', "n_neg_alpha": ', this%n_neg_alpha, '}'
       if (d(1) .lt. 0.0_rp .or. this%n_neg_rtz .gt. 0) then
          write(*, '(A)') 'LANCZOS_WARNING: M^-1 is indefinite (negative Ritz value or r^T z < 0); Ritz vectors are not meaningful'
       else if (d(k) .gt. 1.2_dp) then
          write(*, '(A,ES12.4)') 'LANCZOS_WARNING: largest Ritz value > 1.2, Chebyshev overshoot? ', d(k)
       end if
       open(newunit = unit, file = trim(this%prefix) // '_ritz.txt', &
            status = 'replace', action = 'write')
       write(unit, '(A,I0,A,I0,A,I0)') '# rs_cg_lanczos solve ', &
            this%n_solve, ': k = ', k, ', stored vectors m = ', m
       write(unit, '(A,ES16.8,A,ES16.8,A,ES16.8)') '# theta_min = ', d(1), &
            ', theta_max = ', d(k), ', kappa = ', kappa
       write(unit, '(A)') '# Ritz values of T_k, ascending'
       do j = 1, k
          write(unit, '(I6,ES20.10)') j, d(j)
       end do
       write(unit, '(A)') '# iteration, alpha, beta'
       do j = 1, k
          write(unit, '(I6,2ES20.10)') j, this%alpha(j), this%beta(j)
       end do
       close(unit)
    end if

    ! Slowest Ritz vectors from T_m
    nr = min(this%n_ritz, m)
    if (nr .lt. 1) return
    allocate(dm(m), em(max(1, m-1)), s(m, m), y(n))
    deallocate(work)
    allocate(work(max(1, 2*m-2)))
    call build_tridiag(this%alpha, this%beta, m, dm, em)
    call dstev('V', m, dm, em, s, m, work, info)
    if (info .ne. 0) then
       write(log_buf, '(A,I0)') 'rs_cg_lanczos: dstev(V) failed, info = ', info
       call neko_log%message(log_buf)
       return
    end if

    if (this%write_fld) then
       call fld%init(x%dof, 'ritz')
       call out%init(trim(this%prefix) // '.fld')
    end if

    ! Interpolants at N = 1 and N = 3 for the inter/intra-element split:
    ! e_N = (y - I_N y)^T A (y - I_N y) / y^T A y, where I_N = pointwise
    ! restriction to the N-space and interpolation back (continuous for a
    ! continuous y, so no gather-scatter is needed).
    call Xh1%init(GLL, 2, 2, 2)
    call Xh3%init(GLL, 4, 4, 4)
    call ip1%init(x%Xh, Xh1)
    call ip3%init(x%Xh, Xh3)
    allocate(yc(64 * x%msh%nelv), yt(n))

    associate(w => this%w, z => this%z)
      do j = 1, nr
         theta = dm(j)
         do i = 1, n
            y(i) = 0.0_rp
         end do
         do l = 1, m
            do i = 1, n
               y(i) = y(i) + s(l, j) * this%v(i, l)
            end do
         end do
         ! residual of the Ritz pair in the M^{-1} metric: q = M^{-1} A y
         call Ax%compute(w, y, coef, x%msh, x%Xh)
         call gs_h%op(w, n, GS_OP_ADD)
         call bc_projector%apply(w, n)
         yay = glsc3(y, coef%mult, w, n)
         call this%M%solve(z, w, n)
         ! M-norm residual of the pair: ||M^-1 A y - theta y||_M / theta
         ! = sqrt(w^T z - 2 theta y^T A y + theta^2 y^T M y) / theta with
         ! y^T M y = 1 and y^T A y = theta for a Ritz pair.
         wtz = glsc3(w, coef%mult, z, n)
         resm = sqrt(max(wtz - 2.0_rp * theta * yay + theta**2, 0.0_rp)) / theta
         ! energy outside the N = 1 interpolant
         ! pointwise restriction to the N = 1 nodes (map() would apply the
         ! transpose of the prolongation instead)
         call tnsr3d(yc, 2, y, x%Xh%lx, ip1%Xh_to_Yh, ip1%Xh_to_YhT, &
              ip1%Xh_to_YhT, x%msh%nelv)
         call ip1%map(yt, yc, x%msh%nelv, x%Xh)
         do i = 1, n
            yt(i) = y(i) - yt(i)
         end do
         call Ax%compute(z, yt, coef, x%msh, x%Xh)
         call gs_h%op(z, n, GS_OP_ADD)
         call bc_projector%apply(z, n)
         e1 = glsc3(yt, coef%mult, z, n) / yay
         ! energy outside the N = 3 interpolant
         call tnsr3d(yc, 4, y, x%Xh%lx, ip3%Xh_to_Yh, ip3%Xh_to_YhT, &
              ip3%Xh_to_YhT, x%msh%nelv)
         call ip3%map(yt, yc, x%msh%nelv, x%Xh)
         do i = 1, n
            yt(i) = y(i) - yt(i)
         end do
         call Ax%compute(z, yt, coef, x%msh, x%Xh)
         call gs_h%op(z, n, GS_OP_ADD)
         call bc_projector%apply(z, n)
         e3 = glsc3(yt, coef%mult, z, n) / yay
         ! restore z = M^-1 A y for the Euclidean residual below
         call this%M%solve(z, w, n)
         do i = 1, n
            z(i) = z(i) - theta * y(i)
         end do
         num = glsc3(z, coef%mult, z, n)
         den = glsc3(y, coef%mult, y, n) * theta**2
         call directional_variation(y, x%Xh%lx, x%Xh%ly, x%Xh%lz, &
              x%msh%nelv, var)
         vsum = max(sum(var), tiny(1.0_dp))
         if (pe_rank .eq. 0) then
            write(*, '(A,I0,A,ES14.6,A,ES10.3,A,ES10.3,A,ES10.3,A,ES10.3,A,ES12.5,A,F7.4,A,F7.4,A,F7.4,A)') &
                 'RITZ {"j": ', j, ', "theta": ', theta, &
                 ', "relres_M": ', resm, &
                 ', "relres": ', sqrt(num / den), &
                 ', "e_out_N1": ', e1, ', "e_out_N3": ', e3, &
                 ', "yAy_over_theta": ', yay / theta, &
                 ', "var_r": ', var(1) / vsum, ', "var_s": ', var(2) / vsum, &
                 ', "var_t": ', var(3) / vsum, '}'
         end if
         if (this%write_fld) then
            do i = 1, n
               fld%x(i,1,1,1) = y(i)
            end do
            call out%write(fld, real(j, dp))
         end if
      end do
    end associate

    if (this%write_fld) then
       call out%free()
       call fld%free()
    end if
    call ip1%free()
    call ip3%free()
    call Xh1%free()
    call Xh3%free()
    deallocate(yc, yt)
  end subroutine lanczos_analysis

  function rs_cg_lanczos_solve_coupled(this, Ax, x, y, z, fx, fy, fz, &
       n, coef, bc_projector, gs_h, niter) result(ksp_results)
    class(rs_cg_lanczos_t), intent(inout) :: this
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
    type(ksp_monitor_t), dimension(3) :: ksp_results
    integer, optional, intent(in) :: niter
    type(scalar_bc_projector_t), pointer :: bc_x, bc_y, bc_z

    call vector_bc_projector_components(bc_projector, bc_x, bc_y, bc_z)
    ksp_results(1) = this%solve(Ax, x, fx, n, coef, bc_x, gs_h, niter)
    ksp_results(2) = this%solve(Ax, y, fy, n, coef, bc_y, gs_h, niter)
    ksp_results(3) = this%solve(Ax, z, fz, n, coef, bc_z, gs_h, niter)
  end function rs_cg_lanczos_solve_coupled

  subroutine rs_cg_lanczos_register_types()
    procedure(krylov_allocate), pointer :: allocator

    allocator => rs_cg_lanczos_allocate
    call register_krylov('rs_cg_lanczos', allocator)
  end subroutine rs_cg_lanczos_register_types

  subroutine rs_cg_lanczos_allocate(obj)
    class(ksp_t), allocatable, intent(inout) :: obj

    allocate(rs_cg_lanczos_t :: obj)
  end subroutine rs_cg_lanczos_allocate

end module rs_cg_lanczos
