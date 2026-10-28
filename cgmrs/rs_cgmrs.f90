!> Preconditioned CG with minimal residual smoothing (`ksp_t` user type
!! 'rs_cgmrs').
!!
!! CG minimises the error in the A-norm; its residual 2-norm is not
!! monotone. Neko stops on the residual norm, and in a flow the pressure
!! solves often start only a few times above the absolute tolerance, so the
!! oscillating CG residual decides the iteration count (foil, logbook
!! 2026-10-12 19:09: 10-44 CG iterations where GMRES needs 1-5).
!! Minimal residual smoothing (Zhou & Walker, SIAM J. Sci. Comput. 15, 1994)
!! carries a second iterate y with residual s alongside the CG pair (x, r):
!!   d = r - s,  eta = -(s, d) / (d, d),
!!   s <- s + eta d,  y <- (1 - eta) y + eta x,
!! which makes ||s|| non-increasing (s is the minimum-norm residual on the
!! line through s and r). The solve stops on ||s|| and returns y. CG itself
!! is unchanged; the cost is two vectors, three dot products and three vector
!! updates per iteration (||s|| is computed from s, not by a recurrence). Inner products and norms as in Neko's CG
!! (mult-weighted, scaled by 1/sqrt(volume)). Host and device backends,
!! any working precision (Neko's math routines only).
module rs_cgmrs
  use num_types, only : rp
  use krylov, only : ksp_t, ksp_monitor_t, krylov_allocate, register_krylov
  use precon, only : pc_t
  use ax_product, only : ax_t
  use field, only : field_t
  use coefs, only : coef_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use scalar_bc_projector, only : scalar_bc_projector_t
  use vector_bc_projector, only : vector_bc_projector_t, &
       vector_bc_projector_components
  use math, only : glsc3, abscmp, rzero, copy, add2s1, add2s2, cmult, sub3
  use device_math, only : device_glsc3, device_rzero, device_copy, &
       device_add2s1, device_add2s2, device_cmult, device_sub3
  use device, only : device_map, device_free, device_deassociate, device_get_ptr, &
       device_event_create, device_event_destroy, device_event_sync
  use neko_config, only : NEKO_BCKND_DEVICE
  use comm, only : pe_rank
  use, intrinsic :: iso_c_binding, only : c_ptr, c_null_ptr, c_associated
  implicit none
  private

  type, public, extends(ksp_t) :: rs_cgmrs_t
     real(kind=rp), allocatable :: w(:), r(:), p(:), z(:), y(:), s(:)
     type(c_ptr) :: w_d = c_null_ptr, r_d = c_null_ptr, p_d = c_null_ptr
     type(c_ptr) :: z_d = c_null_ptr, y_d = c_null_ptr, s_d = c_null_ptr
     type(c_ptr) :: gs_event = c_null_ptr
   contains
     procedure, pass(this) :: init => rs_cgmrs_init
     procedure, pass(this) :: free => rs_cgmrs_free
     procedure, pass(this) :: solve => rs_cgmrs_solve
     procedure, pass(this) :: solve_coupled => rs_cgmrs_solve_coupled
  end type rs_cgmrs_t

  public :: rs_cgmrs_register_types

contains

  subroutine rs_cgmrs_init(this, n, max_iter, M, rel_tol, abs_tol, monitor)
    class(rs_cgmrs_t), intent(inout), target :: this
    integer, intent(in) :: n
    integer, intent(in) :: max_iter
    class(pc_t), optional, intent(in), target :: M
    real(kind=rp), optional, intent(in) :: rel_tol
    real(kind=rp), optional, intent(in) :: abs_tol
    logical, optional, intent(in) :: monitor

    call this%free()
    allocate(this%w(n), this%r(n), this%p(n), this%z(n), this%y(n), this%s(n))
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call device_map(this%w, this%w_d, n)
       call device_map(this%r, this%r_d, n)
       call device_map(this%p, this%p_d, n)
       call device_map(this%z, this%z_d, n)
       call device_map(this%y, this%y_d, n)
       call device_map(this%s, this%s_d, n)
       call device_event_create(this%gs_event, 2)
    end if
    if (present(M)) this%M => M

    if (present(rel_tol) .and. present(abs_tol) .and. present(monitor)) then
       call this%ksp_init(max_iter, rel_tol, abs_tol, monitor = monitor)
    else if (present(rel_tol) .and. present(abs_tol)) then
       call this%ksp_init(max_iter, rel_tol, abs_tol)
    else if (present(monitor) .and. present(abs_tol)) then
       call this%ksp_init(max_iter, abs_tol = abs_tol, monitor = monitor)
    else if (present(rel_tol) .and. present(monitor)) then
       call this%ksp_init(max_iter, rel_tol, monitor = monitor)
    else if (present(rel_tol)) then
       call this%ksp_init(max_iter, rel_tol = rel_tol)
    else if (present(abs_tol)) then
       call this%ksp_init(max_iter, abs_tol = abs_tol)
    else if (present(monitor)) then
       call this%ksp_init(max_iter, monitor = monitor)
    else
       call this%ksp_init(max_iter)
    end if
  end subroutine rs_cgmrs_init

  subroutine rs_cgmrs_free(this)
    class(rs_cgmrs_t), intent(inout) :: this
    call this%ksp_free()
    call unmap(this%w, this%w_d)
    call unmap(this%r, this%r_d)
    call unmap(this%p, this%p_d)
    call unmap(this%z, this%z_d)
    call unmap(this%y, this%y_d)
    call unmap(this%s, this%s_d)
    if (c_associated(this%gs_event)) call device_event_destroy(this%gs_event)
    this%gs_event = c_null_ptr
    nullify(this%M)
  contains
    subroutine unmap(a, a_d)
      real(kind=rp), allocatable, intent(inout) :: a(:)
      type(c_ptr), intent(inout) :: a_d
      if (c_associated(a_d)) then
         call device_deassociate(a)
         call device_free(a_d)
      end if
      a_d = c_null_ptr
      if (allocated(a)) deallocate(a)
    end subroutine unmap
  end subroutine rs_cgmrs_free

  function rs_cgmrs_solve(this, Ax, x, f, n, coef, bc_projector, gs_h, niter) &
       result(ksp_results)
    class(rs_cgmrs_t), intent(inout) :: this
    class(ax_t), intent(in) :: Ax
    type(field_t), intent(inout) :: x
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(in) :: f
    type(coef_t), intent(inout) :: coef
    class(scalar_bc_projector_t), intent(inout) :: bc_projector
    type(gs_t), intent(inout) :: gs_h
    integer, optional, intent(in) :: niter
    type(ksp_monitor_t) :: ksp_results
    real(kind=rp), parameter :: one = 1.0_rp, zero = 0.0_rp
    integer :: iter, max_iter
    real(kind=rp) :: rnorm, rtz1, rtz2, beta, pap, alpha, norm_fac
    real(kind=rp) :: ss, ds, dd, eta
    logical :: dev
    type(c_ptr) :: f_d

    dev = NEKO_BCKND_DEVICE .eq. 1
    max_iter = this%max_iter
    if (present(niter)) max_iter = niter
    norm_fac = one / sqrt(coef%volume)

    rtz1 = one
    if (dev) then
       f_d = device_get_ptr(f)
       call device_rzero(x%x_d, n)
       call device_rzero(this%p_d, n)
       call device_rzero(this%y_d, n)
       call device_copy(this%r_d, f_d, n)
       call device_copy(this%s_d, f_d, n)
       ss = device_glsc3(this%s_d, coef%mult_d, this%s_d, n)
    else
       call rzero(x%x, n)
       call rzero(this%p, n)
       call rzero(this%y, n)
       call copy(this%r, f, n)
       call copy(this%s, f, n)
       ss = glsc3(this%s, coef%mult, this%s, n)
    end if
    rnorm = sqrt(ss) * norm_fac
    ksp_results%res_start = rnorm
    ksp_results%res_final = rnorm
    ksp_results%iter = 0
    if (abscmp(rnorm, zero)) then
       ksp_results%converged = .true.
       return
    end if

    call this%monitor_start('CG-MRS')
    do iter = 1, max_iter
       ! --- CG step (as Neko's cg / cg_device) ---
       call this%M%solve(this%z, this%r, n)
       rtz2 = rtz1
       if (dev) then
          rtz1 = device_glsc3(this%r_d, coef%mult_d, this%z_d, n)
       else
          rtz1 = glsc3(this%r, coef%mult, this%z, n)
       end if
       beta = rtz1 / rtz2
       if (iter .eq. 1) beta = zero
       if (dev) then
          call device_add2s1(this%p_d, this%z_d, beta, n)
       else
          call add2s1(this%p, this%z, beta, n)
       end if

       call Ax%compute(this%w, this%p, coef, x%msh, x%Xh)
       if (dev) then
          call gs_h%op(this%w, n, GS_OP_ADD, this%gs_event)
          call device_event_sync(this%gs_event)
       else
          call gs_h%op(this%w, n, GS_OP_ADD)
       end if
       call bc_projector%apply(this%w, n)

       if (dev) then
          pap = device_glsc3(this%w_d, coef%mult_d, this%p_d, n)
       else
          pap = glsc3(this%w, coef%mult, this%p, n)
       end if
       alpha = rtz1 / pap
       if (dev) then
          call device_add2s2(x%x_d, this%p_d, alpha, n)
          call device_add2s2(this%r_d, this%w_d, -alpha, n)
       else
          call add2s2(x%x, this%p, alpha, n)
          call add2s2(this%r, this%w, -alpha, n)
       end if

       ! --- minimal residual smoothing (w is free now: d = r - s) ---
       if (dev) then
          call device_sub3(this%w_d, this%r_d, this%s_d, n)
          ds = device_glsc3(this%s_d, coef%mult_d, this%w_d, n)
          dd = device_glsc3(this%w_d, coef%mult_d, this%w_d, n)
       else
          call sub3(this%w, this%r, this%s, n)
          ds = glsc3(this%s, coef%mult, this%w, n)
          dd = glsc3(this%w, coef%mult, this%w, n)
       end if
       eta = zero
       if (dd .gt. zero) eta = -ds / dd
       if (dev) then
          call device_add2s2(this%s_d, this%w_d, eta, n)
          call device_cmult(this%y_d, one - eta, n)
          call device_add2s2(this%y_d, x%x_d, eta, n)
       else
          call add2s2(this%s, this%w, eta, n)
          call cmult(this%y, one - eta, n)
          call add2s2(this%y, x%x, eta, n)
       end if
       ! ||s_new||^2 from s itself: the recurrence ||s||^2 - (s, d)^2 / (d, d)
       ! cancels to zero in finite precision and stopped the solve early
       ! (neko_extra_nla, 2026-10-28)
       if (dev) then
          ss = device_glsc3(this%s_d, coef%mult_d, this%s_d, n)
       else
          ss = glsc3(this%s, coef%mult, this%s, n)
       end if
       rnorm = sqrt(ss) * norm_fac
       call this%monitor_iter(iter, rnorm)
       if (rnorm .lt. this%abs_tol) exit
    end do
    call this%monitor_stop()
    if (iter .gt. max_iter) iter = max_iter

    ! the smoothed iterate is the solution
    if (dev) then
       call device_copy(x%x_d, this%y_d, n)
    else
       call copy(x%x, this%y, n)
    end if
    ksp_results%res_final = rnorm
    ksp_results%iter = iter
    ksp_results%converged = this%is_converged(iter, rnorm)

    ! env RS_CGMRS_CHECK: true residual ||f - A y|| of the returned iterate
    block
      character(len=8) :: env
      integer :: istat
      real(kind=rp) :: tr
      call get_environment_variable('RS_CGMRS_CHECK', env, status = istat)
      if (istat .eq. 0) then
         call Ax%compute(this%w, x%x, coef, x%msh, x%Xh)
         if (dev) then
            call gs_h%op(this%w, n, GS_OP_ADD, this%gs_event)
            call device_event_sync(this%gs_event)
         else
            call gs_h%op(this%w, n, GS_OP_ADD)
         end if
         call bc_projector%apply(this%w, n)
         if (dev) then
            call device_sub3(this%r_d, f_d, this%w_d, n)
            tr = device_glsc3(this%r_d, coef%mult_d, this%r_d, n)
         else
            call sub3(this%r, f, this%w, n)
            tr = glsc3(this%r, coef%mult, this%r, n)
         end if
         if (pe_rank .eq. 0) write(*, '(A,I0,A,ES12.4,A,ES12.4)') 'RS_CGMRS check: its ', iter, &
              '  smoothed residual ', rnorm, '  true residual ', sqrt(tr) * norm_fac
      end if
    end block
  end function rs_cgmrs_solve

  function rs_cgmrs_solve_coupled(this, Ax, x, y, z, fx, fy, fz, &
       n, coef, bc_projector, gs_h, niter) result(ksp_results)
    class(rs_cgmrs_t), intent(inout) :: this
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
    type(scalar_bc_projector_t), pointer :: bc_x, bc_y, bc_z

    call vector_bc_projector_components(bc_projector, bc_x, bc_y, bc_z)
    ksp_results(1) = this%solve(Ax, x, fx, n, coef, bc_x, gs_h, niter)
    ksp_results(2) = this%solve(Ax, y, fy, n, coef, bc_y, gs_h, niter)
    ksp_results(3) = this%solve(Ax, z, fz, n, coef, bc_z, gs_h, niter)
  end function rs_cgmrs_solve_coupled

  subroutine rs_cgmrs_register_types()
    procedure(krylov_allocate), pointer :: allocator
    allocator => rs_cgmrs_allocate
    call register_krylov('rs_cgmrs', allocator)
  end subroutine rs_cgmrs_register_types

  subroutine rs_cgmrs_allocate(obj)
    class(ksp_t), allocatable, intent(inout) :: obj
    allocate(rs_cgmrs_t :: obj)
  end subroutine rs_cgmrs_allocate

end module rs_cgmrs
