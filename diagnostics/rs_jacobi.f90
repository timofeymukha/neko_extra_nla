!> Test preconditioner for the user-type injection path of `pc_t`.
!!
!! Wraps Neko's point Jacobi, or acts as the identity, depending on the
!! case-file option `mode` ("jacobi" or "identity"). The setup reads only the
!! arguments of the deferred `init(coef, bclst, json)`, and logs what it
!! received, so that a run shows whether a registered type is initialized,
!! used and freed like the built-in ones. CPU backend only.
module rs_jacobi
  use num_types, only : rp
  use precon, only : pc_t, precon_allocate, register_precon
  use jacobi, only : jacobi_t
  use coefs, only : coef_t
  use bc_list, only : bc_list_t
  use json_module, only : json_file
  use json_utils, only : json_get_or_default
  use logger, only : neko_log, LOG_SIZE
  use utils, only : neko_error
  use neko_config, only : NEKO_BCKND_DEVICE
  implicit none
  private

  type, extends(pc_t) :: rs_jacobi_t
     type(jacobi_t) :: jac
     logical :: identity = .false.
     integer :: n_solve = 0
     integer :: n_update = 0
   contains
     procedure :: init => rs_jacobi_init
     procedure :: solve => rs_jacobi_solve
     procedure :: update => rs_jacobi_update
     procedure :: free => rs_jacobi_free
  end type rs_jacobi_t

  public :: rs_jacobi_register_types

contains

  subroutine rs_jacobi_init(this, coef, bclst, json)
    class(rs_jacobi_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json
    character(len=:), allocatable :: mode
    character(len=LOG_SIZE) :: log_buf

    if (NEKO_BCKND_DEVICE .eq. 1) then
       call neko_error('rs_jacobi: CPU backend only')
    end if

    call this%free()
    call json_get_or_default(json, 'mode', mode, 'jacobi')
    select case (trim(mode))
    case ('jacobi')
       this%identity = .false.
    case ('identity')
       this%identity = .true.
    case default
       call neko_error('rs_jacobi: unknown mode ' // trim(mode))
    end select

    call this%jac%init(coef, bclst, json)

    call neko_log%section('rs_jacobi')
    write(log_buf, '(A,A)') 'mode       : ', trim(mode)
    call neko_log%message(log_buf)
    write(log_buf, '(A,I0,A,I0)') 'lx, nelv   : ', coef%Xh%lx, ', ', &
         coef%msh%nelv
    call neko_log%message(log_buf)
    write(log_buf, '(A,I0)') 'bcs in list: ', bclst%size()
    call neko_log%message(log_buf)
    write(log_buf, '(A,ES12.5)') 'volume     : ', coef%volume
    call neko_log%message(log_buf)
    call neko_log%end_section()
  end subroutine rs_jacobi_init

  subroutine rs_jacobi_solve(this, z, r, n)
    class(rs_jacobi_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r

    this%n_solve = this%n_solve + 1
    if (this%identity) then
       z = r
    else
       call this%jac%solve(z, r, n)
    end if
  end subroutine rs_jacobi_solve

  subroutine rs_jacobi_update(this)
    class(rs_jacobi_t), intent(inout) :: this

    this%n_update = this%n_update + 1
    call this%jac%update()
  end subroutine rs_jacobi_update

  subroutine rs_jacobi_free(this)
    class(rs_jacobi_t), intent(inout) :: this
    character(len=LOG_SIZE) :: log_buf

    if (this%n_solve .gt. 0 .or. this%n_update .gt. 0) then
       write(log_buf, '(A,I0,A,I0,A)') 'rs_jacobi: freed after ', &
            this%n_solve, ' solves and ', this%n_update, ' updates'
       call neko_log%message(log_buf)
    end if
    call this%jac%free()
    this%n_solve = 0
    this%n_update = 0
  end subroutine rs_jacobi_free

  subroutine rs_jacobi_register_types()
    procedure(precon_allocate), pointer :: allocator

    allocator => rs_jacobi_allocate
    call register_precon('rs_jacobi', allocator)
  end subroutine rs_jacobi_register_types

  subroutine rs_jacobi_allocate(obj)
    class(pc_t), allocatable, intent(inout) :: obj

    allocate(rs_jacobi_t :: obj)
  end subroutine rs_jacobi_allocate

end module rs_jacobi
