! CPU builds: makeneko extruded.f90 ext_spchol.f90 ext_dev_stub.f90 (the host path
! is used; these entry points of ext_dev.hip are never called on the CPU backend).
function ext_dev_setup() bind(c, name = 'ext_dev_setup') result(ctx)
  use, intrinsic :: iso_c_binding
  type(c_ptr) :: ctx
  ctx = c_null_ptr
  stop 'extruded: ext_dev_setup called in a CPU build'
end function ext_dev_setup

subroutine ext_dev_factor() bind(c, name = 'ext_dev_factor')
  stop 'extruded: ext_dev_factor called in a CPU build'
end subroutine ext_dev_factor

function ext_dev_ptr() bind(c, name = 'ext_dev_ptr') result(p)
  use, intrinsic :: iso_c_binding
  type(c_ptr) :: p
  p = c_null_ptr
  stop 'extruded: ext_dev_ptr called in a CPU build'
end function ext_dev_ptr

subroutine ext_dev_gather() bind(c, name = 'ext_dev_gather')
  stop 'extruded: ext_dev_gather called in a CPU build'
end subroutine ext_dev_gather

subroutine ext_dev_scatter() bind(c, name = 'ext_dev_scatter')
  stop 'extruded: ext_dev_scatter called in a CPU build'
end subroutine ext_dev_scatter

subroutine ext_dev_forward() bind(c, name = 'ext_dev_forward')
  stop 'extruded: ext_dev_forward called in a CPU build'
end subroutine ext_dev_forward

subroutine ext_dev_solve() bind(c, name = 'ext_dev_solve')
  stop 'extruded: ext_dev_solve called in a CPU build'
end subroutine ext_dev_solve

subroutine ext_dev_backward() bind(c, name = 'ext_dev_backward')
  stop 'extruded: ext_dev_backward called in a CPU build'
end subroutine ext_dev_backward
