! Minimal stand-ins for the Neko modules q2d_plane uses (standalone tests only).
module num_types
  integer, parameter :: dp = kind(1.0d0), i8 = selected_int_kind(18)
end module num_types

module comm
  use mpi_f08, only : MPI_Comm
  type(MPI_Comm) :: NEKO_COMM
  integer :: pe_rank = 0, pe_size = 1
end module comm

module utils
contains
  subroutine neko_error(msg)
    use mpi_f08, only : MPI_Abort, MPI_COMM_WORLD
    character(len=*), intent(in) :: msg
    print '(A)', 'ERROR: ' // msg
    call MPI_Abort(MPI_COMM_WORLD, 1)
  end subroutine neko_error
end module utils
