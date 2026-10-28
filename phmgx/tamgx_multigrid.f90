! tamgx_multigrid: tree-AMG coarse solver with exact-diagonal Jacobi scaling
! Derived from Neko (develop 77243ce, BSD-3-Clause, the Neko authors) with the
! user's patch neko_phmg_exact_diag_develop-6a661f1 (Jacobi-scaled tree-AMG
! Chebyshev with an exact coarse diagonal, deeper hierarchy): the phmg part of
! the patch as a user type (registered as "phmgx"), modules renamed so that it
! lives next to Neko's own phmg. The GMRES-restart part of the patch is left out.
! Source: references/neko_phmg_patch/ (patch, README, phmg_study/REPORT.md).
! Copyright (c) 2024-2026, The Neko Authors
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
!   * Redistributions of source code must retain the above copyright
!     notice, this list of conditions and the following disclaimer.
!
!   * Redistributions in binary form must reproduce the above
!     copyright notice, this list of conditions and the following
!     disclaimer in the documentation and/or other materials provided
!     with the distribution.
!
!   * Neither the name of the authors nor the names of its
!     contributors may be used to endorse or promote products derived
!     from this software without specific prior written permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
!> Implements multigrid using the TreeAMG hierarchy structure.
!> USE:
!>
!>  type(tamg_hierarchy_t) :: amg
!>  type(tamg_solver_t) :: amg_solver
!>
!>  call amg%init(ax, Xh, coef, msh, gs_h, 3)
!>  call amg_solver%init(amg, niter)
!>
!>  call amg_solver%solve(x%x, f, n)
!>
module tamgx_multigrid
  use num_types, only : rp, i8
  use utils, only : neko_error, neko_warning
  use math, only : add2, rzero, glsc2, col2, copy, add2s1, invcol1
  use device_math, only : device_rzero, device_col2, device_add2, device_sub3, &
       device_glsc2, device_copy, device_invcol1, device_cmult, device_col3
  use device_jacobi, only : device_jacobi_t
  use jacobi, only : jacobi_t
  use comm
  use mpi_f08, only : MPI_Allreduce, MPI_MIN, MPI_MAX, MPI_SUM, MPI_IN_PLACE, &
       MPI_INTEGER, MPI_Wtime
  use coefs, only : coef_t
  use mesh, only : mesh_t
  use space, only : space_t
  use ax_product, only : ax_t
  use scalar_bc_projector, only : scalar_bc_projector_t
  use gather_scatter, only : gs_t, GS_OP_ADD, GS_OP_MIN
  use tree_amg, only : tamg_hierarchy_t, tamg_lvl_init, tamg_node_init
  use tree_amg_aggregate, only : aggregate_finest_level, aggregate_greedy, &
       aggregate_end, aggregate_pairs
  use tamgx_smoother, only : amg_cheby_t
  use profiler, only : profiler_start_region, profiler_end_region
  use logger, only : neko_log, LOG_SIZE
  use device, only : device_map, device_unmap, device_memcpy, HOST_TO_DEVICE, &
       DEVICE_TO_HOST, &
       device_get_ptr
  use neko_config, only : NEKO_BCKND_DEVICE
  use, intrinsic :: iso_c_binding
  !$ use omp_lib, only : omp_get_max_threads
  implicit none
  private

  type :: tamg_wrk_t
     integer :: n = -1
     real(kind=rp), allocatable :: r(:)
     real(kind=rp), allocatable :: b(:)
     real(kind=rp), allocatable :: x(:)
     type(c_ptr) :: r_d = C_NULL_PTR
     type(c_ptr) :: b_d = C_NULL_PTR
     type(c_ptr) :: x_d = C_NULL_PTR
  end type tamg_wrk_t

  !> Type for the TreeAMG solver
  type, public :: tamg_solver_t
     type(tamg_hierarchy_t), allocatable :: amg
     type(amg_cheby_t), allocatable :: smoo(:)
     type(tamg_wrk_t), allocatable :: wrk(:)
     !type(amg_jacobi_t), allocatable :: jsmoo(:)
     integer :: nlvls
     integer :: max_iter
   contains
     procedure, pass(this) :: init => tamg_mg_init
     procedure, pass(this) :: solve => tamg_mg_solve
     procedure, pass(this) :: free => tamg_mg_free
     procedure, pass(this) :: invalidate_eigs => tamg_mg_invalidate_eigs
     procedure, pass(this) :: set_eig_refresh => tamg_mg_set_eig_refresh
     procedure, private, pass(this) :: mg_cycle => tamg_mg_cycle
     procedure, private, pass(this) :: mg_cycle_d => tamg_mg_cycle_d
  end type tamg_solver_t

contains

  !> Initialization of the TreeAMG multigrid solver
  !! @param ax Finest level matvec operator
  !! @param Xh Finest level field
  !! @param coef Finest level coeff thing
  !! @param msh Finest level mesh information
  !! @param gs_h Finest level gather scatter operator
  !! @param nlvls Number of levels for the TreeAMG hierarchy
  !! @param bc_projector Finest level BC projector
  !! @param max_iter Number of AMG iterations
  subroutine tamg_mg_init(this, ax, Xh, coef, msh, gs_h, nlvls, bc_projector, &
       max_iter, cheby_degree, use_jacobi, jacobi_probes)
    class(tamg_solver_t), intent(inout), target :: this
    class(ax_t), target, intent(in) :: ax
    type(space_t), target, intent(in) :: Xh
    type(coef_t), target, intent(in) :: coef
    type(mesh_t), target, intent(in) :: msh
    type(gs_t), target, intent(in) :: gs_h
    type(scalar_bc_projector_t), target, intent(in) :: bc_projector
    integer, intent(in) :: nlvls
    integer, intent(in) :: max_iter
    integer, intent(in) :: cheby_degree
    logical, intent(in), optional :: use_jacobi
    integer, intent(in), optional :: jacobi_probes
    integer :: lvl, n, mlvl, target_num_aggs, nprobe
    integer, allocatable :: agg_nhbr(:,:), nhbr_tmp(:,:)
    character(len=LOG_SIZE) :: log_buf
    integer :: glb_min_target_aggs
    logical :: use_greedy_agg

    call neko_log%section('AMG')

    write(log_buf, '(A28,I2,A8)') 'Creating AMG hierarchy with', &
         nlvls, 'levels.'
    call neko_log%message(log_buf)

    allocate( this%amg )
    call this%amg%init(ax, Xh, coef, msh, gs_h, nlvls, bc_projector)

    ! Aggregation
    use_greedy_agg = .true.
    ! Create level 1 (neko elements are level 0)
    call aggregate_finest_level(this%amg, Xh%lx, Xh%ly, Xh%lz, msh%nelv)

    ! Create the remaining levels
    allocate( agg_nhbr, SOURCE = msh%facet_neigh )
    do mlvl = 2, nlvls-1
       ! estimate number of aggregates
       if (use_greedy_agg) then
          target_num_aggs = this%amg%lvl(mlvl-1)%nnodes / 8
       else
          target_num_aggs = this%amg%lvl(mlvl-1)%nnodes / 2
       end if

       glb_min_target_aggs = target_num_aggs
       call MPI_Allreduce(MPI_IN_PLACE, glb_min_target_aggs, 1, &
            MPI_INTEGER, MPI_MIN, NEKO_COMM)
       if (glb_min_target_aggs .lt. 4 ) then
          call neko_warning( &
               "TAMG: Too many levels. Not enough DOFs for coarsest grid.")
          this%amg%nlvls = mlvl
          exit
       end if

       if (use_greedy_agg) then
          call print_preagg_info( mlvl, glb_min_target_aggs, 1)
          call aggregate_greedy(this%amg, mlvl, target_num_aggs, &
               agg_nhbr, nhbr_tmp)
       else
          call print_preagg_info( mlvl, glb_min_target_aggs, 2)
          call aggregate_pairs(this%amg, mlvl, target_num_aggs, &
               agg_nhbr, nhbr_tmp)
       end if

       agg_nhbr = nhbr_tmp
       deallocate( nhbr_tmp )
    end do
    deallocate( agg_nhbr )

    ! Create the end point
    call aggregate_end(this%amg, this%amg%nlvls)

    this%max_iter = max_iter

    this%nlvls = this%amg%nlvls
    if (this%nlvls .gt. this%amg%nlvls) then
       call neko_error( &
            "Requested number multigrid levels &
       & is greater than the initialized AMG levels")
    end if

    ! Initialize relaxation methods
    allocate(this%smoo(0:(this%amg%nlvls)))
    do lvl = 0, this%amg%nlvls-1
       n = this%amg%lvl(lvl+1)%fine_lvl_dofs
       call this%smoo(lvl)%init(n, lvl, cheby_degree)
    end do

    ! Allocate work space on each level
    allocate(this%wrk(0:(this%amg%nlvls)))
    do lvl = 0, this%amg%nlvls-1
       n = this%amg%lvl(lvl+1)%fine_lvl_dofs
       this%wrk(lvl)%n = n
       allocate( this%wrk(lvl)%r(n) )
       allocate( this%wrk(lvl)%b(n) )
       allocate( this%wrk(lvl)%x(n) )
       if (NEKO_BCKND_DEVICE .eq. 1) then
          call device_map( this%wrk(lvl)%r, this%wrk(lvl)%r_d, n)
          call device_map( this%wrk(lvl)%b, this%wrk(lvl)%b_d, n)
          call device_map( this%wrk(lvl)%x, this%wrk(lvl)%x_d, n)
       end if
    end do

    !allocate(this%jsmoo(0:(this%amg%nlvls)))
    !do lvl = 0, this%amg%nlvls-1
    !  n = this%amg%lvl(lvl+1)%fine_lvl_dofs
    !  call this%jsmoo(lvl)%init(n ,lvl, cheby_degree)
    !end do

    ! Create index mapping between levels
    call fill_lvl_map(this%amg)

    ! Invert those maps so the flat matvec can be driven from the aggregate
    ! side instead of scattering into the coarse vector with atomics
    call build_agg_csr(this%amg)

    nprobe = 48
    if (present(jacobi_probes)) nprobe = jacobi_probes
    if (present(use_jacobi)) then
       if (use_jacobi) call tamg_setup_jacobi(this, nprobe)
    end if

    call neko_log%end_section()

  end subroutine tamg_mg_init

  !> Build the diagonal scaling for Jacobi-accelerated Chebyshev smoothing:
  !! the assembled diagonal of the finest (lx = 2) operator, summed over
  !! the aggregates for the coarser levels. The exact coarse diagonal
  !! P^T A P would need one matvec per aggregate colour; the lumped sum
  !! has the same spatial variation, which is all the smoother needs.
  subroutine tamg_setup_jacobi(this, nprobe)
    class(tamg_solver_t), intent(inout), target :: this
    integer, intent(in) :: nprobe
    type(device_jacobi_t) :: djac
    type(jacobi_t) :: jac
    real(kind=rp), allocatable :: dg(:), dgc(:), pv(:), pw(:), pacc(:)
    type(c_ptr) :: dg_d, dgc_d, pv_d, pw_d, pacc_d
    integer :: lvl, n, m, k, i, nclamp
    real(kind=rp) :: rn
    character(len=LOG_SIZE) :: log_buf

    n = this%wrk(0)%n
    allocate(dg(n))
    dg_d = C_NULL_PTR
    dgc_d = C_NULL_PTR
    if (NEKO_BCKND_DEVICE .eq. 1) then
       call device_map(dg, dg_d, n)
       call djac%init_from_components(this%amg%coef, this%amg%coef%dof, this%amg%gs_h)
       call device_copy(dg_d, djac%d_d, n)
       call device_invcol1(dg_d, n)
       call this%smoo(0)%set_jacobi_d(dg_d, n)
       call djac%free()
    else
       call jac%init_from_components(this%amg%coef, this%amg%coef%dof, this%amg%gs_h)
       call copy(dg, jac%d, n)
       call invcol1(dg, n)
       call this%smoo(0)%set_jacobi(dg, n)
       call jac%free()
    end if
    do lvl = 1, this%nlvls - 1
       m = this%wrk(lvl)%n
       allocate(dgc(m))
       if (NEKO_BCKND_DEVICE .eq. 1) then
          call device_map(dgc, dgc_d, m)
          call this%amg%interp_f2c_d(dgc_d, dg_d, lvl)
          if (nprobe .lt. 0) then
             call device_memcpy(dgc, dgc_d, m, DEVICE_TO_HOST, .true.)
             call tamg_coarse_diag_exact(this, lvl, dgc, m)
             call device_memcpy(dgc, dgc_d, m, HOST_TO_DEVICE, .true.)
          else if (nprobe .gt. 0) then
             ! Hutchinson estimate of the true coarse diagonal:
             ! diag(A_l) ~ mean_k v_k o (A_l v_k), v_k random +-1.
             ! The lumped sum only serves as a fallback for
             ! non-positive estimates (e.g. Dirichlet-masked rows).
             allocate(pv(m), pw(m), pacc(m))
             pv_d = C_NULL_PTR; pw_d = C_NULL_PTR; pacc_d = C_NULL_PTR
             call device_map(pv, pv_d, m)
             call device_map(pw, pw_d, m)
             call device_map(pacc, pacc_d, m)
             call device_rzero(pacc_d, m)
             do k = 1, nprobe
                do i = 1, m
                   call random_number(rn)
                   pv(i) = sign(1.0_rp, rn - 0.5_rp)
                end do
                call device_memcpy(pv, pv_d, m, HOST_TO_DEVICE, .true.)
                call this%amg%device_matvec(pw, pv, pw_d, pv_d, lvl)
                call device_col2(pw_d, pv_d, m)
                call device_add2(pacc_d, pw_d, m)
             end do
             call device_cmult(pacc_d, 1.0_rp / real(nprobe, rp), m)
             call device_memcpy(pacc, pacc_d, m, DEVICE_TO_HOST, .true.)
             call device_memcpy(dgc, dgc_d, m, DEVICE_TO_HOST, .true.)
             ! The lumped sum bounds the true diagonal from above (the
             ! couplings inside an aggregate are negative); the lower clamp
             ! keeps a noisy small estimate from blowing up the Chebyshev
             ! interval for every other row on the level.
             nclamp = 0
             do i = 1, m
                if (pacc(i) .gt. dgc(i)) then
                   nclamp = nclamp + 1
                else if (pacc(i) .gt. 0.05_rp * dgc(i)) then
                   dgc(i) = pacc(i)
                else
                   dgc(i) = 0.05_rp * dgc(i)
                   nclamp = nclamp + 1
                end if
             end do
             call device_memcpy(dgc, dgc_d, m, HOST_TO_DEVICE, .true.)
             write(log_buf, '(A,I2,A,I4,A,I8)') '-- AMG level', lvl, &
                  ' -- probed diagonal (', nprobe, ' probes), clamped rows:', nclamp
             call neko_log%message(log_buf)
             call device_unmap(pv, pv_d)
             call device_unmap(pw, pw_d)
             call device_unmap(pacc, pacc_d)
             deallocate(pv, pw, pacc)
          end if
          call this%smoo(lvl)%set_jacobi_d(dgc_d, m)
          call device_unmap(dg, dg_d)
       else
          call this%amg%interp_f2c(dgc, dg, lvl)
          if (nprobe .gt. 0) then
             allocate(pv(m), pw(m), pacc(m))
             pacc = 0.0_rp
             do k = 1, nprobe
                do i = 1, m
                   call random_number(rn)
                   pv(i) = sign(1.0_rp, rn - 0.5_rp)
                end do
                pw = 0.0_rp
                call this%amg%matvec(pw, pv, lvl)
                pacc = pacc + pv * pw
             end do
             pacc = pacc / real(nprobe, rp)
             do i = 1, m
                if (pacc(i) .le. dgc(i) .and. pacc(i) .gt. 0.05_rp * dgc(i)) then
                   dgc(i) = pacc(i)
                else if (pacc(i) .le. 0.05_rp * dgc(i)) then
                   dgc(i) = 0.05_rp * dgc(i)
                end if
             end do
             deallocate(pv, pw, pacc)
          end if
          call this%smoo(lvl)%set_jacobi(dgc, m)
       end if
       deallocate(dg)
       call move_alloc(dgc, dg)
       dg_d = dgc_d
       dgc_d = C_NULL_PTR
    end do
    if (NEKO_BCKND_DEVICE .eq. 1) call device_unmap(dg, dg_d)
    deallocate(dg)
    write(log_buf, '(A)') 'Chebyshev smoothers: Jacobi (diagonal) scaled'
    call neko_log%message(log_buf)
  end subroutine tamg_setup_jacobi

  !> Exact diagonal of the level operator A_l that device_matvec applies
  !! (A_l = R M gs A gs M P), by graph colouring of the aggregates.
  !! Two aggregates are coupled only if an element of one lies within
  !! element-vertex-adjacency distance 2 of an element of the other (gs is
  !! applied to both the input and the output of the finest matvec).
  !! The adjacency is built from local elements plus the ids of ALL
  !! aggregates (any rank) touching each local vertex, discovered with a few
  !! gather-scatter MIN rounds, so couplings through other ranks' elements
  !! are excluded exactly. Aggregates coupled across ranks are coloured from
  !! rank-disjoint colour blocks. One collective matvec per colour then
  !! returns the exact diagonal entries of that colour's aggregates.
  !! Non-positive entries (masked rows) keep the lumped value passed in.
  !! @param lvl AMG level (>= 1)
  !! @param dg On input the lumped diagonal (fallback), on output the exact one
  !! @param m Number of aggregates on the level
  subroutine tamg_coarse_diag_exact(this, lvl, dg, m)
    class(tamg_solver_t), intent(inout), target :: this
    integer, intent(in) :: lvl, m
    real(kind=rp), intent(inout) :: dg(m)
    integer, parameter :: NR = 32
    integer :: nelv, lx, ndofe, e, i, j, k, v, vmax, ia, ib, t, c, n1, n
    integer :: jj, kk, ll, nclr, nbad, ntest, mxc, stride, ng, nbnd, nmv
    integer :: mmax, g, r, round, nleft, nrem, nrmax, rbase
    integer, allocatable :: agg(:), vcnt(:), vptr(:), vlist(:)
    integer, allocatable :: acnt(:), aptr(:), alist(:), colour(:)
    integer, allocatable :: mark(:), list1(:), used(:), gcol(:), gcnt(:)
    integer, allocatable :: nid(:), vid(:,:)
    integer, allocatable :: rcnt(:), rptr(:), rlist(:), rstamp(:)
    logical, allocatable :: bnd(:), bnd_prev(:), masked(:)
    real(kind=rp), allocatable :: pv(:), pw(:), dex(:), xk(:), xm(:)
    type(c_ptr) :: pv_d, pw_d, xm_d
    real(kind=rp) :: errmax, errbnd, t0, t1, big
    logical :: remote_ok
    character(len=LOG_SIZE) :: log_buf

    if (NEKO_BCKND_DEVICE .ne. 1) then
       call neko_warning("TAMG exact coarse diagonal: device only, " // &
            "keeping the lumped diagonal")
       return
    end if

    t0 = MPI_Wtime()
    associate(amg => this%amg)
      nelv = amg%msh%nelv
      lx = amg%Xh%lx
      ndofe = lx * lx * lx
      n = amg%lvl(1)%fine_lvl_dofs

      ! aggregate of every element (aggregates are unions of elements)
      allocate(agg(nelv))
      do e = 1, nelv
         agg(e) = amg%lvl(lvl)%map_finest2lvl((e - 1) * ndofe + 1)
      end do

      ! vertex -> local aggregates (CSR, duplicates allowed)
      vmax = int(maxval(amg%coef%dof%dof))
      allocate(vcnt(vmax), vptr(vmax + 1))
      vcnt = 0
      do e = 1, nelv
         do k = 1, lx
            do j = 1, lx
               do i = 1, lx
                  v = int(amg%coef%dof%dof(i, j, k, e))
                  vcnt(v) = vcnt(v) + 1
               end do
            end do
         end do
      end do
      vptr(1) = 1
      do v = 1, vmax
         vptr(v + 1) = vptr(v) + vcnt(v)
      end do
      allocate(vlist(vptr(vmax + 1) - 1))
      vcnt = 0
      do e = 1, nelv
         do k = 1, lx
            do j = 1, lx
               do i = 1, lx
                  v = int(amg%coef%dof%dof(i, j, k, e))
                  vlist(vptr(v) + vcnt(v)) = agg(e)
                  vcnt(v) = vcnt(v) + 1
               end do
            end do
         end do
      end do

      ! aggregate -> vertices (CSR, duplicates allowed)
      allocate(acnt(m), aptr(m + 1))
      acnt = 0
      do e = 1, nelv
         acnt(agg(e)) = acnt(agg(e)) + ndofe
      end do
      aptr(1) = 1
      do jj = 1, m
         aptr(jj + 1) = aptr(jj) + acnt(jj)
      end do
      allocate(alist(aptr(m + 1) - 1))
      acnt = 0
      do e = 1, nelv
         jj = agg(e)
         do k = 1, lx
            do j = 1, lx
               do i = 1, lx
                  alist(aptr(jj) + acnt(jj)) = int(amg%coef%dof%dof(i, j, k, e))
                  acnt(jj) = acnt(jj) + 1
               end do
            end do
         end do
      end do

      ! Global aggregate ids gid = rank*mmax + local index, and the ids of
      ! every aggregate (any rank) touching each local vertex: repeated gs
      ! MIN rounds, each revealing the smallest not yet reported id per
      ! vertex, until every copy of every vertex has reported.
      mmax = m
      call MPI_Allreduce(MPI_IN_PLACE, mmax, 1, MPI_INTEGER, MPI_MAX, NEKO_COMM)
      remote_ok = (real(pe_size, rp) * real(mmax, rp) .lt. &
           real(radix(1.0_rp), rp)**(digits(1.0_rp) - 1))
      allocate(nid(vmax), vid(NR, vmax))
      nid = 0
      if (pe_size .gt. 1 .and. remote_ok) then
         allocate(xk(n), xm(n), masked(n))
         xm_d = C_NULL_PTR
         call device_map(xm, xm_d, n)
         big = real(2, rp)**(digits(1.0_rp) + 2)
         i = 0
         do e = 1, nelv
            do k = 1, ndofe
               i = i + 1
               xk(i) = real(pe_rank * mmax + agg(e), rp)
            end do
         end do
         masked = .false.
         do round = 1, NR
            do i = 1, n
               xm(i) = merge(big, xk(i), masked(i))
            end do
            call device_memcpy(xm, xm_d, n, HOST_TO_DEVICE, .true.)
            call amg%gs_h%op(xm, n, GS_OP_MIN)
            call device_memcpy(xm, xm_d, n, DEVICE_TO_HOST, .true.)
            i = 0
            do e = 1, nelv
               do k = 1, lx
                  do j = 1, lx
                     do ia = 1, lx
                        i = i + 1
                        if (xm(i) .lt. big) then
                           v = int(amg%coef%dof%dof(ia, j, k, e))
                           g = nint(xm(i))
                           if (nid(v) .eq. 0) then
                              nid(v) = 1
                              vid(1, v) = g
                           else if (vid(nid(v), v) .ne. g) then
                              if (nid(v) .lt. NR) then
                                 nid(v) = nid(v) + 1
                                 vid(nid(v), v) = g
                              end if
                           end if
                           if (nint(xk(i)) .eq. g) masked(i) = .true.
                        end if
                     end do
                  end do
               end do
            end do
            nleft = count(.not. masked)
            call MPI_Allreduce(MPI_IN_PLACE, nleft, 1, MPI_INTEGER, MPI_SUM, &
                 NEKO_COMM)
            if (nleft .eq. 0) exit
         end do
         if (nleft .gt. 0) then
            call neko_warning("TAMG exact coarse diagonal: vertex valence " // &
                 "exceeds the id discovery rounds; remote exclusion incomplete")
         end if
         call device_unmap(xm, xm_d)
         deallocate(xk, xm, masked)
      else if (pe_size .gt. 1) then
         call neko_warning("TAMG exact coarse diagonal: too many aggregates " // &
              "for exact ids in this precision; remote couplings not excluded")
      end if

      ! Remote aggregates adjacent to local ones: inverse map
      ! remote gid -> local aggregates touching it (dense over the global
      ! id range; fine for O(10) ranks).
      nrem = 0
      rbase = 0
      if (pe_size .gt. 1 .and. remote_ok) then
         nrmax = pe_size * mmax
         allocate(rcnt(nrmax), rstamp(nrmax))
         rcnt = 0
         rstamp = 0
         do jj = 1, m
            do ia = aptr(jj), aptr(jj + 1) - 1
               v = alist(ia)
               do t = 1, nid(v)
                  g = vid(t, v)
                  if (g .le. pe_rank * mmax .or. g .gt. pe_rank * mmax + m) then
                     if (rstamp(g) .ne. jj) then
                        rstamp(g) = jj
                        rcnt(g) = rcnt(g) + 1
                     end if
                  end if
               end do
            end do
         end do
         allocate(rptr(nrmax + 1))
         rptr(1) = 1
         do g = 1, nrmax
            rptr(g + 1) = rptr(g) + rcnt(g)
         end do
         nrem = count(rcnt .gt. 0)
         allocate(rlist(max(rptr(nrmax + 1) - 1, 1)))
         rcnt = 0
         rstamp = 0
         do jj = 1, m
            do ia = aptr(jj), aptr(jj + 1) - 1
               v = alist(ia)
               do t = 1, nid(v)
                  g = vid(t, v)
                  if (g .le. pe_rank * mmax .or. g .gt. pe_rank * mmax + m) then
                     if (rstamp(g) .ne. jj) then
                        rstamp(g) = jj
                        rlist(rptr(g) + rcnt(g)) = jj
                        rcnt(g) = rcnt(g) + 1
                     end if
                  end if
               end do
            end do
         end do
      end if

      ! Rank boundaries: aggregates adjacent to a remote aggregate, extended
      ! twice by local distance 1. Interior aggregates of different ranks
      ! are at least distance 3 apart and never couple, so they keep the
      ! local colours; boundary aggregates use a rank-disjoint colour block.
      allocate(bnd(m), bnd_prev(m))
      bnd = .false.
      if (nrem .gt. 0) then
         do jj = 1, m
            do ia = aptr(jj), aptr(jj + 1) - 1
               v = alist(ia)
               do t = 1, nid(v)
                  g = vid(t, v)
                  if (g .le. pe_rank * mmax .or. g .gt. pe_rank * mmax + m) then
                     bnd(jj) = .true.
                     exit
                  end if
               end do
               if (bnd(jj)) exit
            end do
         end do
      end if
      do t = 1, 2
         bnd_prev = bnd
         do jj = 1, m
            if (bnd(jj)) cycle
            do ia = aptr(jj), aptr(jj + 1) - 1
               v = alist(ia)
               do ib = vptr(v), vptr(v + 1) - 1
                  if (bnd_prev(vlist(ib))) then
                     bnd(jj) = .true.
                     exit
                  end if
               end do
               if (bnd(jj)) exit
            end do
         end do
      end do
      nbnd = count(bnd)

      ! Greedy colouring. Excluded for jj: local aggregates within local
      ! distance 2, and local aggregates adjacent to a remote aggregate that
      ! jj is adjacent to as well (a length-2 path through another rank).
      mxc = 4096
      allocate(colour(m), mark(m), list1(m), used(mxc))
      colour = 0
      mark = 0
      used = 0
      nclr = 0
      do jj = 1, m
         n1 = 0
         do ia = aptr(jj), aptr(jj + 1) - 1
            v = alist(ia)
            do ib = vptr(v), vptr(v + 1) - 1
               kk = vlist(ib)
               if (mark(kk) .ne. jj) then
                  mark(kk) = jj
                  n1 = n1 + 1
                  list1(n1) = kk
               end if
            end do
         end do
         do t = 1, n1
            kk = list1(t)
            do ia = aptr(kk), aptr(kk + 1) - 1
               v = alist(ia)
               do ib = vptr(v), vptr(v + 1) - 1
                  ll = vlist(ib)
                  if (colour(ll) .gt. 0) used(colour(ll)) = jj
               end do
            end do
         end do
         if (nrem .gt. 0) then
            do ia = aptr(jj), aptr(jj + 1) - 1
               v = alist(ia)
               do t = 1, nid(v)
                  g = vid(t, v)
                  if (g .le. pe_rank * mmax .or. g .gt. pe_rank * mmax + m) then
                     do ib = rptr(g), rptr(g + 1) - 1
                        ll = rlist(ib)
                        if (colour(ll) .gt. 0) used(colour(ll)) = jj
                     end do
                  end if
               end do
            end do
         end if
         c = 1
         do while (used(c) .eq. jj)
            c = c + 1
            if (c .gt. mxc) call neko_error("TAMG colouring: too many colours")
         end do
         colour(jj) = c
         nclr = max(nclr, c)
      end do

      stride = nclr
      call MPI_Allreduce(MPI_IN_PLACE, stride, 1, MPI_INTEGER, MPI_MAX, &
           NEKO_COMM)
      ng = stride * (1 + pe_size)
      allocate(gcol(m), gcnt(ng))
      do jj = 1, m
         if (bnd(jj)) then
            gcol(jj) = stride * (1 + pe_rank) + colour(jj)
         else
            gcol(jj) = colour(jj)
         end if
      end do
      gcnt = 0
      do jj = 1, m
         gcnt(gcol(jj)) = gcnt(gcol(jj)) + 1
      end do
      call MPI_Allreduce(MPI_IN_PLACE, gcnt, ng, MPI_INTEGER, MPI_SUM, &
           NEKO_COMM)
      deallocate(mark, list1, used, vcnt, vptr, vlist, acnt, aptr, alist, agg)
      deallocate(nid, vid)
      if (allocated(rcnt)) deallocate(rcnt, rstamp, rptr, rlist)

      ! one (collective) matvec per globally non-empty colour
      allocate(pv(m), pw(m), dex(m))
      pv_d = C_NULL_PTR
      pw_d = C_NULL_PTR
      call device_map(pv, pv_d, m)
      call device_map(pw, pw_d, m)
      dex = dg
      nmv = 0
      do c = 1, ng
         if (gcnt(c) .eq. 0) cycle
         nmv = nmv + 1
         do jj = 1, m
            pv(jj) = merge(1.0_rp, 0.0_rp, gcol(jj) .eq. c)
         end do
         call device_memcpy(pv, pv_d, m, HOST_TO_DEVICE, .true.)
         call amg%device_matvec(pw, pv, pw_d, pv_d, lvl)
         call device_memcpy(pw, pw_d, m, DEVICE_TO_HOST, .true.)
         do jj = 1, m
            if (gcol(jj) .eq. c) dex(jj) = pw(jj)
         end do
      end do
      deallocate(gcnt)

      ! self-test: single-aggregate matvecs (collective, so every rank does
      ! the same number): 5 spread over the level, then up to 5 boundary ones
      errmax = 0.0_rp
      errbnd = 0.0_rp
      ntest = 5
      kk = 0
      do t = 1, 2 * ntest
         if (t .le. ntest) then
            jj = 1 + mod(((t - 1) * (m - 1)) / max(ntest - 1, 1), m)
         else
            jj = 0
            do ll = kk + 1, m
               if (bnd(ll)) then
                  jj = ll
                  kk = ll
                  exit
               end if
            end do
            if (jj .eq. 0) jj = 1
         end if
         ! one aggregate at a time globally: the matvec is collective and
         ! samples on different ranks may be coupled
         do r = 0, pe_size - 1
            pv = 0.0_rp
            if (pe_rank .eq. r) pv(jj) = 1.0_rp
            call device_memcpy(pv, pv_d, m, HOST_TO_DEVICE, .true.)
            call amg%device_matvec(pw, pv, pw_d, pv_d, lvl)
            call device_memcpy(pw, pw_d, m, DEVICE_TO_HOST, .true.)
            if (pe_rank .eq. r .and. abs(dex(jj)) .gt. 0.0_rp) then
               if (t .le. ntest) then
                  errmax = max(errmax, abs(pw(jj) - dex(jj)) / abs(dex(jj)))
               else if (bnd(jj)) then
                  errbnd = max(errbnd, abs(pw(jj) - dex(jj)) / abs(dex(jj)))
               end if
            end if
         end do
      end do
      deallocate(bnd, bnd_prev, gcol)

      ! masked rows keep the lumped fallback
      nbad = 0
      do jj = 1, m
         if (dex(jj) .gt. 0.0_rp) then
            dg(jj) = dex(jj)
         else
            nbad = nbad + 1
         end if
      end do

      call device_unmap(pv, pv_d)
      call device_unmap(pw, pw_d)
      deallocate(pv, pw, dex, colour)
    end associate
    t1 = MPI_Wtime()

    call MPI_Allreduce(MPI_IN_PLACE, nbad, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM)
    call MPI_Allreduce(MPI_IN_PLACE, nbnd, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM)
    call MPI_Allreduce(MPI_IN_PLACE, nrem, 1, MPI_INTEGER, MPI_SUM, NEKO_COMM)
    call MPI_Allreduce(MPI_IN_PLACE, errmax, 1, MPI_REAL_PRECISION, MPI_MAX, &
         NEKO_COMM)
    call MPI_Allreduce(MPI_IN_PLACE, errbnd, 1, MPI_REAL_PRECISION, MPI_MAX, &
         NEKO_COMM)
    write(log_buf, '(A,I2,A,I5,A,I7,A,ES7.0,A,ES7.0)') '-- AMG lvl', lvl, &
         ' exact diag: mv', nmv, ' bnd', nbnd, ' err', errmax, &
         ' bnd err', errbnd
    call neko_log%message(log_buf)
    write(log_buf, '(A,I2,A,I7,A,I7,A,F6.1,A)') '-- AMG lvl', lvl, &
         ' exact diag: masked', nbad, ' remote nb', nrem, ' setup', t1 - t0, 's'
    call neko_log%message(log_buf)

  end subroutine tamg_coarse_diag_exact

  !> free tree amg solver object
  subroutine tamg_mg_free(this)
    class(tamg_solver_t), intent(inout), target :: this
    integer :: i
    if (allocated(this%amg)) then
       call this%amg%free()
       deallocate(this%amg)
    end if
    if (allocated(this%smoo)) then
       do i = 0, (size(this%smoo)-1)
          call this%smoo(i)%free()
       end do
       deallocate(this%smoo)
    end if
    if (allocated(this%wrk)) then
       do i = 0, (size(this%wrk)-1)
          if (allocated(this%wrk(i)%r)) then
             if (NEKO_BCKND_DEVICE .eq. 1 .and. &
                  c_associated(this%wrk(i)%r_d)) then
                call device_unmap(this%wrk(i)%r, this%wrk(i)%r_d)
             end if
             deallocate(this%wrk(i)%r)
          end if
          if (allocated(this%wrk(i)%b)) then
             if (NEKO_BCKND_DEVICE .eq. 1 .and. &
                  c_associated(this%wrk(i)%b_d)) then
                call device_unmap(this%wrk(i)%b, this%wrk(i)%b_d)
             end if
             deallocate(this%wrk(i)%b)
          end if
          if (allocated(this%wrk(i)%x)) then
             if (NEKO_BCKND_DEVICE .eq. 1 .and. &
                  c_associated(this%wrk(i)%x_d)) then
                call device_unmap(this%wrk(i)%x, this%wrk(i)%x_d)
             end if
             deallocate(this%wrk(i)%x)
          end if
       end do
    end if
  end subroutine tamg_mg_free


  !> Re-estimate every level's eigenvalues on the next solve. Needed when
  !! the operator changes underneath the hierarchy.
  subroutine tamg_mg_invalidate_eigs(this)
    class(tamg_solver_t), intent(inout) :: this
    integer :: lvl
    do lvl = 0, this%amg%nlvls-1
       this%smoo(lvl)%recompute_eigs = .true.
    end do
  end subroutine tamg_mg_invalidate_eigs


  !> Set the eigenvalue re-estimation policy on every level.
  !! @param warm_start Restart from the saved eigenvector
  !! @param power_its_refresh Iterations to use when restarting
  subroutine tamg_mg_set_eig_refresh(this, warm_start, power_its_refresh)
    class(tamg_solver_t), intent(inout) :: this
    logical, intent(in) :: warm_start
    integer, intent(in) :: power_its_refresh
    integer :: lvl
    do lvl = 0, this%amg%nlvls-1
       this%smoo(lvl)%warm_start_eigs = warm_start
       this%smoo(lvl)%power_its_refresh = power_its_refresh
    end do
  end subroutine tamg_mg_set_eig_refresh


  !> Solver function for the TreeAMG solver object
  !! @param z The solution to be returned
  !! @param r The right-hand side
  !! @param n Number of dofs
  subroutine tamg_mg_solve(this, z, r, n)
    integer, intent(in) :: n
    class(tamg_solver_t), intent(inout) :: this
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    type(c_ptr) :: z_d
    type(c_ptr) :: r_d
    integer :: iter, max_iter, i
    logical :: zero_initial_guess

    max_iter = this%max_iter

    if (NEKO_BCKND_DEVICE .eq. 1) then
       z_d = device_get_ptr(z)
       r_d = device_get_ptr(r)
       ! Zero out the initial guess because we do not handle null
       ! spaces very well...
       call device_rzero(this%wrk(0)%x_d, n)
       call device_copy(this%wrk(0)%b_d, r_d, n)
       zero_initial_guess = .true.
       ! Call the amg cycle
       do iter = 1, max_iter
          call this%mg_cycle_d(zero_initial_guess)
          zero_initial_guess = .false.
       end do
       call device_copy(z_d, this%wrk(0)%x_d, n)
    else
       ! Zero out the initial guess becuase we do not handle null spaces
       ! very well...
       !OCL NORECURRENCE, NOVREC, NOALIAS
       !DIR$ CONCURRENT
       !DIR$ IVDEP
       !GCC$ ivdep
       !$omp parallel do
       do i = 1, n
          this%wrk(0)%x(i) = 0.0_rp
          this%wrk(0)%b(i) = r(i)
       end do
       !$omp end parallel do
       zero_initial_guess = .true.
       ! Call the amg cycle
       do iter = 1, max_iter
          call this%mg_cycle(zero_initial_guess)
          zero_initial_guess = .false.
       end do
       call copy(z, this%wrk(0)%x, n)
    end if
  end subroutine tamg_mg_solve


  !> multigrid cycle for the TreeAMG solver object
  !! @param this The Solver object.
  !! @param zero_initial_guess Flag for when the initial guess is zero
  subroutine tamg_mg_cycle(this, zero_initial_guess)
    class(tamg_solver_t), intent(inout), target :: this
    logical, intent(inout) :: zero_initial_guess
    character(len=2) :: lvl_name
    integer :: max_lvl, lvl

    max_lvl = this%nlvls-1
    ! Loop down hierarchy. Fine to coarse
    do lvl = 0, max_lvl-1
       write(lvl_name, '(I0)') lvl
       call profiler_start_region( "AMG_level_" // trim(lvl_name))
       associate(x => this%wrk(lvl)%x, b => this%wrk(lvl)%b, &
            r => this%wrk(lvl)%r, n => this%wrk(lvl)%n)
         !!----------!!
         !! SMOOTH   !!
         !!----------!!
         call this%smoo(lvl)%solve(x, b, n, this%amg, &
              zero_initial_guess)
         !!----------!!
         !! Residual !!
         !!----------!!
         call calc_resid(r, x, b, this%amg, lvl, n)
         !!----------!!
         !! Restrict !!
         !!----------!!
         call this%amg%interp_f2c(this%wrk(lvl+1)%b, r, lvl+1)

         call rzero(this%wrk(lvl+1)%x, this%wrk(lvl+1)%n)
         zero_initial_guess = .true.
       end associate
       call profiler_end_region( "AMG_level_" // trim(lvl_name))
    end do
    write(lvl_name, '(I0)') max_lvl
    call profiler_start_region( "AMG_level_" // trim(lvl_name))
    !!-------------------!!
    !! Call Coarse solve !!
    !!-------------------!!
    call this%smoo(max_lvl)%solve(this%wrk(max_lvl)%x, &
         this%wrk(max_lvl)%b, this%amg%lvl(max_lvl)%nnodes, this%amg, &
         zero_initial_guess)
    call profiler_end_region( "AMG_level_" // trim(lvl_name))

    zero_initial_guess = .false.
    ! Loop up hierarchy. Coarse to fine
    do lvl = max_lvl-1, 0, -1
       write(lvl_name, '(I0)') lvl
       call profiler_start_region( "AMG_level_" // trim(lvl_name))
       associate(x => this%wrk(lvl)%x, b => this%wrk(lvl)%b, &
            r => this%wrk(lvl)%r, n => this%wrk(lvl)%n)
         !!----------!!
         !! Project  !!
         !!----------!!
         call this%amg%interp_c2f(r, this%wrk(lvl+1)%x, lvl+1)
         !!----------!!
         !! Correct  !!
         !!----------!!
         call add2(x, r, n)
         !!----------!!
         !! SMOOTH   !!
         !!----------!!
         call this%smoo(lvl)%solve(x, b, n, this%amg)
       end associate
       call profiler_end_region( "AMG_level_" // trim(lvl_name))
    end do
  end subroutine tamg_mg_cycle

  !> multigrid cycle for the TreeAMG solver object on device
  !! @param this The Solver object.
  !! @param zero_initial_guess Flag for when the initial guess is zero
  subroutine tamg_mg_cycle_d(this, zero_initial_guess)
    class(tamg_solver_t), intent(inout), target :: this
    logical, intent(inout) :: zero_initial_guess
    character(len=2) :: lvl_name
    integer :: max_lvl, lvl

    max_lvl = this%nlvls-1
    ! Loop down hierarchy. Fine to coarse
    do lvl = 0, max_lvl-1
       write(lvl_name, '(I0)') lvl
       call profiler_start_region( "AMG_level_" // trim(lvl_name))
       associate(x => this%wrk(lvl)%x, x_d => this%wrk(lvl)%x_d, &
            b => this%wrk(lvl)%b, b_d => this%wrk(lvl)%b_d, &
            r => this%wrk(lvl)%r, r_d => this%wrk(lvl)%r_d, &
            n => this%wrk(lvl)%n)
         !!----------!!
         !! SMOOTH   !!
         !!----------!!
         call this%smoo(lvl)%device_solve(x, b, x_d, b_d, n, this%amg, &
              zero_initial_guess)
         !!----------!!
         !! Residual !!
         !!----------!!
         call this%amg%device_matvec(r, x, r_d, x_d, lvl)
         call device_sub3(r_d, b_d, r_d, n)
         !!----------!!
         !! Restrict !!
         !!----------!!
         call this%amg%interp_f2c_d(this%wrk(lvl+1)%b_d, r_d, lvl+1)

         call device_rzero(this%wrk(lvl+1)%x_d, this%wrk(lvl+1)%n)
         zero_initial_guess = .true.
       end associate
       call profiler_end_region( "AMG_level_" // trim(lvl_name))
    end do
    write(lvl_name, '(I0)') max_lvl
    call profiler_start_region( "AMG_level_" // trim(lvl_name))
    !!-------------------!!
    !! Call Coarse solve !!
    !!-------------------!!
    call this%smoo(max_lvl)%device_solve( &
         this%wrk(max_lvl)%x, this%wrk(max_lvl)%b, &
         this%wrk(max_lvl)%x_d, this%wrk(max_lvl)%b_d, &
         this%amg%lvl(max_lvl)%nnodes, this%amg, &
         zero_initial_guess)
    call profiler_end_region( "AMG_level_" // trim(lvl_name))

    zero_initial_guess = .false.
    ! Loop up hierarchy. Coarse to fine
    do lvl = max_lvl-1, 0, -1
       write(lvl_name, '(I0)') lvl
       call profiler_start_region( "AMG_level_" // trim(lvl_name))
       associate(x => this%wrk(lvl)%x, x_d => this%wrk(lvl)%x_d, &
            b => this%wrk(lvl)%b, b_d => this%wrk(lvl)%b_d, &
            r => this%wrk(lvl)%r, r_d => this%wrk(lvl)%r_d, &
            n => this%wrk(lvl)%n)
         !!----------!!
         !! Project  !!
         !!----------!!
         call this%amg%interp_c2f_d(r_d, this%wrk(lvl+1)%x_d, lvl+1, r)
         !!----------!!
         !! Correct  !!
         !!----------!!
         call device_add2(x_d, r_d, n)
         !!----------!!
         !! SMOOTH   !!
         !!----------!!
         call this%smoo(lvl)%device_solve(x, b, x_d, b_d, n, this%amg)
       end associate
       call profiler_end_region( "AMG_level_" // trim(lvl_name))
    end do
  end subroutine tamg_mg_cycle_d


  !> Wrapper function to calculate residyal
  !! @param r The residual to be returned
  !! @param x The current solution
  !! @param b The right-hand side
  !! @param amg The TreeAMG object
  !! @param lvl Current level of the cycle
  !! @param n Number of dofs
  subroutine calc_resid(r, x, b, amg, lvl, n)
    integer, intent(in) :: n
    real(kind=rp), intent(inout) :: r(n)
    real(kind=rp), intent(inout) :: x(n)
    real(kind=rp), intent(inout) :: b(n)
    type(tamg_hierarchy_t), intent(inout) :: amg
    integer, intent(in) :: lvl
    integer :: i
    call amg%matvec(r, x, lvl)
    call add2s1(r, b, -1.0_rp, n)
  end subroutine calc_resid


  subroutine print_preagg_info(lvl, nagg, agg_type)
    integer, intent(in) :: lvl, nagg, agg_type
    character(len=LOG_SIZE) :: log_buf
    !TODO: calculate min and max agg size
    if (agg_type .eq. 1) then
       write(log_buf, '(A8,I2,A31)') '-- level', lvl, &
            '-- Calling Greedy Aggregation'
    else if (agg_type .eq. 2) then
       write(log_buf, '(A8,I2,A33)') '-- level', lvl, &
            '-- Calling Pairwise Aggregation'
    else
       write(log_buf, '(A8,I2,A31)') '-- level', lvl, &
            '-- UNKNOWN Aggregation'
    end if
    call neko_log%message(log_buf)
    write(log_buf, '(A33,I6)') 'Target Aggregates:', nagg
    call neko_log%message(log_buf)
  end subroutine print_preagg_info

  subroutine print_resid_info(r, x, b, r_d, x_d, b_d, amg, lvl, n)
    integer, intent(in) :: lvl, n
    real(kind=rp), intent(inout) :: r(n)
    real(kind=rp), intent(inout) :: x(n)
    real(kind=rp), intent(inout) :: b(n)
    type(c_ptr) :: r_d
    type(c_ptr) :: x_d
    type(c_ptr) :: b_d
    type(tamg_hierarchy_t), intent(inout) :: amg
    real(kind=rp) :: val
    character(len=LOG_SIZE) :: log_buf

    call amg%device_matvec(r, x, r_d, x_d, lvl)
    call device_sub3(r_d, b_d, r_d, n)
    val = device_glsc2(r_d, r_d, n)

    write(log_buf, '(A33,I6,F12.6)') 'tAMG resid:', lvl, val
    call neko_log%message(log_buf)
  end subroutine print_resid_info

  !> Create index mapping between levels and directly to finest level
  !! @param amg The tamg hierarchy
  subroutine fill_lvl_map(amg)
    type(tamg_hierarchy_t), intent(inout) :: amg
    integer, allocatable :: dof2gid(:)
    integer :: i, j, k, l, nid, n, nmap
    do j = 1, amg%lvl(1)%nnodes
       do k = 1, amg%lvl(1)%nodes(j)%ndofs
          nid = amg%lvl(1)%nodes(j)%dofs(k)
          amg%lvl(1)%map_finest2lvl(nid) = amg%lvl(1)%nodes(j)%gid
       end do
    end do
    ! map_finest2lvl is allocated 0:fine_lvl_dofs (element 0 carries the
    ! length for the device kernels), so size() is one more than the number
    ! of dofs and must not be used as the upper loop bound.
    n = amg%lvl(1)%fine_lvl_dofs
    do l = 2, amg%nlvls
       ! Invert the level's node->dof lists once into dof2gid, then compose
       ! with the level below. Searching every node for every dof instead
       ! is O(n * dofs-on-level) and dominates setup on large coarse grids.
       allocate(dof2gid(amg%lvl(l)%fine_lvl_dofs))
       dof2gid = 0
       do j = 1, amg%lvl(l)%nnodes
          do k = 1, amg%lvl(l)%nodes(j)%ndofs
             dof2gid(amg%lvl(l)%nodes(j)%dofs(k)) = amg%lvl(l)%nodes(j)%gid
          end do
       end do
       do i = 1, n
          nid = amg%lvl(l-1)%map_finest2lvl(i)
          if (nid .ge. 1 .and. nid .le. amg%lvl(l)%fine_lvl_dofs) then
             if (dof2gid(nid) .gt. 0) then
                amg%lvl(l)%map_finest2lvl(i) = dof2gid(nid)
             end if
          end if
       end do
       deallocate(dof2gid)
    end do
    if (NEKO_BCKND_DEVICE .eq. 1) then
       ! The whole array, element 0 included, is what gets mirrored; keep
       ! this independent of the loop bound above.
       nmap = size(amg%lvl(1)%map_finest2lvl)
       do l = 1, amg%nlvls
          amg%lvl(l)%map_finest2lvl(0) = nmap
          call device_memcpy( amg%lvl(l)%map_finest2lvl, &
               amg%lvl(l)%map_finest2lvl_d, nmap, &
               HOST_TO_DEVICE, .true.)
          call device_memcpy( amg%lvl(l)%map_f2c, &
               amg%lvl(l)%map_f2c_d, amg%lvl(l)%fine_lvl_dofs+1, &
               HOST_TO_DEVICE, .true.)
       end do
    end if
  end subroutine fill_lvl_map

  !> Build the transpose of map_finest2lvl in CSR form: for every aggregate
  !! on a level, the list of finest-level dofs that map to it. This lets the
  !! flat matvec walk aggregates rather than dofs, so its restriction becomes
  !! a segmented reduction with disjoint writes instead of an atomic scatter.
  !! Two O(n) passes per level.
  !! @param amg The tamg hierarchy, with the level maps already filled
  subroutine build_agg_csr(amg)
    type(tamg_hierarchy_t), intent(inout) :: amg
    !> Cache-line padding for the per-thread partials, in elements of rp.
    !! 32 doubles = 256 B, the A64FX line; a multiple of the 64 B line
    !! everywhere else.
    integer, parameter :: PART_PAD = 32
    integer, allocatable :: fill(:), seen(:)
    integer :: l, i, j, k, c, n, nagg, nthrds, ldpart

    n = amg%lvl(1)%fine_lvl_dofs

    nthrds = 1
    !$ nthrds = omp_get_max_threads()

    ! The restriction and prolongation operators thread over nodes on the
    ! assumption that a level's nodes partition the dofs of the level below,
    ! which is what makes their writes disjoint. Check it once here rather
    ! than let a violation turn into a silent data race at solve time.
    do l = 1, amg%nlvls
       allocate(seen(amg%lvl(l)%fine_lvl_dofs))
       seen = 0
       do j = 1, amg%lvl(l)%nnodes
          do k = 1, amg%lvl(l)%nodes(j)%ndofs
             i = amg%lvl(l)%nodes(j)%dofs(k)
             if (i .lt. 1 .or. i .gt. amg%lvl(l)%fine_lvl_dofs) then
                call neko_error('TAMG: node dof outside the level')
             end if
             if (seen(i) .ne. 0) then
                call neko_error('TAMG: dof shared by two nodes on a level')
             end if
             seen(i) = j
          end do
       end do
       deallocate(seen)
    end do

    do l = 1, amg%nlvls
       nagg = amg%lvl(l)%nnodes

       allocate(amg%lvl(l)%agg_ptr(nagg + 1))
       allocate(amg%lvl(l)%agg_dof(n))
       allocate(fill(nagg))

       ! Levels with too few aggregates to give every thread one fall back
       ! to a reduction over dofs into per-thread partials; the coarsest
       ! level always does, it has a single aggregate from aggregate_end.
       ! Pad the leading dimension so every thread's column starts in its
       ! own cache line: unpadded, nagg = 1 puts all nthrds accumulators in
       ! a single line and the inner loop degenerates into line ping-pong,
       ! which is no better than the atomic it replaces. PART_PAD is in
       ! elements and covers the widest line in play (A64FX is 256 B, x86
       ! and most AArch64 are 64 B).
       if (nagg .lt. 2 * nthrds) then
          ldpart = ((nagg + PART_PAD - 1) / PART_PAD) * PART_PAD
          allocate(amg%lvl(l)%agg_part(ldpart, nthrds))
       end if

       associate(aptr => amg%lvl(l)%agg_ptr, adof => amg%lvl(l)%agg_dof, &
            map => amg%lvl(l)%map_finest2lvl)

         ! Histogram the aggregate sizes into aptr(2:nagg+1). map_finest2lvl
         ! is never initialised, so a dof left out of the aggregation shows
         ! up here rather than as silent garbage in the matvec.
         aptr = 0
         do i = 1, n
            c = map(i)
            if (c .lt. 1 .or. c .gt. nagg) then
               call neko_error('TAMG: dof not covered by the aggregation')
            end if
            aptr(c + 1) = aptr(c + 1) + 1
         end do

         ! Prefix sum into 1-based offsets
         aptr(1) = 1
         do c = 1, nagg
            aptr(c + 1) = aptr(c + 1) + aptr(c)
         end do

         ! Place the dof ids. i ascends, so each aggregate's list comes out
         ! sorted, which keeps the indirect reads in the matvec monotone.
         fill = 0
         do i = 1, n
            c = map(i)
            adof(aptr(c) + fill(c)) = i
            fill(c) = fill(c) + 1
         end do
       end associate

       deallocate(fill)
    end do

  end subroutine build_agg_csr
end module tamgx_multigrid
