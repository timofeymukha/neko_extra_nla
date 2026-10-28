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
!> rs_phmg: a copy of Neko's phmg (fix/user_precon_init 54c505c) registered
!! as user type 'rs_phmg', with a switchable N = 1 coarse solver
!! (`coarse_solver`: tamg | cg). Development vehicle of the robust_solvers
!! project; everything else is identical to phmg.
!> Hybrid ph-multigrid preconditioner
module rs_phmg
  use num_types, only : rp, dp, i8
  use precon, only : pc_t, precon_allocate, register_precon
  use cg, only : cg_t
  use rs_line, only : rs_line_t, rs_mem_report
  use rs_colagg, only : rs_colagg_t
  use gather_scatter, only : gs_t, GS_OP_ADD
  use space, only : space_t, GLL
  use dofmap, only : dofmap_t
  use field, only : field_t
  use coefs, only : coef_t, COEF_FULL, COEF_OPERATOR
  use mesh, only : mesh_t
  use bc, only : bc_t
  use bc_list, only : bc_list_t
  use scalar_bc_projector, only : scalar_bc_projector_t
  use dirichlet, only : dirichlet_t
  use utils, only : neko_error, neko_warning
  use cheby, only : cheby_t
  use cheby_device, only : cheby_device_t
  use jacobi, only : jacobi_t
  use device_jacobi, only : device_jacobi_t
  use schwarz, only : schwarz_t
  use ax_product, only : ax_t, ax_helm_allocator
  use tamgx_multigrid, only : tamg_solver_t   ! the user's patched tree-AMG (Jacobi scaling optional)
  use interpolation, only : interpolator_t
  use json_module, only : json_file
  use json_utils, only : json_get_or_default, json_get
  use math, only : copy, col2, add2, add2s2, add2s1, glsc3
  use device, only : device_map, device_memcpy, device_free, device_deassociate, &
       HOST_TO_DEVICE, DEVICE_TO_HOST
  use, intrinsic :: iso_c_binding, only : c_null_ptr
  use device_math, only : device_glsc3, device_add2s1, device_add2s2
  use device, only : device_get_ptr, device_stream_wait_event, glb_cmd_queue, &
       glb_cmd_event
  use device_math, only : device_rzero, device_copy, device_add2, &
       device_add2s2, device_invcol2, device_glsc2, device_col2, device_add2s1
  use neko_config, only : NEKO_BCKND_DEVICE
  use comm, only : pe_rank
  use krylov, only : ksp_t, ksp_monitor_t, KSP_MAX_ITER, &
       krylov_solver_factory
  use profiler, only : profiler_start_region, profiler_end_region
  use logger, only : neko_log, LOG_SIZE
  use, intrinsic :: iso_c_binding
  implicit none
  private


  type, private :: rs_phmg_lvl_t
     integer :: lvl = -1
     integer :: smoother_itrs = 10
     type(space_t), pointer :: Xh
     type(dofmap_t), pointer :: dm_Xh
     type(gs_t), pointer :: gs_h
     type(schwarz_t) :: schwarz
     type(cheby_t) :: cheby
     type(cheby_device_t) :: cheby_device
     type(jacobi_t) :: jacobi
     type(rs_line_t) :: line
     type(device_jacobi_t) :: device_jacobi
     type(coef_t), pointer :: coef
     type(scalar_bc_projector_t) :: bc_projector
     type(dirichlet_t) :: bc
     type(field_t) :: r, w, z
  end type rs_phmg_lvl_t

  type, public :: rs_phmg_hrchy_t
     type(rs_phmg_lvl_t), allocatable :: lvl(:)
  end type rs_phmg_hrchy_t


  type, public, extends(pc_t) :: rs_phmg_t
     type(tamg_solver_t) :: amg_solver
     !> Coarse solver on the N = 1 level: 'tamg' (Neko's tree-AMG, the
     !> phmg default) or 'cg' (CG + Jacobi to the relative tolerance
     !> coarse_tol, i.e. an 'exact' coarse solve for diagnostics).
     character(len=:), allocatable :: coarse_solver
     real(kind=rp) :: coarse_tol = 1.0e-8_rp
     integer :: coarse_max_iter = 2000
     type(cg_t) :: crs_cg
     type(rs_colagg_t) :: colagg
     integer :: colagg_smooth = 3
     integer :: colagg_cycles = 0
     integer :: colagg_dense_max = 2000
     real(kind=rp) :: colagg_theta = 1.0_rp
     logical :: colagg_hierarchy = .false.
     logical :: colagg_crs_host = .false.
     !> sparse direct coarse solve above colagg_dense_max: rs_spchol (default)
     !! or MKL PARDISO (colagg_direct_solver = "pardiso", looked up at run time)
     logical :: colagg_direct = .false.
     character(len=:), allocatable :: colagg_direct_solver
     real(kind=rp) :: colagg_kappa = 4.0_rp
     type(jacobi_t) :: crs_jacobi
     integer :: crs_calls = 0
     integer :: crs_iters = 0
     integer :: crs_fails = 0
     logical :: line_columns = .false.
     logical :: line_geometric = .false.
     integer :: cheby_power_its = 150
     integer :: line_direction = 0
     real(kind=rp) :: line_iso_ratio = 0.0_rp
     logical :: line_global_columns = .true.
     !> plane strips for elements with two strong directions (rs_line)
     logical :: line_planes = .false.
     !> tree-AMG coarse solver (coarse_solver tamg, also the fallback when the
     !! aggregation merges too little): Jacobi-scaled Chebyshev with the
     !! exact coarse diagonal of the user's patch (coarse_grid.jacobi_smoother,
     !! coarse_grid.jacobi_probes: -1 exact on the device, > 0 probes)
     logical :: tamg_jacobi = .false.
     integer :: tamg_jacobi_probes = -1
     real(kind=rp) :: line_plane_ratio = 16.0_rp
     !> line_factor_precision: 'double' (default) or 'single' device factors
     character(len=:), allocatable :: line_factor_precision
     !> > 0: estimate lambda_max of M^{-1}A per level by this many steps of
     !! PCG-Lanczos instead of Neko's 150-step power iteration
     integer :: cheby_lanczos_its = 20
     real(kind=rp) :: colagg_min_merge = 0.05_rp
     !> preconditioner of the coarse CG: 'jacobi' (default) or 'line'
     character(len=:), allocatable :: coarse_pc
     integer :: nlvls
     type(rs_phmg_hrchy_t) :: rs_phmg_hrchy
     class(ax_t), allocatable :: ax
     type(interpolator_t), allocatable :: intrp(:)
     !> Coordinate interpolators.
     type(interpolator_t), allocatable :: crd_intrp(:)
     type(mesh_t), pointer :: msh
     !> Fine-level metrics_version at the last refresh, to detect a change.
     integer :: last_metrics_version = -1
     !> When .false., update() does nothing and coarse levels stay frozen
     !> at their initial geometry. When .true., update() interpolates
     !> the fine-level geometry to the coarse levels.
     logical :: update_enabled = .false.
     !> Whether a refresh also re-estimates the Chebyshev eigenvalues.
     logical :: refresh_eigs = .true.
     !> Re-estimate eigenvalues only every N-th refresh.
     integer :: refresh_eigs_frequency = 20
     !> Restart eigenvalue estimation from the previous eigenvector.
     logical :: eigs_warm_start = .true.
     !> Power iterations used for warm-started re-estimations
     integer :: power_its_refresh = 20
     !> Refreshes done so far, for refresh_eigs_frequency
     integer :: n_refresh = 0
     !> Which accelerator the Chebyshev smoother uses
     character(len=:), allocatable :: cheby_acc
     !> RS_PHMG_SYMCHECK: symmetry test of the preconditioner at the first solve
     logical :: symcheck_done = .false.
   contains
     procedure, pass(this) :: init => rs_phmg_init
     procedure, pass(this) :: init_from_components => &
          rs_phmg_init_from_components
     procedure, pass(this) :: free => rs_phmg_free
     procedure, pass(this) :: solve => rs_phmg_solve
     procedure, pass(this) :: update => rs_phmg_update
     procedure, private, pass(this) :: mg_cycle => rs_phmg_mg_cycle
  end type rs_phmg_t

  public :: rs_phmg_register_types

contains

  subroutine rs_phmg_init(this, coef, bclst, json)
    class(rs_phmg_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    type(json_file), intent(inout) :: json
    integer :: crs_tamg_lvls, crs_tamg_itrs, crs_tamg_cheby_degree
    integer :: smoother_itrs
    character(len=:), allocatable :: cheby_acc
    integer, allocatable :: pcrs_sched(:)
    logical :: update_enabled

    call json_get_or_default(json, 'smoother_iterations', &
         smoother_itrs, 3)

    call json_get_or_default(json, 'smoother_cheby_acc', &
         cheby_acc, "jacobi")

    call json_get_or_default(json, 'coarse_grid.levels', &
         crs_tamg_lvls, 3)

    call json_get_or_default(json, 'coarse_grid.iterations', &
         crs_tamg_itrs, 1)

    call json_get_or_default(json, 'coarse_grid.cheby_degree', &
         crs_tamg_cheby_degree, 4)

    if (json%valid_path('pcoarsening_schedule')) then
       call json_get(json, 'pcoarsening_schedule', pcrs_sched)
    else
       allocate(pcrs_sched(2))
       pcrs_sched(1) = 3
       pcrs_sched(2) = 1
    end if

    call json_get_or_default(json, 'update.enabled', &
         update_enabled, .false.)

    ! Control the eigenvalue re-estimation; geometry always refreshes.
    call json_get_or_default(json, 'update.eigs.enabled', &
         this%refresh_eigs, .true.)
    call json_get_or_default(json, 'update.eigs.frequency', &
         this%refresh_eigs_frequency, 20)
    call json_get_or_default(json, 'update.eigs.warm_start', &
         this%eigs_warm_start, .true.)
    call json_get_or_default(json, 'update.eigs.warm_start_iterations', &
         this%power_its_refresh, 20)

    call json_get_or_default(json, 'coarse_solver', this%coarse_solver, 'tamg')
    call json_get_or_default(json, 'coarse_tol', this%coarse_tol, 1.0e-8_rp)
    call json_get_or_default(json, 'coarse_max_iter', this%coarse_max_iter, 2000)
    call json_get_or_default(json, 'line_columns', this%line_columns, .false.)
    call json_get_or_default(json, 'line_geometric', this%line_geometric, .false.)
    call json_get_or_default(json, 'cheby_power_its', this%cheby_power_its, 150)
    call json_get_or_default(json, 'line_direction', this%line_direction, 0)
    call json_get_or_default(json, 'line_iso_ratio', this%line_iso_ratio, 0.0_rp)
    call json_get_or_default(json, 'line_global_columns', this%line_global_columns, .true.)
    call json_get_or_default(json, 'line_planes', this%line_planes, .false.)
    call json_get_or_default(json, 'coarse_grid.jacobi_smoother', this%tamg_jacobi, .false.)
    call json_get_or_default(json, 'coarse_grid.jacobi_probes', this%tamg_jacobi_probes, -1)
    call json_get_or_default(json, 'line_plane_ratio', this%line_plane_ratio, 16.0_rp)
    call json_get_or_default(json, 'line_factor_precision', this%line_factor_precision, 'double')
    if (trim(this%line_factor_precision) .ne. 'double' .and. trim(this%line_factor_precision) .ne. 'single') &
         call neko_error('rs_phmg: line_factor_precision must be double or single')
    call json_get_or_default(json, 'cheby_lanczos_its', this%cheby_lanczos_its, 20)
    call json_get_or_default(json, 'colagg_min_merge', this%colagg_min_merge, 0.05_rp)
    call json_get_or_default(json, 'coarse_pc', this%coarse_pc, 'auto')
    call json_get_or_default(json, 'colagg_smooth', this%colagg_smooth, 3)
    call json_get_or_default(json, 'colagg_cycles', this%colagg_cycles, 0)
    ! on the device the dense coarse solve runs through the vendor solver and is cheap up to ~16k unknowns
    call json_get_or_default(json, 'colagg_dense_max', this%colagg_dense_max, merge(16000, 2000, NEKO_BCKND_DEVICE .eq. 1))
    call json_get_or_default(json, 'colagg_theta', this%colagg_theta, 1.0_rp)
    call json_get_or_default(json, 'colagg_hierarchy', this%colagg_hierarchy, .false.)
    call json_get_or_default(json, 'colagg_crs_host', this%colagg_crs_host, .false.)
    call json_get_or_default(json, 'colagg_direct', this%colagg_direct, .false.)
    call json_get_or_default(json, 'colagg_direct_solver', this%colagg_direct_solver, 'spchol')
    call json_get_or_default(json, 'colagg_kappa', this%colagg_kappa, 4.0_rp)

    call this%init_from_components(coef, bclst, smoother_itrs, &
         cheby_acc, crs_tamg_lvls, crs_tamg_itrs, crs_tamg_cheby_degree,&
         pcrs_sched, update_enabled)

  end subroutine rs_phmg_init

  subroutine rs_phmg_init_from_components(this, coef, bclst, smoother_itrs, &
       cheby_acc, crs_tamg_lvls, crs_tamg_itrs, crs_tamg_cheby_degree, &
       pcrs_sched, update_enabled)
    class(rs_phmg_t), intent(inout), target :: this
    type(coef_t), intent(in), target :: coef
    type(bc_list_t), intent(inout), target :: bclst
    integer, intent(in) :: smoother_itrs
    character(len=:), allocatable :: cheby_acc
    integer, intent(in) :: crs_tamg_lvls, crs_tamg_itrs
    integer, intent(in) :: crs_tamg_cheby_degree
    integer, intent(in), allocatable :: pcrs_sched(:)
    logical, intent(in), optional :: update_enabled
    integer :: lx_crs, lx_mid
    integer, allocatable :: lx_lvls(:)
    integer :: n, i, j, st
    class(bc_t), pointer :: bc_j
    logical :: use_jacobi, use_cheby
    use_jacobi = .true.
    use_cheby = .true.

    this%msh => coef%msh
    call rs_mem_report('rs_phmg start of setup')
    this%cheby_acc = trim(cheby_acc)

    if (present(update_enabled)) then
       this%update_enabled = update_enabled
    else
       this%update_enabled = .false.
    end if

    this%nlvls = size(pcrs_sched) + 1
    allocate(lx_lvls(0:this%nlvls - 1))
    lx_lvls(1:) = pcrs_sched + 1

    allocate(this%rs_phmg_hrchy%lvl(0:this%nlvls - 1))

    this%rs_phmg_hrchy%lvl(0)%lvl = 0
    this%rs_phmg_hrchy%lvl(0)%smoother_itrs = smoother_itrs
    this%rs_phmg_hrchy%lvl(0)%Xh => coef%Xh
    this%rs_phmg_hrchy%lvl(0)%coef => coef
    this%rs_phmg_hrchy%lvl(0)%dm_Xh => coef%dof
    this%rs_phmg_hrchy%lvl(0)%gs_h => coef%gs_h

    do i = 1, this%nlvls - 1
       allocate(this%rs_phmg_hrchy%lvl(i)%Xh)
       allocate(this%rs_phmg_hrchy%lvl(i)%dm_Xh)
       allocate(this%rs_phmg_hrchy%lvl(i)%gs_h)
       allocate(this%rs_phmg_hrchy%lvl(i)%coef)

       this%rs_phmg_hrchy%lvl(i)%lvl = i
       this%rs_phmg_hrchy%lvl(i)%smoother_itrs = smoother_itrs
       call this%rs_phmg_hrchy%lvl(i)%Xh%init(GLL, lx_lvls(i), lx_lvls(i), &
            lx_lvls(i))
       call this%rs_phmg_hrchy%lvl(i)%dm_Xh%init(coef%msh, &
            this%rs_phmg_hrchy%lvl(i)%Xh)
       call this%rs_phmg_hrchy%lvl(i)%gs_h%init(this%rs_phmg_hrchy%lvl(i)%dm_Xh)
       ! A coarse level reads G_ij, h1, h2, B and mult and nothing else, so
       ! retaining the rest costs memory for no purpose. The exception is a
       ! moving mesh, where update() has to rebuild the geometry from the
       ! derivative arrays.
       if (this%update_enabled) then
          call this%rs_phmg_hrchy%lvl(i)%coef%init( &
               this%rs_phmg_hrchy%lvl(i)%gs_h, COEF_FULL)
       else
          call this%rs_phmg_hrchy%lvl(i)%coef%init( &
               this%rs_phmg_hrchy%lvl(i)%gs_h, COEF_OPERATOR)
       end if
    end do

    do i = 0, this%nlvls - 1
       call this%rs_phmg_hrchy%lvl(i)%r%init(this%rs_phmg_hrchy%lvl(i)%dm_Xh)
       call this%rs_phmg_hrchy%lvl(i)%w%init(this%rs_phmg_hrchy%lvl(i)%dm_Xh)
       call this%rs_phmg_hrchy%lvl(i)%z%init(this%rs_phmg_hrchy%lvl(i)%dm_Xh)

       this%rs_phmg_hrchy%lvl(i)%coef%ifh2 = coef%ifh2
       call copy(this%rs_phmg_hrchy%lvl(i)%coef%h1, coef%h1, &
            this%rs_phmg_hrchy%lvl(i)%dm_Xh%size())

       call this%rs_phmg_hrchy%lvl(i)%bc%init_base(this%rs_phmg_hrchy%lvl(i)%coef)
       if (bclst%size() .gt. 0 ) then
          do j = 1, bclst%size()
             bc_j => bclst%get(j)
             call this%rs_phmg_hrchy%lvl(i)%bc%mark_facets(bc_j%marked_facet)
          end do
       end if
       call this%rs_phmg_hrchy%lvl(i)%bc%finalize()
       call this%rs_phmg_hrchy%lvl(i)%bc%set_g(0.0_rp)
       call this%rs_phmg_hrchy%lvl(i)%bc_projector%mark(this%rs_phmg_hrchy%lvl(i)%bc)

       !> Initialize Smoothers
       if (trim(cheby_acc) .eq. "schwarz") then
          call this%rs_phmg_hrchy%lvl(i)%schwarz%init( &
               this%rs_phmg_hrchy%lvl(i)%Xh, &
               this%rs_phmg_hrchy%lvl(i)%dm_Xh, &
               this%rs_phmg_hrchy%lvl(i)%gs_h, &
               this%rs_phmg_hrchy%lvl(i)%bc_projector, &
               coef%msh)
       end if

       ! the point Jacobi preconditioner only for smoother_cheby_acc jacobi
       if (trim(cheby_acc) .eq. "jacobi") then
          if (NEKO_BCKND_DEVICE .eq. 1) then
             call this%rs_phmg_hrchy%lvl(i)%device_jacobi%init_from_components(&
                  this%rs_phmg_hrchy%lvl(i)%coef, &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh, &
                  this%rs_phmg_hrchy%lvl(i)%gs_h)
          else
             call this%rs_phmg_hrchy%lvl(i)%jacobi%init_from_components(&
                  this%rs_phmg_hrchy%lvl(i)%coef, &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh, &
                  this%rs_phmg_hrchy%lvl(i)%gs_h)
          end if
       end if

       if (NEKO_BCKND_DEVICE .eq. 1) then
          if (trim(cheby_acc) .eq. "jacobi") then
             call this%rs_phmg_hrchy%lvl(i)%cheby_device%init( &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh%size(), smoother_itrs, &
                  this%rs_phmg_hrchy%lvl(i)%device_jacobi)
             st = 1
          else if (trim(cheby_acc) .eq. "line") then
             ! the line smoother's device path (rs_line_cuda); init later
             call this%rs_phmg_hrchy%lvl(i)%cheby_device%init( &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh%size(), smoother_itrs, &
                  this%rs_phmg_hrchy%lvl(i)%line)
             st = 1
          else
             call this%rs_phmg_hrchy%lvl(i)%cheby_device%init( &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh%size(), smoother_itrs)
             st = 0
             if (trim(cheby_acc) .eq. "schwarz") then
                this%rs_phmg_hrchy%lvl(i)%cheby_device%schwarz => &
                     this%rs_phmg_hrchy%lvl(i)%schwarz
                st = 2
             end if
          end if
       else
          if (trim(cheby_acc) .eq. "jacobi") then
             call this%rs_phmg_hrchy%lvl(i)%cheby%init( &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh%size(), smoother_itrs, &
                  this%rs_phmg_hrchy%lvl(i)%jacobi)
             st = 1
          else if (trim(cheby_acc) .eq. "line") then
             ! the line blocks need this%ax, allocated below; init later
             call this%rs_phmg_hrchy%lvl(i)%cheby%init( &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh%size(), smoother_itrs, &
                  this%rs_phmg_hrchy%lvl(i)%line)
             st = 1
          else
             call this%rs_phmg_hrchy%lvl(i)%cheby%init( &
                  this%rs_phmg_hrchy%lvl(i)%dm_Xh%size(), smoother_itrs)
             st = 0
             if (trim(cheby_acc) .eq. "schwarz") then
                this%rs_phmg_hrchy%lvl(i)%cheby%schwarz => &
                     this%rs_phmg_hrchy%lvl(i)%schwarz
                st = 2
             end if
          end if
       end if

    end do

    call print_rs_phmg_info(this%nlvls, st, this%rs_phmg_hrchy)

    ! Create backend specific Ax operator. The coarse levels compress their
    ! geometric factors when the fine level does, and keep them in step
    ! with a moving mesh through recompute_metrics()
    if (coef%geo_compression) then
       do i = 1, this%nlvls - 1
          call this%rs_phmg_hrchy%lvl(i)%coef%enable_geo_compression( &
               coef%geo_compression_tol)
       end do
       call ax_helm_allocator(this%ax, type_name = "standard_compr")
    else
       call ax_helm_allocator(this%ax, type_name = "standard")
    end if

    do i = 0, this%nlvls - 1
       this%rs_phmg_hrchy%lvl(i)%cheby%power_its = this%cheby_power_its
    end do
    if (trim(cheby_acc) .eq. "line") then
       do i = 0, this%nlvls - 1
          call this%rs_phmg_hrchy%lvl(i)%line%init_from_components( &
               this%ax, this%rs_phmg_hrchy%lvl(i)%coef, &
               this%rs_phmg_hrchy%lvl(i)%gs_h, &
               this%rs_phmg_hrchy%lvl(i)%bc_projector, &
               this%line_columns, this%line_geometric, this%line_direction, &
               this%line_iso_ratio, this%line_global_columns, this%line_planes, &
               this%line_plane_ratio, trim(this%line_factor_precision) .eq. 'single')
       end do
    end if

    if (this%cheby_lanczos_its .gt. 0) then
       do i = 0, this%nlvls - 1
          if (NEKO_BCKND_DEVICE .eq. 1) then
             call rs_phmg_lanczos_eigs_device(this, this%rs_phmg_hrchy%lvl(i), &
                  this%cheby_lanczos_its, i)
          else
             call rs_phmg_lanczos_eigs(this, this%rs_phmg_hrchy%lvl(i), &
                  this%cheby_lanczos_its, i)
          end if
       end do
    end if

    ! Interpolator Fine + mg levels
    allocate(this%intrp(this%nlvls - 1))
    do i = 1, this%nlvls -1
       call this%intrp(i)%init(this%rs_phmg_hrchy%lvl(i-1)%Xh, &
            this%rs_phmg_hrchy%lvl(i)%Xh)
    end do

    ! Coarse space first. Each level maps from level 0.
    if (this%update_enabled) then
       allocate(this%crd_intrp(this%nlvls - 1))
       do i = 1, this%nlvls - 1
          call this%crd_intrp(i)%init(this%rs_phmg_hrchy%lvl(i)%Xh, &
               this%rs_phmg_hrchy%lvl(0)%Xh)
       end do
    end if

    if (.not. allocated(this%coarse_solver)) this%coarse_solver = 'tamg'
    ! the tree-AMG coarse solver only where it is used (it was built
    ! unconditionally before 2026-10-13)
    if (trim(this%coarse_solver) .eq. 'tamg') then
       call this%amg_solver%init(this%ax, this%rs_phmg_hrchy%lvl(this%nlvls -1)%Xh, &
            this%rs_phmg_hrchy%lvl(this%nlvls -1)%coef, this%msh, &
            this%rs_phmg_hrchy%lvl(this%nlvls-1)%gs_h, crs_tamg_lvls, &
            this%rs_phmg_hrchy%lvl(this%nlvls -1)%bc_projector, &
            crs_tamg_itrs, crs_tamg_cheby_degree, use_jacobi = this%tamg_jacobi, &
            jacobi_probes = this%tamg_jacobi_probes)
    end if
    select case (trim(this%coarse_solver))
    case ('tamg')
    case ('cg')
       if (NEKO_BCKND_DEVICE .eq. 1) then
          call neko_error('rs_phmg: coarse_solver cg is CPU only')
       end if
       associate(c => this%rs_phmg_hrchy%lvl(this%nlvls-1))
         ! the coarse coef is operator-only (no volume); cg_t normalizes
         ! its residual by 1/sqrt(volume), so give it the fine-level volume
         if (c%coef%volume .le. 0.0_rp) then
            c%coef%volume = this%rs_phmg_hrchy%lvl(0)%coef%volume
         end if
         if (.not. allocated(this%coarse_pc)) this%coarse_pc = 'auto'
         ! 'auto' = jacobi: the line-preconditioned inner CG needs ~25 %
         ! fewer inner iterations but costs more time (2026-10-08)
         if (trim(this%coarse_pc) .eq. 'auto') this%coarse_pc = 'jacobi'
         if (trim(this%coarse_pc) .eq. 'line') then
            ! the N = 1 line smoother (columns) as the coarse CG preconditioner
            if (trim(cheby_acc) .ne. 'line') then
               call c%line%init_from_components(this%ax, c%coef, c%gs_h, &
                    c%bc_projector, this%line_columns, this%line_geometric, &
                    this%line_direction, this%line_iso_ratio, this%line_global_columns, &
                    this%line_planes, this%line_plane_ratio, trim(this%line_factor_precision) .eq. 'single')
            end if
            call this%crs_cg%init(c%dm_Xh%size(), this%coarse_max_iter, &
                 M = c%line, abs_tol = 1.0e-30_rp, monitor = .false.)
         else
            call this%crs_jacobi%init_from_components(c%coef, c%dm_Xh, c%gs_h)
            call this%crs_jacobi%update()
            call this%crs_cg%init(c%dm_Xh%size(), this%coarse_max_iter, &
                 M = this%crs_jacobi, abs_tol = 1.0e-30_rp, monitor = .false.)
         end if
         call neko_log%message('rs_phmg: coarse cg preconditioner ' // trim(this%coarse_pc))
       end associate
    case ('colagg')
       if (trim(cheby_acc) .ne. 'line' .or. .not. this%line_columns) then
          call neko_error('rs_phmg: coarse_solver colagg needs smoother_cheby_acc line and line_columns')
       end if
       associate(c => this%rs_phmg_hrchy%lvl(this%nlvls-1))
         if (c%coef%volume .le. 0.0_rp) then
            c%coef%volume = this%rs_phmg_hrchy%lvl(0)%coef%volume
         end if
         if (NEKO_BCKND_DEVICE .eq. 1) then
            call this%colagg%init_from_components(this%ax, c%coef, c%gs_h, &
                 c%bc_projector, c%cheby, this%msh, c%Xh, c%dm_Xh, c%line, &
                 this%colagg_smooth, this%colagg_theta, this%colagg_dense_max, &
                 this%colagg_hierarchy, this%colagg_kappa, c%cheby_device, this%colagg_crs_host, &
                 this%colagg_direct, this%colagg_direct_solver)
         else
            call this%colagg%init_from_components(this%ax, c%coef, c%gs_h, &
                 c%bc_projector, c%cheby, this%msh, c%Xh, c%dm_Xh, c%line, &
                 this%colagg_smooth, this%colagg_theta, this%colagg_dense_max, &
                 this%colagg_hierarchy, this%colagg_kappa, direct = this%colagg_direct, &
                 direct_solver = this%colagg_direct_solver)
         end if
         if (NEKO_BCKND_DEVICE .eq. 1 .and. this%colagg_cycles .le. 0) then
            call neko_error('rs_phmg: on the device coarse_solver colagg needs colagg_cycles > 0')
         end if
         if (this%colagg%merged_fraction .lt. this%colagg_min_merge) then
            ! the aggregation merges (almost) nothing: tamg is cheaper
            if (pe_rank .eq. 0) then
               write(*, '(A,F6.3,A)') 'RS_COLAGG merged fraction ', &
                    this%colagg%merged_fraction, ' below colagg_min_merge: using tamg'
            end if
            call this%colagg%free()
            this%coarse_solver = 'tamg'
            ! built here: the tree-AMG is only built up front when selected
            call this%amg_solver%init(this%ax, c%Xh, c%coef, this%msh, c%gs_h, &
                 crs_tamg_lvls, c%bc_projector, crs_tamg_itrs, crs_tamg_cheby_degree, &
                 use_jacobi = this%tamg_jacobi, jacobi_probes = this%tamg_jacobi_probes)
         else
            call this%crs_cg%init(c%dm_Xh%size(), this%coarse_max_iter, &
                 M = this%colagg, abs_tol = 1.0e-30_rp, monitor = .false.)
         end if
       end associate
    case default
       call neko_error('rs_phmg: unknown coarse_solver ' // trim(this%coarse_solver))
    end select
    call neko_log%message('rs_phmg: coarse solver ' // trim(this%coarse_solver))


    ! update() only refreshes when `lvl(0)%coef%metrics_version` changes.
    this%last_metrics_version = this%rs_phmg_hrchy%lvl(0)%coef%metrics_version

    ! Hand the eigenvalue policy to every smoother.
    if (this%update_enabled) then
       do i = 0, this%nlvls - 1
          this%rs_phmg_hrchy%lvl(i)%cheby%warm_start_eigs = this%eigs_warm_start
          this%rs_phmg_hrchy%lvl(i)%cheby%power_its_refresh = &
               this%power_its_refresh
          this%rs_phmg_hrchy%lvl(i)%cheby_device%warm_start_eigs = &
               this%eigs_warm_start
          this%rs_phmg_hrchy%lvl(i)%cheby_device%power_its_refresh = &
               this%power_its_refresh
       end do
       if (trim(this%coarse_solver) .eq. 'tamg') &
            call this%amg_solver%set_eig_refresh(this%eigs_warm_start, this%power_its_refresh)

       if (trim(cheby_acc) .eq. "schwarz") then
          call neko_warning("PHMG: the Schwarz smoother is not refreshed " // &
               "when the mesh changes. Its local solves stay at the " // &
               "initial geometry.")
       end if
    end if

    call rs_mem_report('rs_phmg before release_host')
    ! GPU runs: the host copies of the line factors and blocks are no
    ! longer needed once the whole hierarchy (incl. rs_colagg) is built
    if (trim(cheby_acc) .eq. "line") then
       do i = 0, this%nlvls - 1
          call this%rs_phmg_hrchy%lvl(i)%line%release_host()
       end do
    end if
    call rs_mem_report('rs_phmg after release_host')

  end subroutine rs_phmg_init_from_components

  subroutine rs_phmg_free(this)
    class(rs_phmg_t), intent(inout) :: this
    integer :: i
    character(len=LOG_SIZE) :: log_buf

    if (this%crs_calls .gt. 0) then
       write(log_buf, '(A,I0,A,I0,A,I0,A)') 'rs_phmg: coarse cg: ', &
            this%crs_iters, ' iterations in ', this%crs_calls, ' calls, ', &
            this%crs_fails, ' not converged'
       call neko_log%message(log_buf)
    end if
    if (pe_rank .eq. 0 .and. allocated(this%rs_phmg_hrchy%lvl)) then
       do i = 0, this%nlvls - 2
          if (NEKO_BCKND_DEVICE .eq. 1) then
             write(*, '(A,I0,A,ES11.4,A,ES11.4)') 'RS_PHMG_CHEBY level ', i, &
                  ' tha ', this%rs_phmg_hrchy%lvl(i)%cheby_device%tha, ' dlt ', &
                  this%rs_phmg_hrchy%lvl(i)%cheby_device%dlt
          else
             write(*, '(A,I0,A,ES11.4,A,ES11.4)') 'RS_PHMG_CHEBY level ', i, &
                  ' tha ', this%rs_phmg_hrchy%lvl(i)%cheby%tha, ' dlt ', &
                  this%rs_phmg_hrchy%lvl(i)%cheby%dlt
          end if
       end do
    end if
    if (allocated(this%rs_phmg_hrchy%lvl)) then
       do i = 0, this%nlvls - 1   ! the coarsest level's line (colagg's smoother) included
          call this%rs_phmg_hrchy%lvl(i)%line%free()
       end do
    end if
    this%crs_calls = 0
    this%crs_iters = 0
    this%crs_fails = 0
    call this%crs_cg%free()
    call this%crs_jacobi%free()
    call this%colagg%free()
    call this%amg_solver%free()

    if (allocated(this%intrp)) then
       do i = 1, size(this%intrp)
          call this%intrp(i)%free()
       end do
       deallocate(this%intrp)
    end if

    if (allocated(this%crd_intrp)) then
       do i = 1, size(this%crd_intrp)
          call this%crd_intrp(i)%free()
       end do
       deallocate(this%crd_intrp)
    end if

    if (allocated(this%ax)) then
       call this%ax%free()
       deallocate(this%ax)
    end if

    if (allocated(this%rs_phmg_hrchy%lvl)) then
       do i = lbound(this%rs_phmg_hrchy%lvl, 1), ubound(this%rs_phmg_hrchy%lvl, 1)
          call this%rs_phmg_hrchy%lvl(i)%r%free()
          call this%rs_phmg_hrchy%lvl(i)%w%free()
          call this%rs_phmg_hrchy%lvl(i)%z%free()

          call this%rs_phmg_hrchy%lvl(i)%cheby%free()
          call this%rs_phmg_hrchy%lvl(i)%cheby_device%free()
          call this%rs_phmg_hrchy%lvl(i)%jacobi%free()
          call this%rs_phmg_hrchy%lvl(i)%device_jacobi%free()

          if (allocated(this%rs_phmg_hrchy%lvl(i)%schwarz%work1)) then
             call this%rs_phmg_hrchy%lvl(i)%schwarz%free()
          end if

          call this%rs_phmg_hrchy%lvl(i)%bc_projector%free()
          call this%rs_phmg_hrchy%lvl(i)%bc%free()

          ! Level 0 borrows Xh, dm_Xh, gs_h and coef from the caller,
          ! all other levels own them
          if (this%rs_phmg_hrchy%lvl(i)%lvl .gt. 0) then
             call this%rs_phmg_hrchy%lvl(i)%coef%free()
             call this%rs_phmg_hrchy%lvl(i)%gs_h%free()
             call this%rs_phmg_hrchy%lvl(i)%dm_Xh%free()
             call this%rs_phmg_hrchy%lvl(i)%Xh%free()
             deallocate(this%rs_phmg_hrchy%lvl(i)%coef)
             deallocate(this%rs_phmg_hrchy%lvl(i)%gs_h)
             deallocate(this%rs_phmg_hrchy%lvl(i)%dm_Xh)
             deallocate(this%rs_phmg_hrchy%lvl(i)%Xh)
          end if

          nullify(this%rs_phmg_hrchy%lvl(i)%coef)
          nullify(this%rs_phmg_hrchy%lvl(i)%gs_h)
          nullify(this%rs_phmg_hrchy%lvl(i)%dm_Xh)
          nullify(this%rs_phmg_hrchy%lvl(i)%Xh)
       end do
       deallocate(this%rs_phmg_hrchy%lvl)
    end if

    nullify(this%msh)

  end subroutine rs_phmg_free

  subroutine rs_phmg_solve(this, z, r, n)
    class(rs_phmg_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r

    if (.not. this%symcheck_done) then
       this%symcheck_done = .true.
       block
         character(len=8) :: env
         integer :: istat
         call get_environment_variable('RS_PHMG_SYMCHECK', env, status = istat)
         if (istat .eq. 0) call rs_phmg_symcheck(this, n)
       end block
    end if
    call rs_phmg_apply(this, z, r, n)
  end subroutine rs_phmg_solve

  !> Symmetry test (env RS_PHMG_SYMCHECK): CG needs (u, M v) = (v, M u) in
  !! the mult-weighted inner product. Two random continuous vectors with the
  !! Dirichlet nodes masked; prints the relative asymmetry
  !! |(u,Mv) - (v,Mu)| / sqrt((u,Mu)(v,Mv)) and the Rayleigh quotients.
  subroutine rs_phmg_symcheck(this, n)
    class(rs_phmg_t), intent(inout) :: this
    integer, intent(in) :: n
    type(field_t) :: u, v, mu, mv
    real(kind=rp) :: a, b, c, d, uu, vv
    associate(mg0 => this%rs_phmg_hrchy%lvl(0))
      call u%init(mg0%dm_Xh, 'rs_symu'); call v%init(mg0%dm_Xh, 'rs_symv')
      call mu%init(mg0%dm_Xh, 'rs_symmu'); call mv%init(mg0%dm_Xh, 'rs_symmv')
      call random_number(u%x); call random_number(v%x)
      u%x = u%x - 0.5_rp; v%x = v%x - 0.5_rp
      if (NEKO_BCKND_DEVICE .eq. 1) then
         call device_memcpy(u%x, u%x_d, n, HOST_TO_DEVICE, sync = .true.)
         call device_memcpy(v%x, v%x_d, n, HOST_TO_DEVICE, sync = .true.)
      end if
      call mg0%gs_h%op(u%x, n, GS_OP_ADD)
      call mg0%gs_h%op(v%x, n, GS_OP_ADD)
      if (NEKO_BCKND_DEVICE .eq. 1) then
         call device_col2(u%x_d, mg0%coef%mult_d, n)
         call device_col2(v%x_d, mg0%coef%mult_d, n)
      else
         call col2(u%x, mg0%coef%mult, n)
         call col2(v%x, mg0%coef%mult, n)
      end if
      call mg0%bc_projector%apply(u%x, n)
      call mg0%bc_projector%apply(v%x, n)
      call rs_phmg_apply(this, mu%x, u%x, n)
      call rs_phmg_apply(this, mv%x, v%x, n)
      if (NEKO_BCKND_DEVICE .eq. 1) then
         a = device_glsc3(u%x_d, mg0%coef%mult_d, mv%x_d, n)
         b = device_glsc3(v%x_d, mg0%coef%mult_d, mu%x_d, n)
         c = device_glsc3(u%x_d, mg0%coef%mult_d, mu%x_d, n)
         d = device_glsc3(v%x_d, mg0%coef%mult_d, mv%x_d, n)
         uu = device_glsc3(u%x_d, mg0%coef%mult_d, u%x_d, n)
         vv = device_glsc3(v%x_d, mg0%coef%mult_d, v%x_d, n)
      else
         a = glsc3(u%x, mg0%coef%mult, mv%x, n)
         b = glsc3(v%x, mg0%coef%mult, mu%x, n)
         c = glsc3(u%x, mg0%coef%mult, mu%x, n)
         d = glsc3(v%x, mg0%coef%mult, mv%x, n)
         uu = glsc3(u%x, mg0%coef%mult, u%x, n)
         vv = glsc3(v%x, mg0%coef%mult, v%x, n)
      end if
      if (pe_rank .eq. 0) write(*, '(A,ES10.3,A,2ES12.4,A,2ES12.4)') &
           'RS_PHMG symcheck: rel. asymmetry ', abs(a - b) / sqrt(abs(c * d)), &
           '  (u,Mv) (v,Mu) ', a, b, '  Rayleigh (u,Mu)/(u,u) (v,Mv)/(v,v) ', c / uu, d / vv
      call u%free(); call v%free(); call mu%free(); call mv%free()
    end associate
  end subroutine rs_phmg_symcheck

  subroutine rs_phmg_apply(this, z, r, n)
    class(rs_phmg_t), intent(inout) :: this
    integer, intent(in) :: n
    real(kind=rp), dimension(n), intent(inout) :: z
    real(kind=rp), dimension(n), intent(inout) :: r
    type(c_ptr) :: z_d, r_d
    type(ksp_monitor_t) :: ksp_results
    integer :: i

    call profiler_start_region('PHMG_solve', 8)
    associate( mglvl => this%rs_phmg_hrchy%lvl)
      if (NEKO_BCKND_DEVICE .eq. 1) then
         z_d = device_get_ptr(z)
         r_d = device_get_ptr(r)
         !We should not work with the input
         call device_copy(mglvl(0)%r%x_d, r_d, n)

         call device_rzero(mglvl(0)%z%x_d, n)
         call device_rzero(mglvl(0)%w%x_d, n)

         call this%mg_cycle()

         call device_copy(z_d, mglvl(0)%z%x_d, n)
      else
         !OCL NORECURRENCE, NOVREC, NOALIAS
         !DIR$ CONCURRENT
         !DIR$ IVDEP
         !GCC$ ivdep
         !$omp parallel do
         do i = 1, n
            !We should not work with the input
            mglvl(0)%r%x(i,1,1,1) = r(i)

            mglvl(0)%z%x(i,1,1,1) = 0.0_rp
            mglvl(0)%w%x(i,1,1,1) = 0.0_rp
         end do
         !$omp end parallel do

         call this%mg_cycle()

         !OCL NORECURRENCE, NOVREC, NOALIAS
         !DIR$ CONCURRENT
         !DIR$ IVDEP
         !GCC$ ivdep
         !$omp parallel do
         do i = 1, n
            z(i) = mglvl(0)%z%x(i,1,1,1)
         end do
         !$omp end parallel do
      end if
    end associate
    call profiler_end_region('PHMG_solve', 8)

  end subroutine rs_phmg_apply

  !> Bring the preconditioner back in sync after the mesh has changed.
  subroutine rs_phmg_update(this)
    class(rs_phmg_t), intent(inout) :: this
    integer :: fine_version
    logical :: do_eigs

    if (.not. this%update_enabled) return

    fine_version = this%rs_phmg_hrchy%lvl(0)%coef%metrics_version

    ! Mesh has not changed since the last refresh.
    if (fine_version .eq. this%last_metrics_version) return

    call rs_phmg_update_coarse_geometry(this)

    call rs_phmg_update_smoother_acc(this)

    do_eigs = (this%refresh_eigs) .and. &
         (this%refresh_eigs_frequency .gt. 0) .and. &
         (mod(this%n_refresh, max(this%refresh_eigs_frequency, 1)) &
         .eq. 0)
    if (do_eigs) call rs_phmg_update_smoother_eigs(this)

    this%n_refresh = this%n_refresh + 1
    this%last_metrics_version = fine_version

  end subroutine rs_phmg_update


  !> Sample the new fine coordinates on the coarse levels and rebuild their
  !! metrics. Level 0 shares the caller's dofmap and coef, so it is skipped.
  subroutine rs_phmg_update_coarse_geometry(this)
    class(rs_phmg_t), intent(inout) :: this
    integer :: i

    if (.not. allocated(this%crd_intrp)) then
       call neko_error("PHMG: update requested but coordinate " // &
            "interpolators were not initialized.")
    end if

    call profiler_start_region('PHMG_update_geometry')
    associate (mg => this%rs_phmg_hrchy%lvl, nelv => this%msh%nelv)
      do i = 1, this%nlvls - 1
         call this%crd_intrp(i)%map(mg(i)%dm_Xh%x%x, mg(0)%dm_Xh%x%x, &
              nelv, mg(i)%Xh)
         call this%crd_intrp(i)%map(mg(i)%dm_Xh%y%x, mg(0)%dm_Xh%y%x, &
              nelv, mg(i)%Xh)
         call this%crd_intrp(i)%map(mg(i)%dm_Xh%z%x, mg(0)%dm_Xh%z%x, &
              nelv, mg(i)%Xh)

         call mg(i)%coef%recompute_metrics()
      end do
    end associate
    call profiler_end_region('PHMG_update_geometry')

  end subroutine rs_phmg_update_coarse_geometry


  !> Rebuild the accelerator the Chebyshev smoother uses, which depends on
  !! the geometry.
  subroutine rs_phmg_update_smoother_acc(this)
    class(rs_phmg_t), intent(inout) :: this
    integer :: i

    call profiler_start_region('PHMG_update_smoother_acc')
    associate (mg => this%rs_phmg_hrchy%lvl)
      select case (trim(this%cheby_acc))
      case ("jacobi")
         do i = 0, this%nlvls - 1
            if (NEKO_BCKND_DEVICE .eq. 1) then
               call mg(i)%device_jacobi%update()
            else
               call mg(i)%jacobi%update()
            end if
         end do
      case ("schwarz")
         ! Not refreshed. The local solves are built by the FDM from the
         ! element coordinates, and fdm_t has no update path.
      case default
         ! No accelerator to refresh.
      end select
    end associate
    call profiler_end_region('PHMG_update_smoother_acc')

  end subroutine rs_phmg_update_smoother_acc


  !> Mark every smoother to re-estimate its eigenvalues on the next solve.
  subroutine rs_phmg_update_smoother_eigs(this)
    class(rs_phmg_t), intent(inout) :: this
    integer :: i

    associate (mg => this%rs_phmg_hrchy%lvl)
      do i = 0, this%nlvls - 1
         if (NEKO_BCKND_DEVICE .eq. 1) then
            mg(i)%cheby_device%recompute_eigs = .true.
         else
            mg(i)%cheby%recompute_eigs = .true.
         end if
      end do
    end associate

    if (trim(this%coarse_solver) .eq. 'tamg') call this%amg_solver%invalidate_eigs()

  end subroutine rs_phmg_update_smoother_eigs


  subroutine rs_phmg_mg_cycle(this)
    class(rs_phmg_t), intent(inout) :: this
    type(ksp_monitor_t) :: ksp_results
    character(len=2) :: lvl_name
    integer :: lvl, i

    associate(mg => this%rs_phmg_hrchy%lvl, intrp => this%intrp, &
         msh => this%msh, Ax => this%Ax)
      do lvl = 0, this%nlvls-2
         write(lvl_name, '(I0)') lvl
         call profiler_start_region( "PHMG_level_" // trim(lvl_name))
         associate(z => mg(lvl)%z, r => mg(lvl)%r, w => mg(lvl)%w)
           !------------!
           !   SMOOTH   !
           !------------!
           if (NEKO_BCKND_DEVICE .eq. 1) then
              mg(lvl)%cheby_device%zero_initial_guess = .true.
              ksp_results = mg(lvl)%cheby_device%solve(Ax, z, &
                   r%x, mg(lvl)%dm_Xh%size(), &
                   mg(lvl)%coef, mg(lvl)%bc_projector, &
                   mg(lvl)%gs_h, niter = mg(lvl)%smoother_itrs)
           else
              mg(lvl)%cheby%zero_initial_guess = .true.
              ksp_results = mg(lvl)%cheby%solve(Ax, z, &
                   r%x, mg(lvl)%dm_Xh%size(), &
                   mg(lvl)%coef, mg(lvl)%bc_projector, &
                   mg(lvl)%gs_h, niter = mg(lvl)%smoother_itrs)
           end if

           !------------!
           !  Residual  !
           !------------!
           call Ax%compute(w%x, z%x, mg(lvl)%coef, msh, mg(lvl)%Xh)
           call mg(lvl)%gs_h%op(w%x, mg(lvl)%dm_Xh%size(), GS_OP_ADD, &
                glb_cmd_event)
           call device_stream_wait_event(glb_cmd_queue, glb_cmd_event, 0)
           call mg(lvl)%bc_projector%apply(w%x, mg(lvl)%dm_Xh%size())

           if (NEKO_BCKND_DEVICE .eq. 1) then
              call device_add2s1(w%x_d, r%x_d, -1.0_rp, mg(lvl)%dm_Xh%size())
           else
              !OCL NORECURRENCE, NOVREC, NOALIAS
              !DIR$ CONCURRENT
              !DIR$ IVDEP
              !GCC$ ivdep
              !$omp parallel do
              do i = 1, mg(lvl)%dm_Xh%size()
                 w%x(i,1,1,1) = r%x(i,1,1,1) - w%x(i,1,1,1)
              end do
              !$omp end parallel do
           end if

           !------------!
           !  Restrict  !
           !------------!
           if (NEKO_BCKND_DEVICE .eq. 1) then
              call device_col2(w%x_d, mg(lvl)%coef%mult_d, mg(lvl)%dm_Xh%size())
           else
              call col2(w%x, mg(lvl)%coef%mult, mg(lvl)%dm_Xh%size())
           end if

           call intrp(lvl+1)%map(mg(lvl+1)%r%x, w%x, msh%nelv, mg(lvl+1)%Xh)

           call mg(lvl+1)%gs_h%op(mg(lvl+1)%r%x, mg(lvl+1)%dm_Xh%size(), &
                GS_OP_ADD, glb_cmd_event)
           call device_stream_wait_event(glb_cmd_queue, glb_cmd_event, 0)

           call mg(lvl+1)%bc_projector%apply( &
                mg(lvl+1)%r%x, &
                mg(lvl+1)%dm_Xh%size())

           if (NEKO_BCKND_DEVICE .eq. 1) then
              call device_rzero(mg(lvl+1)%z%x_d, mg(lvl+1)%dm_Xh%size())
           else
              !OCL NORECURRENCE, NOVREC, NOALIAS
              !DIR$ CONCURRENT
              !DIR$ IVDEP
              !GCC$ ivdep
              !$omp parallel do
              do i = 1, mg(lvl+1)%dm_Xh%size()
                 mg(lvl+1)%z%x(i,1,1,1) = 0.0_rp
              end do
              !$omp end parallel do
           end if
         end associate
         call profiler_end_region( "PHMG_level_" // trim(lvl_name))
      end do

      call profiler_start_region( 'PHMG_coarse-solve' )
      !------------!
      !   SOLVE    !
      !------------!
      if (trim(this%coarse_solver) .eq. 'cg' .or. &
           trim(this%coarse_solver) .eq. 'colagg') then
         call rs_phmg_coarse_cg(this, mg(this%nlvls-1))
      else
         call this%amg_solver%solve(mg(this%nlvls-1)%z%x, &
              mg(this%nlvls-1)%r%x, &
              mg(this%nlvls-1)%dm_Xh%size())
      end if
      call profiler_end_region( 'PHMG_coarse-solve' )

      do lvl = (this%nlvls-2), 0, -1
         write(lvl_name, '(I0)') lvl
         call profiler_start_region( "PHMG_level_" // trim(lvl_name))
         associate(z => mg(lvl)%z, r => mg(lvl)%r, w => mg(lvl)%w)
           !------------!
           !  Project   !
           !------------!
           call intrp(lvl+1)%map(w%x, mg(lvl+1)%z%x, msh%nelv, mg(lvl)%Xh)

           call mg(lvl)%gs_h%op(w%x, mg(lvl)%dm_Xh%size(), GS_OP_ADD, &
                glb_cmd_event)
           call device_stream_wait_event(glb_cmd_queue, glb_cmd_event, 0)

           if (NEKO_BCKND_DEVICE .eq. 1) then
              call device_col2(w%x_d, mg(lvl)%coef%mult_d, mg(lvl)%dm_Xh%size())
           else
              call col2(w%x, mg(lvl)%coef%mult, mg(lvl)%dm_Xh%size())
           end if

           !------------!
           !  Correct   !
           !------------!
           if (NEKO_BCKND_DEVICE .eq. 1) then
              call device_add2(z%x_d, w%x_d, mg(lvl)%dm_Xh%size())
           else
              !OCL NORECURRENCE, NOVREC, NOALIAS
              !DIR$ CONCURRENT
              !DIR$ IVDEP
              !GCC$ ivdep
              !$omp parallel do
              do i = 1, mg(lvl)%dm_Xh%size()
                 z%x(i,1,1,1) = z%x(i,1,1,1) + w%x(i,1,1,1)
              end do
              !$omp end parallel do
           end if

           !------------!
           !   SMOOTH   !
           !------------!
           if (NEKO_BCKND_DEVICE .eq. 1) then
              ksp_results = mg(lvl)%cheby_device%solve(Ax, z, &
                   r%x, mg(lvl)%dm_Xh%size(), &
                   mg(lvl)%coef, mg(lvl)%bc_projector, &
                   mg(lvl)%gs_h, niter = mg(lvl)%smoother_itrs)
           else
              ksp_results = mg(lvl)%cheby%solve(Ax, z, &
                   r%x, mg(lvl)%dm_Xh%size(), &
                   mg(lvl)%coef, mg(lvl)%bc_projector, &
                   mg(lvl)%gs_h, niter = mg(lvl)%smoother_itrs)
           end if
         end associate
         call profiler_end_region( "PHMG_level_" // trim(lvl_name))
      end do
    end associate

  end subroutine rs_phmg_mg_cycle

  subroutine print_rs_phmg_info(nlvls, smoo_type, rs_phmg)
    integer, intent(in) :: nlvls
    integer, intent(in) :: smoo_type
    type(rs_phmg_hrchy_t) :: rs_phmg
    integer :: i, clvl
    character(len=LOG_SIZE) :: log_buf, smoo_name

    call neko_log%section('PHMG')

    if (smoo_type .eq. 1) then
       write(smoo_name, '(A16)') 'CHEBY-acc JACOBI'
    else if (smoo_type .eq. 2) then
       write(smoo_name, '(A17)') 'CHEBY-acc SCHWARZ'
    else
       write(smoo_name, '(A5)') 'CHEBY'
    end if

    write(log_buf, '(A28,I2,A8)') &
         'Creating PHMG hierarchy with', &
         nlvls, 'levels.'
    call neko_log%message(log_buf)

    clvl = nlvls - 1
    do i = 0, nlvls-1
       write(log_buf, '(A8,I2,A8,I2)') &
            '-- level', i, '-- lx:', rs_phmg%lvl(i)%Xh%lx
       call neko_log%message(log_buf)

       if (i .eq. clvl) then
          write(log_buf, '(A19,A20)') &
               'Solve:', 'tAMG'
          call neko_log%message(log_buf)
       else
          write(log_buf, '(A22,A20)') &
               'Smoother:', &
               trim(smoo_name)
          call neko_log%message(log_buf)

          write(log_buf, '(A28,I2)') &
               'Smoother Iters:', &
               rs_phmg%lvl(i)%smoother_itrs
          call neko_log%message(log_buf)
       end if
    end do

    call neko_log%end_section()

  end subroutine print_rs_phmg_info

  !> 'Exact' coarse solve: CG + Jacobi on the N = 1 level to coarse_tol
  !! relative to the coarse residual, zero initial guess (like tamg).
  subroutine rs_phmg_coarse_cg(this, c)
    class(rs_phmg_t), intent(inout) :: this
    type(rs_phmg_lvl_t), intent(inout) :: c
    type(ksp_monitor_t) :: res
    real(kind=rp) :: rn
    integer :: n
    character(len=LOG_SIZE) :: log_buf

    n = c%dm_Xh%size()
    if (trim(this%coarse_solver) .eq. 'colagg' .and. this%colagg_cycles .gt. 0) then
       ! fixed number of symmetric two-level cycles (a linear SPD preconditioner)
       call this%colagg%cycles(c%z%x, c%r%x, n, this%colagg_cycles)
       this%crs_calls = this%crs_calls + 1
       this%crs_iters = this%crs_iters + this%colagg_cycles
       return
    end if
    rn = sqrt(glsc3(c%r%x, c%coef%mult, c%r%x, n)) / sqrt(c%coef%volume)
    this%crs_cg%abs_tol = max(this%coarse_tol * rn, 1.0e-30_rp)
    res = this%crs_cg%solve(this%ax, c%z, c%r%x, n, c%coef, &
         c%bc_projector, c%gs_h)
    this%crs_calls = this%crs_calls + 1
    this%crs_iters = this%crs_iters + res%iter
    if (.not. res%converged) this%crs_fails = this%crs_fails + 1
  end subroutine rs_phmg_coarse_cg

  !> Chebyshev interval of a level from the largest Ritz value of a short
  !! PCG-Lanczos run on M^{-1}A (M = the accelerator), with Neko's safety
  !! factors (boost 1.1, lambda_factor 30), instead of the 150-step power
  !! iteration (which costs 150 accelerator applications per level).
  subroutine rs_phmg_lanczos_eigs(this, l, its, ilvl)
    class(rs_phmg_t), intent(inout) :: this
    type(rs_phmg_lvl_t), intent(inout) :: l
    integer, intent(in) :: its, ilvl
    real(kind=rp), allocatable :: r(:), z(:), p(:), q(:), al(:), be(:)
    real(kind=dp), allocatable :: d(:), e(:), work(:)   ! dstev: double
    real(kind=dp) :: dummy(1, 1)
    integer :: n, i, j, m, info
    integer(kind=i8) :: seed
    real(kind=rp) :: rz, rz_new, pq, lam, a, b, rz0

    if (associated(l%cheby%schwarz)) return
    n = l%dm_Xh%size()
    allocate(r(n), z(n), p(n), q(n), al(its), be(its), d(its), e(its), work(max(1, 2 * its)))
    seed = 7919_i8 + int(pe_rank, i8)
    do i = 1, n
       seed = mod(seed * 6364136223846793005_i8 + 1442695040888963407_i8, 2_i8**62)
       r(i) = real(mod(seed, 1000000_i8), rp) / 1.0e6_rp - 0.5_rp
    end do
    call l%gs_h%op(r, n, GS_OP_ADD)
    call l%bc_projector%apply(r, n)
    call l%cheby%M%solve(z, r, n)
    call copy(p, z, n)
    rz = glsc3(r, l%coef%mult, z, n)
    rz0 = rz
    m = 0
    be = 0.0_rp
    do j = 1, its
       call this%ax%compute(q, p, l%coef, this%msh, l%Xh)
       call l%gs_h%op(q, n, GS_OP_ADD)
       call l%bc_projector%apply(q, n)
       pq = glsc3(p, l%coef%mult, q, n)
       if (pq .le. 0.0_rp) exit
       al(j) = rz / pq
       m = j
       do i = 1, n
          r(i) = r(i) - al(j) * q(i)
       end do
       call l%cheby%M%solve(z, r, n)
       rz_new = glsc3(r, l%coef%mult, z, n)
       if (rz_new .le. 1.0e-28_rp * rz0) exit
       be(j) = rz_new / rz
       rz = rz_new
       do i = 1, n
          p(i) = z(i) + be(j) * p(i)
       end do
    end do
    if (m .lt. 1) return
    do j = 1, m
       d(j) = 1.0_rp / al(j)
       if (j .gt. 1) d(j) = d(j) + be(j-1) / al(j-1)
       if (j .lt. m) e(j) = -sqrt(be(j)) / al(j)
    end do
    call dstev('N', m, d, e, dummy, 1, work, info)
    if (info .ne. 0) call neko_error('rs_phmg: dstev failed in the Lanczos estimate')
    lam = d(m)
    b = 1.1_rp * lam
    a = lam / 30.0_rp
    l%cheby%tha = (b + a) / 2.0_rp
    l%cheby%dlt = (b - a) / 2.0_rp
    l%cheby%recompute_eigs = .false.
    l%cheby%eigs_computed = .true.
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,I0,A,ES12.5)') 'RS_PHMG level ', ilvl, ' lanczos steps ', m, &
            ' lambda_max estimate ', lam
    end if
    deallocate(r, z, p, q, al, be, d, e, work)
  end subroutine rs_phmg_lanczos_eigs

  !> Device version of the Lanczos estimate: the vectors live on the
  !! device, inner products by device_glsc3, the accelerator's device solve.
  subroutine rs_phmg_lanczos_eigs_device(this, l, its, ilvl)
    class(rs_phmg_t), intent(inout) :: this
    type(rs_phmg_lvl_t), intent(inout) :: l
    integer, intent(in) :: its, ilvl
    real(kind=rp), allocatable :: r(:), z(:), p(:), q(:), al(:), be(:)
    real(kind=dp), allocatable :: d(:), e(:), work(:)   ! dstev: double
    real(kind=dp) :: dummy(1, 1)
    type(c_ptr) :: r_d, z_d, p_d, q_d
    integer :: n, i, j, m, info
    integer(kind=i8) :: seed
    real(kind=rp) :: rz, rz_new, pq, lam, a, b, rz0

    if (associated(l%cheby_device%schwarz)) return
    n = l%dm_Xh%size()
    allocate(r(n), z(n), p(n), q(n), al(its), be(its), d(its), e(its), work(max(1, 2 * its)))
    seed = 7919_i8 + int(pe_rank, i8)
    do i = 1, n
       seed = mod(seed * 6364136223846793005_i8 + 1442695040888963407_i8, 2_i8**62)
       r(i) = real(mod(seed, 1000000_i8), rp) / 1.0e6_rp - 0.5_rp
    end do
    z = 0.0_rp; p = 0.0_rp; q = 0.0_rp
    r_d = c_null_ptr; z_d = c_null_ptr; p_d = c_null_ptr; q_d = c_null_ptr
    call device_map(r, r_d, n); call device_map(z, z_d, n)
    call device_map(p, p_d, n); call device_map(q, q_d, n)
    call device_memcpy(r, r_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_memcpy(z, z_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_memcpy(p, p_d, n, HOST_TO_DEVICE, sync = .true.)
    call device_memcpy(q, q_d, n, HOST_TO_DEVICE, sync = .true.)
    call l%gs_h%op(r, n, GS_OP_ADD)
    call l%bc_projector%apply(r, n)
    call l%cheby_device%M%solve(z, r, n)
    call device_copy(p_d, z_d, n)
    rz = device_glsc3(r_d, l%coef%mult_d, z_d, n)
    rz0 = rz
    m = 0
    be = 0.0_rp
    do j = 1, its
       call this%ax%compute(q, p, l%coef, this%msh, l%Xh)
       call l%gs_h%op(q, n, GS_OP_ADD)
       call l%bc_projector%apply(q, n)
       pq = device_glsc3(p_d, l%coef%mult_d, q_d, n)
       if (pq .le. 0.0_rp) exit
       al(j) = rz / pq
       m = j
       call device_add2s2(r_d, q_d, -al(j), n)
       call l%cheby_device%M%solve(z, r, n)
       rz_new = device_glsc3(r_d, l%coef%mult_d, z_d, n)
       if (rz_new .le. 1.0e-28_rp * rz0) exit
       be(j) = rz_new / rz
       rz = rz_new
       call device_add2s1(p_d, z_d, be(j), n)
    end do
    call device_deassociate(r); call device_free(r_d)
    call device_deassociate(z); call device_free(z_d)
    call device_deassociate(p); call device_free(p_d)
    call device_deassociate(q); call device_free(q_d)
    if (m .lt. 1) return
    do j = 1, m
       d(j) = 1.0_rp / al(j)
       if (j .gt. 1) d(j) = d(j) + be(j-1) / al(j-1)
       if (j .lt. m) e(j) = -sqrt(be(j)) / al(j)
    end do
    call dstev('N', m, d, e, dummy, 1, work, info)
    if (info .ne. 0) call neko_error('rs_phmg: dstev failed in the Lanczos estimate')
    lam = d(m)
    b = 1.1_rp * lam
    a = lam / 30.0_rp
    l%cheby_device%tha = (b + a) / 2.0_rp
    l%cheby_device%dlt = (b - a) / 2.0_rp
    l%cheby_device%recompute_eigs = .false.
    l%cheby_device%eigs_computed = .true.
    if (pe_rank .eq. 0) then
       write(*, '(A,I0,A,I0,A,ES12.5)') 'RS_PHMG level ', ilvl, ' lanczos (device) steps ', m, &
            ' lambda_max estimate ', lam
    end if
    deallocate(r, z, p, q, al, be, d, e, work)
  end subroutine rs_phmg_lanczos_eigs_device

  subroutine rs_phmg_register_types()
    procedure(precon_allocate), pointer :: allocator

    allocator => rs_phmg_allocate
    call register_precon('rs_phmg', allocator)
  end subroutine rs_phmg_register_types

  subroutine rs_phmg_allocate(obj)
    class(pc_t), allocatable, intent(inout) :: obj

    allocate(rs_phmg_t :: obj)
  end subroutine rs_phmg_allocate

end module rs_phmg
