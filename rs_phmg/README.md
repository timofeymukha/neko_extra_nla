# rs_phmg: robust p-multigrid for the pressure on bad meshes

`rs_phmg` is a preconditioner for the pressure Poisson problem on **meshes where Neko's `phmg`/`hsmg` stall**:
- very high aspect ratios, e.g. wall-resolved boundary layers or pancake elements;
- deep boundary-layer stacks;
- the start-up transients on such meshes.

It works on any hexahedral mesh and needs no mesh information beyond what Neko has: no frames, orderings or hand-set directions. Every choice is made algebraically.

## Method

The structure is Neko's hybrid ph-multigrid: p-levels from `pcoarsening_schedule` (default $N \to 3 \to 1$), Chebyshev smoothing on each p-level, and a coarse solve at $N = 1$. Two components are replaced.

**1. Line (column) smoother as the Chebyshev accelerator** (`smoother_cheby_acc: "line"`, module `rs_line`).
- **Stiff direction.** For each element, the stiff reference direction is the one with the largest off-diagonal coupling of the element-local operator, measured with comb vectors.
- **Columns.** The GLL lines along that direction are chained across element faces into columns through whole stacks of elements (`line_columns`). With `line_global_columns` the chains continue across ranks, made exact by a Schur-complement exchange.
- **Solves.** Each column is a banded system of the *assembled* operator, factored once at setup.
- **Application:** $z = \sum_{\text{lines}} R^{\mathsf T} W^{1/2} A_{\text{line}}^{-1} W^{1/2} R\, r$. This is symmetric, and it reduces to Jacobi for point blocks.
- This removes the stiff direction that defeats point smoothers at high aspect ratios.

**2. Aggregation coarse level at $N = 1$** (`coarse_solver: "colagg"`, module `rs_colagg`).
- **Merging.** Consecutive $N = 1$ nodes along the column lines are merged where the line coupling dominates the lateral couplings, with a length cap $\kappa$ (`colagg_kappa`). Isotropic regions stay unaggregated, so the coarse problem is exact there.
- **Coarse operator:** $A_c = P^{\mathsf T} A P$, with $P$ piecewise constant on the aggregates.
- **Coarse solve:** solved by the own sparse Cholesky `rs_spchol` (`colagg_direct`), or by CG.
- **One application** is a symmetric V(1,1) cycle: line-Chebyshev pre-smoothing, the coarse correction, post-smoothing.
- **Fallback:** if less than `colagg_min_merge` of the nodes merge (an isotropic mesh), the tree-AMG of `phmgx` is used instead.

**Cost.** About 2.5× `phmgx` per iteration (6 line solves and an aggregation cycle per application). It pays off when it saves iterations, i.e. where `phmg` stalls.

## Compatibility

- **Equation:** the pressure, or any SPD Helmholtz problem. It is a preconditioner: use it inside `cg`, `gmres` or `rs_cgmrs`.
- **Mesh:** any hexahedral mesh, curved allowed. GPU kernels need **$l_x \le 8$** ($N \le 7$).
- **Boundary conditions:** any of Neko's. A pure-Neumann pressure is handled by pinning one aggregate.
- **Not rebuilt when the mesh moves** (`update.enabled`).
- **Backends:**
  - **CPU;**
  - **CUDA:** H100, Helma;
  - **HIP:** MI300A, single precision; `rs_line_hip.cpp`.
- **Precision:**
  - double and single-precision Neko;
  - the line and coarse data stay in double, and Neko's vectors are converted where they enter;
  - single-precision builds need `-DRS_REAL_SP` for the kernels, which `build.sh` sets.
- **Memory:** the line factors and the aggregation level are resident.
  - On the foil (340 k elements, N = 7, 4 H100, dp): 92.7 GB per GPU against 84.7 GB for `phmgx`.
  - The host setup needs up to about 35 GB per rank at 85 k elements per rank. A first run on a new case should leave headroom.

## Build

**`build.sh` is required.**
- On GPUs, makeneko cannot link cuBLAS/cuSOLVER or rocBLAS/rocSOLVER.
- On the CPU, its driver makes a stub name ambiguous.
- `rs_phmg` also compiles the tree-AMG modules from `../phmgx/`.

```sh
export PKG_CONFIG_PATH=<neko>/lib/pkgconfig
../build.sh -o neko_rs rs_phmg cgmrs        # rs_cgmrs is the recommended Krylov solver
CUDA_ARCH=-arch=sm_90 ../build.sh -o neko_rs rs_phmg cgmrs            # CUDA Neko
HIP_ARCH=--offload-arch=gfx942 ../build.sh -o neko_rs rs_phmg cgmrs   # HIP Neko
```

What `build.sh` does, by hand (CUDA):

```sh
nvcc -O3 -arch=sm_90 [-DRS_REAL_SP] -Xcompiler -fPIC -c rs_line_cuda.cu
$(pkg-config --variable=compiler neko) -O3 $(pkg-config --cflags neko) \
  rs_spchol.f90 rs_line.f90 rs_colagg.f90 ../phmgx/tamgx_smoother.f90 ../phmgx/tamgx_multigrid.f90 \
  rs_phmg.f90 usr_driver.f90 rs_line_cuda.o -o neko_rs \
  $(pkg-config --libs neko) -lcublas -lcusolver -lcudart -lstdc++ -ldl
```

- `usr_driver.f90` calls `rs_phmg_register_types()` before `neko_init`.
- **CPU:** replace `rs_line_cuda.o` with `rs_line_cuda_stub.f90`, placed first.
- **HIP:** compile `hipcc -O3 --offload-arch=gfx942 -fPIC -c rs_line_hip.cpp` and link `-lamdhip64 -lrocblas -lrocsolver -lstdc++`.

**Change against the source (robust_solvers 00b2b3d):** `rs_line_hip.cpp` now defines `cudaMalloc`/`cudaFree`. The HIP macro layer of `rs_line_cuda.cu` lacks them, and the HIP build fails without this. It is the same fix used for the MI300A runs on 2026-10-14.

## Case file

The recommended settings, used for all foil runs:

```json
"pressure_solver": {
  "type": "rs_cgmrs",
  "preconditioner": {
    "type": "rs_phmg",
    "pcoarsening_schedule": [3, 1],
    "smoother_iterations": 3,
    "coarse_grid": {"levels": 3, "iterations": 1, "cheby_degree": 4},
    "smoother_cheby_acc": "line",
    "line_columns": true,
    "coarse_solver": "colagg",
    "colagg_cycles": 1,
    "colagg_hierarchy": true,
    "colagg_crs_host": true,
    "colagg_dense_max": 0,
    "colagg_direct": true,
    "line_factor_precision": "single"
  },
  "projection_space_size": 0,
  "absolute_tolerance": 1e-4,
  "max_iterations": 250
}
```

- **Krylov solver:**
  - `"gmres"` works too, with slightly fewer iterations and about 5× the Krylov memory;
  - plain `"cg"` is not recommended for flow solves: its residual is not monotone;
  - `rs_cgmrs` in this collection has a fix for an early stop (`../cgmrs/README.md`).
- **`smoother_iterations: 2`** is about 17% cheaper per step once the flow is developed, and about 4% dearer in the start-up transient.
- **`line_factor_precision: "single"`** stores the GPU line factors in single precision (factored and applied in double). It is about 10% faster, and the iteration counts are identical in all tests.
- **Small cases** (tens of thousands of dofs per rank): `smoother_iterations: 2` keeps the robustness at about `hsmg`'s cost per step.

**Options beyond Neko's `phmg` keys** (`pcoarsening_schedule`, `smoother_iterations`, `coarse_grid.*`, `update.*`):

| Key | Default | Meaning |
|---|---|---|
| `smoother_cheby_acc` | `jacobi` | Chebyshev accelerator: `line` (robust), `jacobi`, `schwarz`. |
| `line_columns` | false | Chain element lines into columns (needed for `colagg` and on the GPU). |
| `line_global_columns` | true | Exact columns across ranks (Schur-complement exchange). |
| `line_factor_precision` | `double` | `single`: float device copies of the line factors. |
| `line_direction` | 0 | 0: algebraic choice; 1-3: force a reference direction. |
| `line_geometric` | false | Choose the direction from the geometric factors instead of the operator. |
| `line_iso_ratio` | 0 | Elements below this coupling anisotropy use Jacobi. |
| `line_planes`, `line_plane_ratio` | false, 16 | Experimental plane strips for elements thin in two directions (not in single precision). |
| `coarse_solver` | `tamg` | `colagg` (aggregation, robust), `tamg` (tree-AMG as in `phmgx`), `cg` (CPU only). |
| `colagg_cycles` | 0 | Aggregation cycles per coarse solve; must be > 0 on the GPU (use 1). |
| `colagg_direct` | false | Sparse direct solve of $A_c$ (recommended). |
| `colagg_direct_solver` | `spchol` | `spchol` (own) or `pardiso` (MKL, looked up at run time). |
| `colagg_dense_max` | 2000 (CPU), 16000 (GPU) | Dense coarse factor up to this size (double only; set 0 with `colagg_direct`). |
| `colagg_hierarchy`, `colagg_crs_host` | false, false | Settings of the replicated coarse CG (use true, true). |
| `colagg_min_merge` | 0.05 | Below this merged fraction, use the tree-AMG instead. |
| `colagg_kappa` | 4 | Length cap of merged runs (0: no cap). |
| `colagg_theta` | 1 | Strength threshold of the merge criterion. |
| `colagg_smooth` | 3 | Chebyshev steps in the aggregation cycle. |
| `coarse_tol`, `coarse_max_iter`, `coarse_pc` | 1e-8, 2000, `auto` | The iterative coarse solve. |
| `cheby_lanczos_its`, `cheby_power_its` | 20, 150 | Eigenvalue estimates of the Chebyshev smoothers. |
| `coarse_grid.jacobi_smoother`, `coarse_grid.jacobi_probes` | false, -1 | The `phmgx` tree-AMG options when `coarse_solver` is `tamg`. |

**Diagnostics** (environment variables):

| Variable | Effect |
|---|---|
| `RS_TIMERS` | Per-level timers, printed at the end (adds device syncs). |
| `RS_LINE_CHECK` | One-time comparison of the GPU line solve with the host solve. |
| `RS_PHMG_SYMCHECK` | Symmetry test of the preconditioner at the first solve. |

The log shows `RS_LINE`/`RS_COLAGG` setup lines and `RS_MEM` lines (memory per setup phase, maximum over ranks).

## Examples

`examples/2d_cylinder/` uses Neko's `2d_cylinder` mesh at N = 7, with `rs_phmg` in `cg`, `gmres` and `rs_cgmrs`:

```sh
cp <neko>/examples/2d_cylinder/2d_cylinder.nmsh .
mpirun -n 4 ./neko_rs rs_phmg_gmres.case
```

On that mild mesh it takes 13-15 iterations at tolerance 1e-7, against 14-19 for `phmg`. The gain is on bad meshes.

## Status

**Poisson benchmarks:**

| Mesh | Iterations |
|---|---|
| pancake, aspect ratio 1e4 | 6 |
| graded wall | 9 |
| real meshes | 6-14 |
| airfoil (CG, random right-hand side) | 365, where `phmg` stalls |

- It is faster than `phmg` in wall time on every mesh tried except the isotropic box (1.2× slower).
- **16.8 M-dof pancake, 192 CPU ranks:** 6 iterations in 0.46 s; `phmg` fails.
- **Foil flow on H100:** it converges in the start-up, where `phmgx` reaches the iteration cap. Once the flow is developed, each solve needs 1-2 iterations.

**Open:**
- the setup's scalability (3-7 s at scale);
- distributing $A_c$;
- Schwarz smoothing for non-aligned anisotropy.
