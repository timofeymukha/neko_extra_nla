# neko_extra_nla: extra solvers and preconditioners for Neko

This is a collection of linear solvers (`ksp_t`) and preconditioners (`pc_t`) for [Neko](https://github.com/ExtremeFLOW/neko). They come from two research projects, `box_solvers` and `robust_solvers`.

**How they plug in.** Each type is injected at run time through Neko's type registry (`register_precon` / `register_krylov`). Neko itself is not modified: you build a Neko executable that contains the types you want, and select them by name in the case file.

**Requirement:** Neko `develop` at or after **77243ce** ("Add deferred init and free to precon, fixes type injection", #2845). Older Neko cannot initialize injected preconditioners.

## What is here

| Folder | JSON type(s) | Kind | Use it for |
|---|---|---|---|
| [`fdm/`](fdm/README.md) | `fdm` | pc + solver | Pressure, velocity and scalars on an **axis-aligned box of hexahedra** (tensor-product grid). It is an exact direct solve by fast diagonalization, so each solve takes 1 iteration. |
| [`extruded/`](extruded/README.md) | `extruded` | pc + solver | The **pressure on extruded meshes**: a 2D quad mesh, possibly curved, extruded in z. It is an exact direct solve, one sparse Cholesky per z mode. |
| [`quasi2d/`](quasi2d/README.md) | `quasi2d` | pc + solver | **Quasi-2D cases**: one element thick in z, periodic in z, like Neko's 2D examples. It handles the pressure, the velocity (standard and full-stress formulations) and scalars. Coefficients may vary in (x, y), e.g. $\nu_t(x, y)$. It is a direct solve, distributed over all ranks. |
| [`rs_phmg/`](rs_phmg/README.md) | `rs_phmg` | pc | The **pressure on bad meshes**: very high aspect ratios, boundary-layer stacks, start-up transients where `phmg` stalls. It is p-multigrid with a column-line smoother and an aggregation coarse level. Any hexahedral mesh. |
| [`phmgx/`](phmgx/README.md) | `phmgx` | pc | Neko's `phmg` with a stronger tree-AMG coarse level (Jacobi-scaled Chebyshev, exact coarse diagonal, deeper hierarchy). Use it where `phmg` struggles but converges. Any mesh. |
| [`cgmrs/`](cgmrs/README.md) | `rs_cgmrs` | solver | CG with minimal-residual smoothing: CG memory and a monotone residual. A low-memory alternative to GMRES for flow solves. Contains a fix not yet in robust_solvers (see its README). |
| [`diagnostics/`](diagnostics/README.md) | `rs_cg_lanczos`, `rs_jacobi` | solver, pc | Analysis tools: Ritz values and slow modes of $M^{-1}A$, and a test preconditioner. These are not meant for production. |

### Which one to use

**Pressure:**
- If the geometry is a box, use `fdm`.
- If it is one element thick and periodic in z, use `quasi2d`.
- If it is an extrusion of a 2D mesh, use `extruded`.
- Otherwise, use `phmgx`. If that stalls (very high aspect ratios), use `rs_phmg`.

**Velocity and scalars:**
- On a box, use `fdm`. It is exact for constant viscosity, and a preconditioner otherwise.
- On a quasi-2D mesh, use `quasi2d`. It is exact for $\nu(x, y)$ and also covers the full-stress formulation.

## Compatibility

| Type | Mesh | Coefficients $h_1$, $h_2$ | Boundary conditions | CPU | CUDA | HIP | Neko precision |
|---|---|---|---|---|---|---|---|
| `fdm` | an axis-aligned box on a tensor-product grid, no deformation | constant for an exact solve; otherwise it is a preconditioner (logged) | Dirichlet on whole box faces, Neumann, periodic; no symmetry/mixed conditions; a periodic direction must not have exactly 2 elements | yes | yes (GH200) | yes (MI300A) | dp only |
| `extruded` | a 2D quad mesh (curved allowed) extruded along z, any layers, z periodic or not | constant (the pressure); the solver type refuses anything else | Dirichlet/Neumann, a Dirichlet set independent of z; at least one Dirichlet node; periodicity only between the z ends | yes | compiles; untested | yes (MI300A) | dp, sp |
| `quasi2d` | one element thick in z, z periodic (Neko's "2D" cases) | any $h_1(x, y)$, $h_2(x, y)$; z-dependent coefficients use their z average (then it is a preconditioner) | Dirichlet/Neumann, z-independent Dirichlet set, periodic in x/y allowed; no symmetry/mixed conditions | yes | no | no | dp (tested) |
| `rs_phmg` | any hexahedral mesh | any (it is a preconditioner) | any of Neko's | yes | yes (H100) | yes (MI300A, sp) | dp, sp |
| `phmgx` | any | any | any | yes | yes (H100) | should work; untested | dp, sp |
| `rs_cgmrs` | any | any | any | yes | yes (H100) | untested | any |
| diagnostics | any | any | any | yes | no | no | dp |

- In the hardware columns, a GPU in brackets says where that combination was run.
- **"Exact"** means the solve is Neko's inverse up to round-off. The Krylov method around it then converges in 1 iteration, or the solver types skip the Krylov iteration altogether.
- **OpenCL, Metal and SX-Aurora are not supported.**

## Building

The types are compiled into the Neko executable. There are two routes.

**1. `build.sh` (recommended; works for every combination and backend):**

```sh
export PKG_CONFIG_PATH=<neko install>/lib/pkgconfig:$PKG_CONFIG_PATH   # the Neko to link against
./build.sh -o neko_nla quasi2d fdm             # the components you want
./build.sh -o neko_nla all                     # everything the backend supports
./build.sh -o neko_case rs_phmg cgmrs user.f90 # plus the case's user file
```

- The script reads Neko's backend (CPU, CUDA, HIP) and precision (dp, sp) from `neko.pc`. It then:
  - picks each type's device sources or CPU stubs;
  - compiles the device kernels (`nvcc` / `hipcc`) and adds the libraries they need (cuBLAS/cuSOLVER, rocBLAS/rocSOLVER);
  - writes the driver that registers the types, and links.
- Options are set through the environment:
  - `FCFLAGS` (default `-O3`) and `BUILD_DIR`;
  - `CUDA_ARCH` (default `-arch=sm_90`) and `CUDA_HOME`;
  - `HIP_ARCH` (default `--offload-arch=gfx942`) and `ROCM_PATH`.

**2. Neko's `makeneko`.** This works for `fdm`, `extruded`, `quasi2d`, `cgmrs` and `diagnostics`; the exact lines are in their READMEs.
- It does not work for `rs_phmg`, on any backend. makeneko cannot link the cuBLAS/cuSOLVER or rocBLAS/rocSOLVER libraries the kernels need. Its generated driver also imports every module without `only`, which makes a stub routine name ambiguous on the CPU.
- Use `build.sh` for `rs_phmg` and `phmgx`.

## Provenance

| Folder | Source | Commit |
|---|---|---|
| `fdm` | `box_solvers/src/neko-fdm` | a6a29a2 |
| `extruded` | `box_solvers/src/neko-extruded` | f64444e |
| `quasi2d` | `box_solvers/src/neko-quasi2d` | 66526d4 |
| `rs_phmg`, `phmgx`, `cgmrs`, `diagnostics` | `robust_solvers/src/neko_types` | 00b2b3d |

`rs_phmg.f90`, `phmgx.f90` and the `tamgx_*` modules are derived from Neko's `phmg` and tree-AMG (BSD-3-Clause, the Neko authors); their license headers are kept.

**License:** BSD-3-Clause (`LICENSE`), the same terms as Neko.

The files are copies of the committed sources, with two changes:
- `rs_phmg/rs_line_hip.cpp` defines `cudaMalloc`/`cudaFree` for the HIP build (see `rs_phmg/README.md`);
- `cgmrs/rs_cgmrs.f90` computes the smoothed residual norm directly instead of by a recurrence that stopped solves early (see `cgmrs/README.md`). The method write-ups are in the projects' logbooks and briefs:
- `box_solvers`: briefs 3-6, logbooks 2026-10-12 to 2026-10-28;
- `robust_solvers`: brief 6 and the technical report of 2026-10-14.
