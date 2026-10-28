# fdm: exact box solver (fast diagonalization)

On an **axis-aligned box of hexahedral elements** this type applies the exact inverse of Neko's Helmholtz operator $h_1 A + h_2 B$. It covers:
- the pressure ($h_2 = 0$);
- the velocity components and scalars (with $h_2 = \beta_0/\Delta t$ terms).

Around it, CG and GMRES converge in one iteration. The solver type `fdm` skips the Krylov iteration altogether.

## Method

On a tensor-product box the assembled SEM stiffness and mass matrices are Kronecker sums of 1D matrices:

$$A = B_z \otimes B_y \otimes A_x + B_z \otimes A_y \otimes B_x + A_z \otimes B_y \otimes B_x .$$

The $B$-orthonormal 1D eigenbases $S_d$ ($A_d S_d = B_d S_d \Lambda_d$) diagonalize them:

$$(h_1 A + h_2 B)^{-1} = (S_z \otimes S_y \otimes S_x)\, \big[h_1 (\Lambda_x \oplus \Lambda_y \oplus \Lambda_z) + h_2\big]^{-1} (S_z \otimes S_y \otimes S_x)^{\mathsf T} .$$

A solve proceeds in five steps:
1. gather the residual to the free nodes of the global grid;
2. three transforms ($S^{\mathsf T}$ per direction, as GEMMs);
3. a pointwise division;
4. three transforms back;
5. scatter.

Dirichlet nodes are removed from the 1D bases, and periodic directions use periodic 1D operators. For a singular operator (pure Neumann, $h_2 = 0$) the constant mode is set to zero, which gives the pseudo-inverse.

**Parallel layout.** The grid is split into z-slabs, which hold the x and y transforms. One all-to-all transposes the slabs into z-pencils, for the z transform and the scaling, and another transposes them back. Optionally there is a 2D pencil decomposition (`rank_grid`).

**On the CPU two cheaper bases are used where they apply:**
- **Bloch bases for uniform periodic directions:** a cos/sin transform over the elements, plus small $2N \times 2N$ blocks.
- **Even/odd bases for mirror-symmetric directions,** such as the wall-normal direction of a channel. They need two half-size GEMMs instead of one.

## Compatibility

- **Mesh:** an axis-aligned box on a tensor-product grid (elements in i, j, k order, edges per direction). Element sizes may vary per direction. A deformed box is refused at setup.
- **Polynomial order:** $l_x = l_y = l_z \ge 2$.
- **Boundary conditions:**
  - Dirichlet conditions must cover whole box faces;
  - Neumann (natural) and periodic conditions are allowed;
  - mixed and symmetry conditions are refused;
  - a periodic direction with exactly 2 elements is refused.
- **Coefficients:** $h_1$ and $h_2$ must be uniform for the solve to be exact.
  - With variable viscosity, SVV or the full-stress formulation, it still works as a preconditioner. `update()` checks the uniformity and warns once.
  - Use a Krylov solver (`cg`/`gmres`) then, not the `fdm` solver type.
- **Backends:**
  - **CPU (host path):** any number of ranks;
  - **HIP:** tested on MI300A;
  - **CUDA:** tested on a GH200.
- **Precision:** double-precision Neko only. `exchange_precision: "single"` sends the all-to-alls in single precision on the device.

## Build

Use `build.sh` from the collection root (any backend):

```sh
../build.sh -o neko_fdm fdm
```

Or use makeneko directly:

```sh
makeneko box_fdm.f90 fdm_dev_stub.f90   # CPU Neko
makeneko box_fdm.f90 fdm_dev.hip        # HIP Neko
cp fdm_dev.hip fdm_dev.cu && makeneko box_fdm.f90 fdm_dev.cu   # CUDA Neko
```

- `fdm_dev.hip` loads hipBLAS/cuBLAS with `dlopen`, so no extra link flags are needed.
- `fdm_dev_stub.f90` holds the CPU stand-ins of the device entry points. It is generated from `box_fdm.f90` by `gen_stub.py`; rerun that after changing the interface block.

## Case file

As a preconditioner, for the pressure, the velocity or a scalar:

```json
"pressure_solver": {
  "type": "gmres",
  "preconditioner": {"type": "fdm"},
  "projection_space_size": 0,
  "absolute_tolerance": 1e-9,
  "max_iterations": 20
}
```

As the solver type: one application, with no Krylov vectors in memory, no extra operator application and no dot products.

```json
"velocity_solver": {
  "type": "fdm",
  "preconditioner": {"type": "fdm", "residual": true},
  "projection_space_size": 0,
  "absolute_tolerance": 1e-9,
  "max_iterations": 20
}
```

- The `fdm` solver type needs the `fdm` preconditioner and reports one iteration.
- With `"residual": true` it measures the true residual at every solve. Without it, it measures at the first ten solves and every 100th, and reports the last measured value.

Options (inside `"preconditioner"`):

| Key | Default | Meaning |
|---|---|---|
| `verbose` | false | Log the boundary condition types and the setup phase times. |
| `residual` | false | The solver type measures the true residual at every solve. |
| `profile` | false | Synchronizing timers of the solve phases, printed every 50 solves. |
| `exchange_precision` | `"double"` | `"single"`: the all-to-alls move floats (device MPI only). Pressure accurate to about 1e-5 relative, velocity and scalars to about 1e-7. |
| `rank_grid` | – | `[Py, Pz]`: the 2D pencil decomposition (device MPI). It is meant for more ranks than z planes; on one node it is slower than the default slabs. |
| `bloch` | true | CPU: Bloch basis for uniform periodic directions. |
| `fold` | true | CPU: even/odd basis for mirror-symmetric directions. |
| `chunks` | 1 | Device: the slab transposes in chunks, overlapped with the transforms (not faster on one node). |

## Example

`examples/channel/` is a 6 × 8 × 6-element channel (periodic in x and z, walls in y, N = 5) with the fluid and a Boussinesq scalar.
- `fdm.case`: the fdm preconditioner in CG/GMRES.
- `fdm_ksp.case`: the solver type.
- `ref.case`: Neko's own solvers at tight tolerances, for comparison.

```sh
genmeshbox 0 6.283185307179586 0 2 0 3.141592653589793 6 8 6 .true. .false. .true.   # writes box.nmsh
mpirun -n 2 ./neko_fdm fdm.case
```

## Status

**Results:**
- Every solve converges in 1 iteration. The fields agree with Neko's own solvers at tolerance 1e-12 to 8e-8 relative, on 1-4 ranks.
- **2 × EPYC 9654, 192 ranks, 12.6 M points:** 148.9 ms per step with Neko's solvers, 107.7 ms with fdm for all five solves.
- **The user's 37 M-point case on 4 MI300A:** 150 → 91 ms per step.

**Open:**
- the Bloch transform on the device;
- multi-node runs of the slab and 2D schemes.
