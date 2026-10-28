# extruded: exact pressure solver for extruded meshes

The preconditioner and solver type `extruded` solve the pressure exactly on meshes that are a **2D quad mesh extruded along z**:
- the quads may be curved;
- the layers may have any thicknesses;
- z may be periodic or not;
- the elements may have any orientation.

Typical cases are airfoil and cylinder sections, i.e. spanwise-extruded 2D meshes.

## Method

**Splitting z.** Take the $B_z$-orthonormal 1D eigenbasis $S$ of the z direction ($A_z S = B_z S \Lambda$). Neko's operator then splits into one 2D problem per z mode $k$:

$$\big(h_1 A_{xy} + (h_1 \lambda_k + h_2)\, B_{xy}\big)\, \hat u_k = (S^{\mathsf T} f)_k, \qquad u = S \hat u .$$

**Factorization.** Each distinct mode matrix is factored once at setup with `ext_spchol`, the collection's own sparse supernodal Cholesky. It uses AMD or nested-dissection ordering and the multifrontal method; BLAS/LAPACK is the only dependency.
- Modes with equal $\lambda_k$ (the cos/sin pairs of a periodic z) share a factor.
- Each supernode is stored as $G = [L_{11}^{-1};\ L_{21} L_{11}^{-1}]$, so both triangular solves become matrix-vector products.
- On the GPU, $G$ is kept in fp32 and the arithmetic is fp64.

**Data movement.** The free 2D nodes are split into P ranges. A solve takes three exchanges:
1. the element layout goes to z-pencils;
2. after the z transform (a GEMM), the pencils go to mode planes; modes are assigned to ranks round-robin over the groups of equal $\lambda$;
3. the same exchanges run in reverse.

**Setup from Neko's data:**
- the extrusion axis per element, from the GLL coordinates;
- the merged 2D vertices, which give a canonical 2D numbering;
- the z layers, from the element extents;
- periodicity, from the mesh's periodic facets;
- the Dirichlet set, from the masks of the boundary-condition list.

## Compatibility

- **Equation:** the pressure, or any Helmholtz problem with **uniform $h_1$, $h_2$**. Non-uniform coefficients stop with an error, so it is not usable for the velocity with variable viscosity.
- **Solver type:** for the pressure only.
- **Mesh:**
  - every element must have a unique extrusion axis along z;
  - the 2D mesh may have at most $2^{21}$ vertices;
  - **periodic facets are allowed only between the z ends**, so there is no in-plane periodicity.
- **Boundary conditions:**
  - Dirichlet and Neumann only;
  - the Dirichlet set must not depend on z;
  - at least one Dirichlet node is needed. The singular pure-Neumann pressure (e.g. a fully periodic channel) is not supported yet.
- **Ranks:** at most the number of mode groups, which for periodic z is about half the number of z modes, $N_z = \text{layers} \times N$. With more ranks the setup stops.
- **Backends:**
  - **HIP:** tested on MI300A;
  - **CPU:** host path; checked on 2 ranks for this collection;
  - **CUDA:** the CUDA branch of `ext_dev.hip` compiles with nvcc but has never run.
- **Precision:** double- and single-precision Neko. **In single-precision Neko, use it inside `gmres`/`cg`.** The solver type alone converges slowly there, because Neko's fp32 operator differs from the fp64-assembled one by Neko's metric error.

## Build

Use `build.sh` from the collection root:

```sh
../build.sh -o neko_ext extruded
```

Or use makeneko directly:

```sh
makeneko ext_spchol.f90 extruded.f90 ext_dev_stub.f90    # CPU Neko
makeneko ext_spchol.f90 extruded.f90 ext_dev.hip         # HIP Neko
```

`ext_dev.hip` loads rocBLAS (or cuBLAS) with `dlopen`. On GPUs the exchanges are Fortran MPI calls on device buffers, so a Neko with device-aware MPI is needed.

## Case file

```json
"pressure_solver": {
  "type": "extruded",
  "preconditioner": {"type": "extruded"},
  "projection_space_size": 0,
  "absolute_tolerance": 1e-4,
  "max_iterations": 2
}
```

- **The solver type** applies $x = M f$, then up to `max_iterations - 1` (at most 3) refinement steps $x \leftarrow x + M (f - A x)$ with Neko's operator. It stops as soon as the true residual is below `absolute_tolerance`.
- **As a preconditioner** (`"type": "gmres"` or `"cg"` with `"preconditioner": {"type": "extruded"}`), it converges in 1-2 iterations.
- **Option:** `"verbose": true` logs the setup.

## Example

`examples/cylinder/` is Neko's `cylinder` example: curved quads, 6 periodic layers, N = 7, outflow Dirichlet.
- `cyl_solver.case` uses the solver type;
- `cyl_gmres.case` uses GMRES with the preconditioner.

```sh
cp <neko>/examples/cylinder/cylinder.nmsh .
mpirun -n 2 ./neko_ext cyl_solver.case
```

## Status

**Results:**
- **Neko's cylinder, 1 and 4 ranks:** GMRES + `extruded` converges in 1-2 iterations to round-off. The solver type with one refinement reaches 1e-13 absolute.
- **The user's foil, 1.7e8 points, 4 MI300A:**
  - 0.19 s per step;
  - GMRES + `rs_phmg` took 18-21 s per step there, and GMRES + `phmg` 57 s (not converged).

**Open:**
- the singular case;
- more ranks than mode groups;
- setup time (224 s on the foil);
- rare stalls of 0.1-4 s in a device solve.
