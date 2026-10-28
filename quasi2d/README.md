# quasi2d: direct solves for quasi-2D cases

Neko has no 2D mode. A "2D" case is a 3D mesh **one element thick in z and periodic in z**, as in `examples/lid` (`lid2d.case`) and `examples/2d_cylinder`. The preconditioner and solver type `quasi2d` solve such cases directly. They cover:
- the pressure;
- the velocity, in both the standard and the **full-stress formulation**;
- scalars.

The coefficients may vary in the plane, e.g. an eddy viscosity $\nu_t(x, y)$. Within its limits this is the one type here for RANS-like 2D work with variable viscosity.

## Method

**Separating z.** Take the $B_z$-orthonormal eigenbasis $S$ of the 1D periodic z operators of the single element ($A_z S = B_z S \Lambda$, with $N_z = l_x - 1$ modes).

- **Scalar Helmholtz problems** (the pressure, the segregated velocity, scalars) split exactly into one 2D problem per mode:

  $$\big(A^{h_1}_{xy} + \lambda_k B^{h_1}_{xy} + B^{h_2}_{xy}\big)\,\hat u_k = (S^{\mathsf T} f)_k .$$

- **The full-stress velocity** (Neko's `ax_helm_full`, with $h_1 = \nu(x, y)$):
  - mode 0, the z-constant part, splits off exactly: the coupled 2D $(u, v)$ stress operator, plus a $\nu$-weighted 2D Helmholtz problem for $w$;
  - higher modes couple $u$–$w$ and $v$–$w$ across modes. For them the preconditioner keeps the per-mode $(u, v)$ and $w$ blocks, which is exact for 2D flow, with eigenvalues in $[0.42, 1.65]$ for 3D content.

**Planes: sparse Cholesky, distributed over all ranks.**
- Rank 0 orders each 2D plane (nested dissection) and does the symbolic factorization; this is the only serial step.
- The supernodal elimination tree is mapped onto the ranks subtree-to-subcube. Subtrees are local to one rank; the top separators are dense fronts distributed block-cyclically over groups of ranks.
- **Numeric factorization, in parallel.** Each rank assembles its own element matrices; local fronts are multifrontal, distributed fronts use a blocked Cholesky. It runs on first use, and again when $h_1$ or $h_2$ change.
- **A solve:**
  1. the z transform, an element-local GEMM;
  2. one all-to-all to the plane-dof owners;
  3. forward and backward sweeps; a distributed front costs one all-reduce per direction;
  4. one all-to-all back.
- Modes and right-hand sides are batched; the three velocity components are solved together.
- Modes with a negligible right-hand side are skipped (`skip_tol`). A 2D flow therefore solves mode 0 only.

**Singular planes.** The pure-Neumann pressure (mode 0, $h_2 = 0$) is solved with one node pinned, and the solution is made mean-free.

**The solver type** is PCG with this preconditioner on Neko's own operator, with Neko's norms. `solve_coupled` handles the full-stress formulation. Neko's `coupled_cg` cannot be used: it applies the preconditioner per component.

## Compatibility

- **Mesh:**
  - exactly **one element across z**;
  - z periodic;
  - every element spanning the whole z extent.
- **In the plane:**
  - unstructured quads, curved elements allowed;
  - periodic directions in x/y allowed;
  - any element orientation.
- **Coefficients:**
  - any $h_1(x, y)$, $h_2(x, y)$;
  - if they depend on z, the z average is used and a message is logged. The solve is then a preconditioner and PCG iterates.
- **Boundary conditions:**
  - Dirichlet and Neumann only;
  - **no symmetry/mixed conditions**;
  - the Dirichlet set must not depend on z.
- **Backends:** **CPU only.** A device backend is refused at setup.
- **Precision:** tested with double-precision Neko.
- **Projection:** keep `projection_space_size` at 0. The solve is direct, and Neko's velocity projection is wrong with the stress formulation.

## Build

Use `build.sh` from the collection root:

```sh
../build.sh -o neko_q2d quasi2d [user.f90]
```

Or use makeneko directly (CPU Neko):

```sh
makeneko q2d_spchol.f90 q2d_plane.f90 quasi2d.f90 [user.f90]
```

## Case file

```json
"pressure_solver": {"type": "quasi2d", "preconditioner": {"type": "quasi2d"},
                    "projection_space_size": 0, "absolute_tolerance": 1e-8, "max_iterations": 20},
"velocity_solver": {"type": "quasi2d", "preconditioner": {"type": "quasi2d"},
                    "projection_space_size": 0, "absolute_tolerance": 1e-8, "max_iterations": 50}
```

- **Standard formulation:** `"type": "cg"` with the `quasi2d` preconditioner also works for the velocity.
- **Full-stress formulation** (`"full_stress_formulation": true`): the solver type `quasi2d` is required.

Options (inside `"preconditioner"`):

| Key | Default | Meaning |
|---|---|---|
| `verbose` | false | Log every factorization. |
| `skip_tol` | 1e-6 | Modes with $\|\hat f_k\| \le$ `skip_tol` $\max_k \|\hat f_k\|$ are not solved. |
| `residual` | `"auto"` | `"always"`: PCG with the true residual at every solve. `"auto"`: check the first 10 solves and every 50th, and switch to `"always"` once a checked solve needs more than one iteration. In between, the solve is $x = M f$. |
| `refactor_tol` | 0 | Refactor when $h_1$ or $h_2$ change by more than this, relative (max norm). With 0 it refactors on any change; with a larger value the factors act as a preconditioner and PCG iterates. |

At the end of a run, each instance logs:
- the number of solves and factorizations;
- the phase times;
- the sweep and factorization timers of rank 0.

## Example

`examples/2d_cylinder/` uses Neko's `2d_cylinder` mesh at N = 7.
- `q2d.case`: standard formulation;
- `q2d_stress.case`: full-stress formulation.

```sh
cp <neko>/examples/2d_cylinder/2d_cylinder.nmsh .
mpirun -n 4 ./neko_q2d q2d.case
```

**Standalone test of the plane solver** (no Neko; it uses the mock modules in `tests/plane/`):

```sh
mpifort -O2 tests/plane/mock_neko.f90 q2d_spchol.f90 q2d_plane.f90 tests/plane/test_plane.f90 -o test_plane -llapack -lblas
mpirun -n 4 ./test_plane 40          # 40 x 40 elements (lx = 8); args: E [nr] [nm] [-2]
```

The test builds the whole matrix on every rank, so keep it small, or use few ranks for large E.

## Status

**Correctness** (`lid2d`, `2d_cylinder`, and a lid with z-dependent initial condition and $\nu_t(x, y)$):
- 1 iteration per solve; 2-3 for full stress with 3D content;
- the fields agree with Neko's own solvers to their tolerance.

**Speed, Helma CPU node (384 cores), 2d_cylinder refined to 1.27e6 plane nodes** (median step):

| Ranks | Standard: Neko → quasi2d | Full stress: Neko → quasi2d |
|---|---|---|
| 96 | 1.16 → 0.34 s (3.4×) | 1.43 → 0.46 s (3.1×) |
| 384 | 0.22 → 0.072 s (3.1×) | 0.26 → 0.095 s (2.7×) |

- **Setup** at that size:
  - serial ordering and symbolic factorization: 6 s per scalar plane, 23 s for the (u, v) plane;
  - parallel numeric factorization: 0.8 s and 3.7 s.
- With full stress on 384 ranks it breaks even with Neko after about 60 steps.

**Open:**
- the device backend;
- fewer synchronizations in the distributed sweep;
- a factorization that keeps scaling beyond about 100 ranks.

**A Neko caveat:** Neko's thin-slab mesh joins some vertical faces mirrored in z. Mode 0 is not affected, and the output is averaged over Neko's dof copies.
