# phmgx: phmg with a stronger tree-AMG coarse level

`phmgx` is Neko's hybrid ph-multigrid preconditioner (`phmg`, develop 77243ce) with the user's tree-AMG patch (`neko_phmg_exact_diag_develop-6a661f1`). It is packaged as a user type: the modules are renamed (`phmgx`, `tamgx_multigrid`, `tamgx_smoother`), so it lives next to Neko's own `phmg`.

Use it for the pressure **where `phmg` struggles but converges**: large meshes, moderate aspect ratios. Where it stalls, use `rs_phmg`.

## Method

The structure is the same as `phmg`: p-levels $N \to \ldots \to 1$ with Chebyshev-Jacobi smoothing, then a tree-AMG (aggregation) hierarchy on the $N = 1$ problem. The patch changes the AMG part:

- **A Jacobi-scaled Chebyshev smoother on the AMG levels** (`coarse_grid.jacobi_smoother`). Each level needs its operator diagonal. With `jacobi_probes: -1` that diagonal is **exact**: the matrix-free coarse operator is applied to a graph colouring of the aggregates, which is safe across ranks. With a positive value it is a stochastic estimate from that many probes.
- **A deeper hierarchy** (`coarse_grid.levels` 5-6 instead of 3) and **more V-cycles** (`coarse_grid.iterations` 2). The default 3 levels are too shallow for $10^5$-$10^6$ elements.

The patch also had a GMRES-restart option; it is left out here (Neko's own GMRES is unchanged).

## Compatibility

- **Equation and mesh:** any mesh, any boundary conditions; the pressure (Poisson) is the target. It is a preconditioner: use it inside `gmres`, `cg` or `rs_cgmrs`.
- **Backends:**
  - **CPU** and **CUDA** (H100) have both run;
  - HIP uses the same existing Neko device kernels, so it should work, but it is untested.
- **Precision:** double and single.
  - **With `jacobi_probes: -1` in single precision,** the exact aggregate ids travel as reals. The product (ranks) × (max aggregates per rank) must stay below $2^{23}$; otherwise a warning is printed and the remote exclusion is skipped.
  - At larger counts use `"jacobi_probes": 48`.
- **Cost:** each coarse matvec is matrix-free through the $l_x = 2$ operator with gather-scatter. The deep hierarchy therefore costs 2.8× per iteration at $N = 5$ on 4 GPUs (5× at $N = 3$). It pays off where `phmg` stagnates.

## Build

Use `build.sh` from the collection root (any backend):

```sh
../build.sh -o neko_phmgx phmgx [cgmrs]
```

By hand, the Fortran files go in this order: `tamgx_smoother.f90 tamgx_multigrid.f90 phmgx.f90`. The driver calls `phmgx_register_types()`. No device sources are needed.

## Case file

```json
"pressure_solver": {
  "type": "gmres",
  "preconditioner": {
    "type": "phmgx",
    "pcoarsening_schedule": [3, 1],
    "coarse_grid": {"levels": 6, "iterations": 2,
                    "jacobi_smoother": true, "jacobi_probes": -1}
  },
  "projection_space_size": 0,
  "absolute_tolerance": 1e-4,
  "max_iterations": 250
}
```

- **Polynomial order 3:** use `"pcoarsening_schedule": [1]`, which drops a duplicated $l_x = 4$ level.
- **Options:** all of `phmg`'s, plus:

| Key | Default | Meaning |
|---|---|---|
| `coarse_grid.jacobi_smoother` | false | Diagonally scaled Chebyshev on the AMG levels (the patch; set true). |
| `coarse_grid.jacobi_probes` | 48 | -1: exact coarse diagonal by colouring; > 0: stochastic estimate with that many probes. |

**Startup log.** Look for one pair of lines per AMG level ≥ 1:

```
-- AMG lvl 1 exact diag: mv  293 bnd  56530 err 0.E+00 bnd err 0.E+00
```

`err` and `bnd err` are self-test relative errors and must be about 1e-6 or smaller.

## Examples

`examples/2d_cylinder/` uses Neko's `2d_cylinder` mesh at N = 7:
- `phmgx.case`;
- `phmg_reference.case`, the same case with Neko's own `phmg`.

```sh
cp <neko>/examples/2d_cylinder/2d_cylinder.nmsh .
mpirun -n 4 ./neko_phmgx phmgx.case
```

In the check for this collection (2 ranks, 5 steps) it took 11-16 GMRES iterations per pressure solve, against 14-19 for `phmg`.
