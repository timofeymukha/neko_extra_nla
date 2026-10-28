# diagnostics: rs_cg_lanczos and rs_jacobi

These are two analysis types from the robust_solvers project. They are tools for understanding why a preconditioner converges slowly, not production solvers.

## rs_cg_lanczos (solver type)

Textbook preconditioned CG with the same residual recurrence as Neko's CPU `cg`, so the iteration counts are identical. It also records the CG coefficients $\alpha_j$, $\beta_j$ and, optionally, the $M$-orthonormal Lanczos vectors.

After each solve it builds the Lanczos tridiagonal $T_k$ (Saad, *Iterative Methods for Sparse Linear Systems*, sec. 6.7.3) and diagonalizes it with LAPACK `dstev`. It then reports:
- **the Ritz values of $M^{-1}A$:** log line `LANCZOS {...}`, and the file `<prefix>_ritz.txt`. These give the extreme eigenvalues, hence the condition number the preconditioner achieves;
- **the slowest Ritz pairs**, with an accuracy check and their variation along the three reference directions inside the elements (log lines `RITZ {...}`). This shows *which* modes the preconditioner misses, e.g. modes that are smooth along the stiff direction of thin elements;
- **the slowest Ritz vectors**, written to `<prefix>.fld` for viewing in ParaView/VisIt.

`ksp_t`'s init receives no JSON, so the options come from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `RS_LANCZOS_PREFIX` | `lanczos` | Output prefix. |
| `RS_LANCZOS_NRITZ` | 4 | Number of slowest Ritz vectors kept (0 = off). |
| `RS_LANCZOS_MAXSTORE` | 800 | Maximum number of stored Lanczos vectors. |
| `RS_LANCZOS_FLD` | 1 | 1 = write the Ritz vectors to `<prefix>.fld`. |

## rs_jacobi (preconditioner)

This wraps Neko's point Jacobi, or acts as the identity (`"mode": "jacobi"` or `"identity"`). It logs what its `init(coef, bclst, json)` receives and counts its solves, which checks that the type-injection path initializes, uses and frees a type like the built-in ones.

## Compatibility

- Any mesh, any equation for which CG applies.
- **CPU backend only.**
- `rs_cg_lanczos` needs double precision (`dstev`).

## Build

```sh
../build.sh -o neko_diag diagnostics [other components]
# or: makeneko rs_jacobi.f90 rs_cg_lanczos.f90
```

## Case file

Analyse a preconditioner on the pressure (any `pc`, e.g. `rs_phmg` or `phmg`):

```json
"pressure_solver": {
  "type": "rs_cg_lanczos",
  "preconditioner": {"type": "phmg"},
  "projection_space_size": 0,
  "absolute_tolerance": 1e-10,
  "max_iterations": 800
}
```

```sh
RS_LANCZOS_PREFIX=pressure RS_LANCZOS_NRITZ=4 mpirun -n 4 ./neko_diag case.case
```

Run only a few time steps: the analysis is repeated after every solve. The preconditioner `rs_jacobi` is selected with `"preconditioner": {"type": "rs_jacobi", "mode": "jacobi"}`.
