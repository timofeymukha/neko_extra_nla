# rs_cgmrs: CG with minimal residual smoothing

`rs_cgmrs` is a Krylov solver type. It runs preconditioned CG with **minimal residual smoothing** (Zhou & Walker, SIAM J. Sci. Comput. 15, 1994): CG's memory plus two vectors, and a residual norm that never increases.

**Why.** CG minimizes the error in the $A$-norm, so its residual 2-norm oscillates. Neko stops on the residual norm, and flow solves often start only a few times above the tolerance. The oscillating residual then decides the iteration count. On the foil:

| Solver | Iterations per step |
|---|---|
| plain CG | 10-44 per solve (20.3 per step on average) |
| GMRES | 4.8 |
| `rs_cgmrs` | 5.6 |

`rs_cgmrs` gets close to GMRES with 6 work vectors instead of GMRES's 2 × 30.

## Method

Alongside the CG pair $(x, r)$, a second iterate $y$ with residual $s$ is carried:

$$d = r - s,\quad \eta = -\frac{(s, d)}{(d, d)},\quad s \leftarrow s + \eta\, d,\quad y \leftarrow (1 - \eta)\, y + \eta\, x .$$

$s$ is the minimum-norm residual on the line through $s$ and $r$, so $\|s\|$ does not increase. The solve stops on $\|s\|$ and returns $y$.
- CG itself is unchanged.
- The extra cost is two vectors, three dot products and three vector updates per iteration.
- $\|s\|$ is computed from $s$ itself, not by a recurrence (see the fix below).
- Inner products are Neko's: mult-weighted and scaled by $1/\sqrt{\text{volume}}$.

## Compatibility

- Any mesh, any equation for which CG applies (an SPD operator with a symmetric preconditioner).
- **Backends:**
  - **CPU** and **CUDA** (H100, foil flow runs);
  - HIP: it uses only Neko's device math, so it should work, but it is untested.
- **Precision:** any; it uses Neko's math routines only.

## Fix in this collection (2026-10-28): the smoothed norm

**What was wrong.** The robust_solvers source (00b2b3d) updated the smoothed norm by the recurrence $\|s\|^2 \leftarrow \|s\|^2 - (s, d)^2/(d, d)$, clamped at zero.
- In finite precision the recurrence drifts. It can reach zero, so the solver stops early and reports a final residual of `0.0`. It can also stay above the tolerance while the iterate has converged.
- **Seen in the check for this collection:** `2d_cylinder`, N = 7, absolute tolerance 1e-7.
  - With `phmg`, solves reported 0.0. `RS_CGMRS_CHECK=1` showed true residuals of 1.5e-7 and 3e-7.
  - With `rs_phmg`, one solve ran to the 250-iteration cap at 4.4e-7.

**The fix.** $\|s\|^2 = (s, s)$ is computed directly after each update: one more dot product, i.e. one more global reduction, per iteration.

**After the fix, the same runs:**
- the smoothed residual equals the true residual to all printed digits, and every solve ends below the tolerance;
- with `phmg`: 14-19 iterations;
- with `rs_phmg`: 13-15 iterations, and no capped solve.

**Status in robust_solvers:** this fix is not yet in the robust_solvers repository; the issue is noted in its logbook (2026-10-28). Runs made with the original version and tight tolerances may have stopped early; `RS_CGMRS_CHECK=1` shows it.

## Build

Use `build.sh` from the collection root:

```sh
../build.sh -o neko_cgmrs cgmrs [rs_phmg] [phmgx]
```

It is plain Fortran (`rs_cgmrs.f90`), registered by `rs_cgmrs_register_types()`. Neko's makeneko also works for it alone: `makeneko rs_cgmrs.f90`.

## Case file

```json
"pressure_solver": {
  "type": "rs_cgmrs",
  "preconditioner": {"type": "phmg"},
  "projection_space_size": 0,
  "absolute_tolerance": 1e-4,
  "max_iterations": 250
}
```

There are no options of its own. The environment variable `RS_CGMRS_CHECK` prints the true residual $\|f - A y\|$ of each solution, at the cost of one extra operator application.

## Example

`examples/2d_cylinder/cgmrs_phmg.case` uses Neko's `2d_cylinder` mesh (copy `2d_cylinder.nmsh` next to it), with `rs_cgmrs` and Neko's `phmg`. For `rs_cgmrs` with `rs_phmg`, see `../rs_phmg/examples/2d_cylinder/rs_phmg_cgmrs.case`.
