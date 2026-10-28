#!/bin/bash
# Builds a Neko executable with any subset of the solvers in this collection
# (and optionally a case's user file), like makeneko, but with the device
# kernels and libraries some of the types need (makeneko's link line is fixed).
#
#   ./build.sh [-o OUTPUT] COMPONENT... [user.f90 ...]
#
# COMPONENT: fdm extruded quasi2d rs_phmg phmgx cgmrs diagnostics, or "all"
#            ("all" = every component the Neko backend supports).
# Neko is found with pkg-config: put the neko.pc of the Neko build you want to
# link against on PKG_CONFIG_PATH. Its backend (CPU, CUDA, HIP) and precision
# (dp, sp) are read from neko.pc.
#
# Environment (optional):
#   FCFLAGS    Fortran flags                         (default -O3)
#   BUILD_DIR  where objects and modules go          (default ./build-<output name>)
#   NVCC, CUDA_ARCH, CUDA_HOME   CUDA compiler, e.g. CUDA_ARCH=-arch=sm_90 (default)
#   HIPCC, HIP_ARCH, ROCM_PATH   HIP compiler, e.g. HIP_ARCH=--offload-arch=gfx942 (default)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=neko
COMPONENTS=()
USER_FILES=()
while [ $# -gt 0 ]; do
  case "$1" in
    -o) OUT=$2; shift 2 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *.f90|*.F90) USER_FILES+=("$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"); shift ;;
    *) COMPONENTS+=("$1"); shift ;;
  esac
done
[ ${#COMPONENTS[@]} -gt 0 ] || { sed -n '2,22p' "$0"; exit 1; }

pkg-config --exists neko || { echo "error: neko.pc not found; add its directory to PKG_CONFIG_PATH" >&2; exit 1; }
FC=$(pkg-config --variable=compiler neko)
BACKEND=$(pkg-config --variable=backend neko)
PREC=$(pkg-config --variable=precision neko)
INCPKG=$(pkg-config --variable=includedir_pkg neko)
echo "Neko: $(pkg-config --variable=prefix neko) (backend $BACKEND, precision $PREC, compiler $FC)"

case "$BACKEND" in
  CPU|CUDA|HIP) ;;
  *) echo "error: backend $BACKEND is not supported by any type here (CPU, CUDA, HIP)" >&2; exit 1 ;;
esac

if [ "${COMPONENTS[*]}" = all ]; then
  COMPONENTS=(fdm rs_phmg phmgx cgmrs diagnostics)
  [ "$BACKEND" != CUDA ] && COMPONENTS+=(extruded)
  [ "$BACKEND" = CPU ] && COMPONENTS+=(quasi2d)
fi
has() { local c; for c in "${COMPONENTS[@]}"; do [ "$c" = "$1" ] && return 0; done; return 1; }
for c in "${COMPONENTS[@]}"; do
  case "$c" in
    fdm|extruded|quasi2d|rs_phmg|phmgx|cgmrs|diagnostics) ;;
    *) echo "error: unknown component '$c'" >&2; exit 1 ;;
  esac
done

# Backend restrictions (see each component's README)
if has quasi2d && [ "$BACKEND" != CPU ]; then
  echo "error: quasi2d supports the CPU backend only" >&2; exit 1
fi
if has extruded && [ "$BACKEND" = CUDA ]; then
  echo "warning: extruded has only been run with HIP and on the host; its CUDA branch is untested" >&2
fi
if has diagnostics && [ "$BACKEND" != CPU ]; then
  echo "warning: the diagnostics types run on the CPU backend only (they stop with an error on a device)" >&2
fi

F90=()        # Fortran sources, in dependency order
DEV=()        # device sources: "source:object-name:extra-flags"
LIBS=()       # extra libraries
REG=()        # modules with a <module>_register_types routine

if has fdm; then
  F90+=("$HERE/fdm/box_fdm.f90"); REG+=(box_fdm)
  case "$BACKEND" in
    CPU) F90+=("$HERE/fdm/fdm_dev_stub.f90") ;;
    *) DEV+=("$HERE/fdm/fdm_dev.hip:fdm_dev:") ;;
  esac
fi
if has extruded; then
  F90+=("$HERE/extruded/ext_spchol.f90" "$HERE/extruded/extruded.f90"); REG+=(extruded)
  case "$BACKEND" in
    CPU) F90+=("$HERE/extruded/ext_dev_stub.f90") ;;
    *) DEV+=("$HERE/extruded/ext_dev.hip:ext_dev:") ;;
  esac
fi
if has quasi2d; then
  F90+=("$HERE/quasi2d/q2d_spchol.f90" "$HERE/quasi2d/q2d_plane.f90" "$HERE/quasi2d/quasi2d.f90"); REG+=(quasi2d)
fi
if has rs_phmg; then
  RSFLAG=""; [ "$PREC" = sp ] && RSFLAG=-DRS_REAL_SP
  case "$BACKEND" in
    CPU) F90+=("$HERE/rs_phmg/rs_line_cuda_stub.f90") ;;
    CUDA) DEV+=("$HERE/rs_phmg/rs_line_cuda.cu:rs_line_cuda:$RSFLAG")
          LIBS+=(${CUDA_HOME:+-L$CUDA_HOME/lib64} -lcublas -lcusolver -lcudart -lstdc++) ;;
    HIP) DEV+=("$HERE/rs_phmg/rs_line_hip.cpp:rs_line_hip:$RSFLAG")
         LIBS+=(-L${ROCM_PATH:-/opt/rocm}/lib -lamdhip64 -lrocblas -lrocsolver -lstdc++) ;;
  esac
  F90+=("$HERE/rs_phmg/rs_spchol.f90" "$HERE/rs_phmg/rs_line.f90" "$HERE/rs_phmg/rs_colagg.f90")
  # rs_phmg's tree-AMG fallback lives in phmgx/
  F90+=("$HERE/phmgx/tamgx_smoother.f90" "$HERE/phmgx/tamgx_multigrid.f90" "$HERE/rs_phmg/rs_phmg.f90")
  REG+=(rs_phmg)
fi
if has phmgx; then
  has rs_phmg || F90+=("$HERE/phmgx/tamgx_smoother.f90" "$HERE/phmgx/tamgx_multigrid.f90")
  F90+=("$HERE/phmgx/phmgx.f90"); REG+=(phmgx)
fi
if has cgmrs; then F90+=("$HERE/cgmrs/rs_cgmrs.f90"); REG+=(rs_cgmrs); fi
if has diagnostics; then
  F90+=("$HERE/diagnostics/rs_jacobi.f90" "$HERE/diagnostics/rs_cg_lanczos.f90"); REG+=(rs_jacobi rs_cg_lanczos)
fi
LIBS+=(-ldl)

# The case's user file(s), handled as makeneko does
HAS_USER=0
for f in ${USER_FILES[@]+"${USER_FILES[@]}"}; do
  grep -qiE '^[[:space:]]*module[[:space:]]+user([[:space:]]|$)' "$f" && HAS_USER=1
  for m in $(grep -hiE '^[[:space:]]*module[[:space:]]+[a-z_][a-z0-9_]*' "$f" | grep -viE 'module[[:space:]]+(procedure|user([[:space:]]|$))' | awk '{print tolower($2)}'); do
    grep -qiE "subroutine[[:space:]]+${m}_register_types" "$f" && REG+=("$m")
  done
done

B=${BUILD_DIR:-$PWD/build-$(basename "$OUT")}
mkdir -p "$B"
OUTABS=$(cd "$(dirname "$OUT")" && pwd)/$(basename "$OUT")
cd "$B"
rm -f ./*.mod ./*.smod ./*.o

OBJ=()
for d in ${DEV[@]+"${DEV[@]}"}; do
  IFS=: read -r src obj flags <<< "$d"
  if [ "$BACKEND" = CUDA ]; then
    # the .hip sources carry a CUDA branch; nvcc needs a .cu name
    case "$src" in *.hip) cp "$src" "$obj.cu"; src=$obj.cu ;; esac
    ${NVCC:-nvcc} -O3 ${CUDA_ARCH:--arch=sm_90} $flags -Xcompiler -fPIC -I"$INCPKG" -c "$src" -o "$obj.o"
  else
    ${HIPCC:-hipcc} -O3 ${HIP_ARCH:---offload-arch=gfx942} -fPIC $flags -I"$INCPKG" -c "$src" -o "$obj.o"
  fi
  OBJ+=("$obj.o")
done

{
  echo "program usrneko"
  echo "  use neko"
  [ $HAS_USER -eq 1 ] && echo "  use user"
  for m in "${REG[@]}"; do echo "  use $m, only : ${m}_register_types"; done
  echo "  type(case_t), target :: C"
  if [ $HAS_USER -eq 1 ]; then
    echo "  call user_setup(C%user)"
    echo "  if (C%user%suppress_type_injection .eqv. .false.) then"
    for m in "${REG[@]}"; do echo "     call ${m}_register_types()"; done
    echo "  end if"
  else
    for m in "${REG[@]}"; do echo "  call ${m}_register_types()"; done
  fi
  echo "  call neko_init(C)"
  echo "  call neko_user_access%init(C)"
  echo "  call neko_solve(C)"
  echo "  call neko_finalize(C)"
  echo "end program usrneko"
} > usr_driver.f90

echo "Registering: ${REG[*]}"
$FC ${FCFLAGS:--O3} $(pkg-config --cflags neko) "${F90[@]}" ${USER_FILES[@]+"${USER_FILES[@]}"} usr_driver.f90 \
    ${OBJ[@]+"${OBJ[@]}"} -o "$OUTABS" $(pkg-config --libs neko) "${LIBS[@]}"
echo "built $OUTABS"
