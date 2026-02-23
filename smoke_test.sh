#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${1:-results/smoke_run}"
GAMMA_POINTS="${GAMMA_POINTS:-2}"
LAMBDA_POINTS="${LAMBDA_POINTS:-2}"
SEEDS="${SEEDS:-1}"
N_WORKERS="${N_WORKERS:-1}"
MPL_CACHE_DIR="${MPLCONFIGDIR:-$ROOT_DIR/.cache/matplotlib}"

mkdir -p "$MPL_CACHE_DIR"

echo "[smoke] root: $ROOT_DIR"
echo "[smoke] python: $PYTHON_BIN"
"$PYTHON_BIN" -V

echo "[smoke] checking required python modules..."
if ! "$PYTHON_BIN" -c 'import importlib,sys;mods=["numpy","matplotlib","tqdm","simpy"];miss=[];[miss.append(m) for m in mods if importlib.util.find_spec(m) is None];print("missing:", ",".join(miss)) if miss else print("all required modules found");sys.exit(1 if miss else 0)'; then
  echo "[smoke] missing dependencies detected."
  echo "[smoke] if you use conda: conda env create -f environment.yml && conda activate mfhr_sti"
  exit 1
fi

echo "[smoke] compile check..."
"$PYTHON_BIN" -m compileall src >/dev/null

echo "[smoke] running minimal experiment..."
PYTHONPATH=. MPLCONFIGDIR="$MPL_CACHE_DIR" "$PYTHON_BIN" -m experiments.main_experiment \
  --gamma_points "$GAMMA_POINTS" \
  --lambda_points "$LAMBDA_POINTS" \
  --seeds "$SEEDS" \
  --n_workers "$N_WORKERS" \
  --out_dir "$OUT_DIR"

echo "[smoke] passed."
echo "[smoke] outputs written to: $OUT_DIR"
