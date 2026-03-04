$ErrorActionPreference = "Stop"

$ROOT_DIR = Get-Location
cd $ROOT_DIR

$PYTHON_BIN = "python"
$OUT_DIR = "results/smoke_run"
$GAMMA_POINTS = 2
$LAMBDA_POINTS = 2
$SEEDS = 1
$N_WORKERS = 1

$env:PYTHONPATH="."
$env:MPLCONFIGDIR="$ROOT_DIR\.cache\matplotlib"

python -m compileall src

python -m experiments.main_experiment `
  --gamma_points $GAMMA_POINTS `
  --lambda_points $LAMBDA_POINTS `
  --seeds $SEEDS `
  --n_workers $N_WORKERS `
  --out_dir $OUT_DIR