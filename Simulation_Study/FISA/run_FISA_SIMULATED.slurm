#!/bin/bash
#SBATCH --job-name=FISA_SIM
#SBATCH --output=outputs/hpc_run_%j.out
#SBATCH --error=outputs/hpc_run_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

# module purge
# module load gcc
# module load arrow/21.0.0
# module load python/3.11
# unset PYTHONPATH 


# cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FISA

# # Run FIDP model on SIMULATED dataset
# /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/.venv/bin/python Experiment.py \
#  Experiment.py \
#   -i ../data/data_I.csv \
#   -p . \
#   -m "FIDP" \
#   -d "SIMULATED" \
#   -b 128 \
#   -lr 0.01 \
#   -e 5

set -euo pipefail

module purge
module load gcc
module load python/3.11
unset PYTHONPATH

VENV=/scratch/dinghaoc/Fairness-in-Machine-Learning-Model/.venv
TARGET=$VENV/lib/python3.11/site-packages

$VENV/bin/pip uninstall -y pytz python-dateuti

$VENV/bin/pip install --no-cache-dir --no-build-isolation \
  --target "$TARGET" --upgrade --force-reinstall --ignore-installed \
  pytz==2025.1 python-dateutil==2.9.0.post0

$VENV/bin/python - <<'PY'
import pytz, dateutil, sys
print("python:", sys.executable)
print("pytz     ->", pytz.__file__)
print("dateutil ->", dateutil.__file__)
print("OK if both paths contain: /.venv/lib/python3.11/site-packages")
PY

