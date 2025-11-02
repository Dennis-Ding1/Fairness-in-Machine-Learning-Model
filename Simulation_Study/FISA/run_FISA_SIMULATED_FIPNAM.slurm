#!/bin/bash
#SBATCH --job-name=FISA_NAM_SIM
#SBATCH --output=outputs/hpc_run_%j.out
#SBATCH --error=outputs/hpc_run_%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load python/3.10

cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FISA

source ../../.venv/bin/activate

# Debug: Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import typing_extensions; print('typing_extensions found at:', typing_extensions.__file__)" || echo "ERROR: typing_extensions not found in venv!"

# Run FIPNAM model on SIMULATED dataset
python Experiment.py \
  -i ../data/data_I.csv \
  -p . \
  -m "FIPNAM" \
  -d "SIMULATED" \
  -b 128 \
  -lr 0.01 \
  -e 100


