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
module load python/3.11
# unset PYTHONPATH 


cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FISA

# /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/.venv/bin/python Experiment.py \
python Experiment.py \
  -i ../data/data_I.csv \
  -p . \
  -m "FIDP" \
  -d "SIMULATED" \
  -b 128 \
  -lr 0.01 \
  -e 5


