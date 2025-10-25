#!/bin/bash
#SBATCH --job-name=FCPH_ind
#SBATCH --output=outputs/hpc_run_%j.out
#SBATCH --error=outputs/hpc_run_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4

module load python/3.10

cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FCPH_FDCPH

python run_individualFair_CPH.py
