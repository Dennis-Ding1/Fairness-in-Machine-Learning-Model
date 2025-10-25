#!/bin/bash
#SBATCH --job-name=FCPH_ind
#SBATCH --output=outputs/hpc_run_%j.out
#SBATCH --error=outputs/hpc_run_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# ---- Load environment ----
module load python/3.10
source ~/Fairness-in-Machine-Learning-Model/.venv/bin/activate

# ---- Go to working directory ----
cd ~/Fairness-in-Machine-Learning-Model/Simulation_Study/FCPH_FDCPH

# ---- Run ----
python run_individualFair_CPH.py
