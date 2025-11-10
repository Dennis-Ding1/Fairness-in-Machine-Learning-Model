#!/bin/bash
#SBATCH --job-name=FIDP_SIM_I
#SBATCH --output=outputs/FIDP_SIM_I_%A_%a.out
#SBATCH --error=outputs/FIDP_SIM_I_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-2

LAMBDA_VALUES=(0 0.01 0.02)

LAMBDA=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}

module load python/3.11
module load arrow/21.0.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip > /dev/null 2>&1
pip install --no-index -r /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/requirements.txt > /dev/null 2>&1

cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FISA
python Experiment.py \
  -i ../data/data_I.csv \
  -p . \
  -m "FIDP" \
  -d "SIMULATED" \
  -b 128 \
  -lr 0.01 \
  -e 30 \
  -l $LAMBDA
