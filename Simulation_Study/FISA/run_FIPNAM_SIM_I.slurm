#!/bin/bash
#SBATCH --job-name=FIPNAM_SIM_I
#SBATCH --output=outputs/FIPNAM_SIM_I_%A_%a.out
#SBATCH --error=outputs/FIPNAM_SIM_I_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --array=0-2

LAMBDA_VALUES=(0 0.01 0.02)

LAMBDA=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}

module purge
module load StdEnv/2023
module load python/3.11
module load arrow/21.0.0
module load scipy-stack/2025a

export MPLCONFIGDIR=$SLURM_TMPDIR/mpl-config
export XDG_CACHE_HOME=$SLURM_TMPDIR/mpl-cache
mkdir -p $MPLCONFIGDIR
mkdir -p $XDG_CACHE_HOME
export MPLBACKEND=Agg

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip > /dev/null 2>&1
pip install --no-index -r /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/requirements.txt > /dev/null 2>&1

cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FISA
python Experiment.py \
  -i ../data/data_I.csv \
  -p . \
  -m "FIPNAM" \
  -d "SIMULATED" \
  -b 128 \
  -lr 0.01 \
  -e 30 \
  -l $LAMBDA
