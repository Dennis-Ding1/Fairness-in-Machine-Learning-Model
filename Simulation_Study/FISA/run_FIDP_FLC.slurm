#!/bin/bash
#SBATCH --job-name=FIDP_FLC
#SBATCH --output=outputs/FIDP_FLC_%j.out
#SBATCH --error=outputs/FIDP_FLC_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10

module load python/3.11
module load arrow/21.0.0

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip > /dev/null 2>&1
pip install --no-index -r /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/requirements.txt > /dev/null 2>&1

cd /scratch/dinghaoc/Fairness-in-Machine-Learning-Model/Simulation_Study/FISA
python Experiment.py \
  -i Data/FLChain/flchain.csv \
  -p . \
  -m "FIDP" \
  -d "FLChain" \
  -b 32 \
  -lr 0.01 \
  -e 50
