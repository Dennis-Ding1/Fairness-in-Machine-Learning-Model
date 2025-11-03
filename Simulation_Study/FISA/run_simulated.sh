#!/bin/bash

cd "$(dirname "$0")"
source ../../.venv/bin/activate

python Experiment.py \
  -i ../data/data_I.csv \
  -p . \
  -m "FIDP" \
  -d "SIMULATED" \
  -b 128 \
  -lr 0.01 \
  -e 10



