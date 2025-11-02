#!/bin/bash
# Quick test script for SIMULATED dataset

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "========================================="
echo "Running FIDP on SIMULATED dataset"
echo "========================================="
echo ""

python Experiment.py \
  -i ../data/data_I.csv \
  -p . \
  -m "FIDP" \
  -d "SIMULATED" \
  -b 128 \
  -lr 0.01 \
  -e 10

echo ""
echo "========================================="
echo "Done! Check Results/Results_FIDP_SIMULATED.xls"
echo "========================================="

