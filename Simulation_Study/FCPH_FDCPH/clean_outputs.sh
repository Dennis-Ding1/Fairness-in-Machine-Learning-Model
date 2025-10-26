#!/bin/bash

cd outputs || { echo "Error: outputs directory not found"; exit 1; }
rm -rf *
echo "✅ Cleaned outputs/"
cd ..

cd trained-models || { echo "Error: trained-models directory not found"; exit 1; }
rm -rf *
echo "✅ Cleaned trained-models/"

cd ..

