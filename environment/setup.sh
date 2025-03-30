#!/bin/bash
cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_sec_project

pip install -e .[algs]
export PYTHONPATH=./JaxMARL:$PYTHONPATH

echo "Environment setup complete."