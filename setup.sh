#!/bin/bash

create --name ml_sec_proj python=3.11
conda activate ml_sec_proj

pip install -U "jax[cuda12]"
pip install jaxmarl
pip install -e .[algs]

export PYTHONPATH=./JaxMARL:$PYTHONPATH

echo "Environment setup complete."