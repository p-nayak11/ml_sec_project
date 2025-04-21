#!/bin/bash

pip install poetry
poetry install
eval $(poetry env activate)
pip install -e .[algs]
poetry add jax[cuda]

export PYTHONPATH=./JaxMARL:$PYTHONPATH
export JAX_TRACEBACK_FILTERING=off

echo "Environment setup complete."