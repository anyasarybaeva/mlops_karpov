#!/bin/sh
export MLFLOW_RUN_ID=$(python scripts/create_run.py)
python scripts/prepare.py
python scripts/split.py
python scripts/train.py
python scripts/evaluate.py
