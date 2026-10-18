#!/bin/bash
# One reference network on one task, one GPU (msbench/train.py):
#
#     sbatch scripts/train.sbatch <task> <model> <seed>        e.g.  sbatch scripts/train.sbatch T2-1mm unet 0
#
# Set DATASET (the v1.0.0 folder) and RUNS (where the run folders go) for the cluster; both on scratch. A job that
# reaches the time limit can be submitted again with the same arguments: it continues from last.pt.
# The lines marked CLUSTER are the ones to adapt (partition, QoS, environment).
#SBATCH --job-name=msbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=...                               # CLUSTER
##SBATCH --qos=...                                     # CLUSTER

set -euo pipefail
TASK=$1 MODEL=$2 SEED=$3
: "${DATASET:?set DATASET to the dataset/v1.0.0 folder}"
: "${RUNS:?set RUNS to the folder of the run folders}"
: "${HOURS:=22}"                                       # stop training before this; scoring takes the rest

# source activate msbench                              # CLUSTER: an environment with torch, h5py, pyarrow, scipy

cd "$SLURM_SUBMIT_DIR"                                 # submitted from the repository folder (logs/ must exist)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m msbench.train "$DATASET" "$RUNS/$TASK/$MODEL-s$SEED" --task "$TASK" --model "$MODEL" --seed "$SEED" \
    --uncertainty references/label_uncertainty.parquet --hours "$HOURS" --workers "$SLURM_CPUS_PER_TASK" "${@:4}"
