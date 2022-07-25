#!/bin/sh
#SBATCH -N 1
#SBATCH --job-name=AudioMatch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=40GB
#SBATCH -o .logs/%x_%A_%a.out
#SBATCH -e .logs/%x_%A_%a.err
#SBATCH --array=2000-2124
##SBATCH --gres=gpu:1
##SBATCH --account=conf-cvpr-2021.11.23-ghanembs

# 2124
echo `hostname`
source activate torch1.3

FILES_PATH=/ibex/scratch/projects/c2134/audiovault_data/downloads
OUT_PATH=/ibex/scratch/projects/c2134/audiovault_data/alignment_results

python align_signals.py --files_path $FILES_PATH\
                        --out_path $OUT_PATH\
                        --ID ${SLURM_ARRAY_TASK_ID}