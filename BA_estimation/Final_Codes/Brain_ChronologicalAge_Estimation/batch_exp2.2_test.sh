#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=exp2.2_train
#SBATCH --output=job_name%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --mem-per-gpu=18G
#SBATCH --gpus=1
#SBATCH --qos=batch
 
# Activate everything you need
module load cuda/10.1
conda  activate  ba_vis_tf2
# Run your python code
python3 /usrhomes/g009/shashanks/Master_Thesis_BA_DeepVis/BA_estimation/Final_Codes/Brain_ChronologicalAge_Estimation/oasis_train_3D_slices_with_gender_v2.py --exp_name exp2.2 --mode test
