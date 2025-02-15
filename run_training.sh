#!/bin/bash
#SBATCH --partition=msc_appbio
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --job-name=retinal_train

# Initialize conda
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate retinal_project

# Change to project directory
cd ~/retinal_project2/src
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Print debug information
echo "Current directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo "Conda env: $CONDA_PREFIX"
python --version

# Run training
python main.py \
   --data_dir ../balanced_processed_data \
   --output_dir ../outputs \
   --batch_size 16 \
   --epochs 100 \
   --learning_rate 1e-4 \
   --patience 10 \
   --num_workers 4