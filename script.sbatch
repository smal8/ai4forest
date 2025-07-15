#!/bin/bash
#SBATCH --job-name=ai4for-1run

#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/htc/smalipati/output/%j
#SBATCH --exclude=htc-gpu[017,020-023,037,038] # Only A4 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=gpu  # Specify the desired partition, e.g. gpu or big
#SBATCH --time=0-01:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --ntasks=1

echo 'Getting node information'
date;hostname;id;pwd

codebase_path="/home/htc/smalipati/ai4forest_sushanth"

# Switch to the cwd
cd $codebase_path
git checkout test3

# Acquire node information
echo 'Getting node information'
date;hostname;id;pwd

# Setup environment
echo 'Activating virtual environment'
bash
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"
micromamba activate pytorch
which python
nvidia-smi

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Set the wandb directory variable'
export WANDB_DIR=/home/htc/$USER/SCRATCH

# Setup temporary directory (this avoids the problem that wandb fills up the git repo with 100k files)
if [ -d "/scratch/local" ]; then
  # Create /scratch/local/$USER and /scratch/local/$USER/tmp if they do not exist
  mkdir -p "/scratch/local/$USER/tmp"

  # Set TMPDIR to the temporary directory
  export TMPDIR="/scratch/local/$USER/tmp"
fi

python main.py --debug