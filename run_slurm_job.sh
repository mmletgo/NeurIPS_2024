#!/bin/bash
#SBATCH --job-name=smalltransformer_w_sigma          # This is the job name that will appear when you search for your jobs on slurm
#SBATCH --output=slurm/smalltransformer_w_sigma.out    # This is the directory of the output file where slurm will show you the output of your run. 
                                        # You have to create the slurm directory yourself.
#SBATCH --time=30:00:00                 # Time limit hrs:min:sec (set your appropriate time for your job, you can cancel earlier)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:a100-80:1            # Request for 1 a100 40GB GPU (see link on available GPUs)
#SBATCH --mem=64GB                      # GPU VRAM needed 

# Activate your conda environment
source ~/.bashrc
conda activate dl

# Run the Python Script
python -u smalltransformer_w_sigma.py