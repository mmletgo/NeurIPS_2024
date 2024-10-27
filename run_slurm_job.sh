#!/bin/bash
#SBATCH --job-name=NoiseReduction          # This is the job name that will appear when you search for your jobs on slurm
#SBATCH --output=slurm/NoiseReduction.out    # This is the directory of the output file where slurm will show you the output of your run. 
                                        # You have to create the slurm directory yourself.
#SBATCH --time=30:00:00                 # Time limit hrs:min:sec (set your appropriate time for your job, you can cancel earlier)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:a100-80:1            # Request for 1 a100 40GB GPU (see link on available GPUs)
#SBATCH --mem=64GB                      # GPU VRAM needed 

# Activate your conda environment
source ~/.bashrc
conda activate dl

# Run the Python Script
python -u NoiseReduction.py