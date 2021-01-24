#!/bin/bash
#SBATCH --job-name=testJob
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=testJob.out
#SBATCH --error=testJob.err
 
python runFileSLURM.py