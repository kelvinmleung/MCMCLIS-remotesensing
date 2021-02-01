#!/bin/bash
#SBATCH --job-name=MCMC
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=mcmc.out
#SBATCH --error=mcmc.err
 
python runFileSLURM.py