#!/bin/bash
#SBATCH --job-name=mcmc
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob.out
#SBATCH --error=runjob.err
#SBATCH --exclusive
 
python runFile.py