#!/bin/bash
#SBATCH --job-name=mcm0
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob0.out
#SBATCH --error=runjob0.err
#SBATCH --exclusive
 
python runFileH3S_test.py