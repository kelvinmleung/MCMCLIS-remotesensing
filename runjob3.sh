#!/bin/bash
#SBATCH --job-name=mcm3
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob3.out
#SBATCH --error=runjob3.err
#SBATCH --exclusive
 
python runFileG216.py