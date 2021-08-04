#!/bin/bash
#SBATCH --job-name=mcm0
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob1.out
#SBATCH --error=runjob1.err
#SBATCH --exclusive
 
python runFileH10.py