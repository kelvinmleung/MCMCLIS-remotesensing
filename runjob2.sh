#!/bin/bash
#SBATCH --job-name=mcm2
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob2.out
#SBATCH --error=runjob2.err
#SBATCH --exclusive
 
python runFileG21.py