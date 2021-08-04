#!/bin/bash
#SBATCH --job-name=mcm4
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob4.out
#SBATCH --error=runjob4.err
#SBATCH --exclusive
 
python runFileH14.py