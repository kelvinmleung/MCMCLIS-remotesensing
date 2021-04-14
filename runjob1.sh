#!/bin/bash
#SBATCH --job-name=mcm1
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob1.out
#SBATCH --error=runjob1.err
#SBATCH --exclusive
 
mv atmRuns/runFileA1.py .
python runFileA1.py
mv runFileA1.py atmRuns/