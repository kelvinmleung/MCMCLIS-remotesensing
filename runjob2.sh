#!/bin/bash
#SBATCH --job-name=mcm2
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob2.out
#SBATCH --error=runjob2.err
#SBATCH --exclusive
 
mv atmRuns/runFileA3.py .
python runFileA3.py
mv runFileA3.py atmRuns/