#!/bin/bash
#SBATCH --job-name=mcma
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=runjob2.out
#SBATCH --error=runjob2.err
 
python runFile.py