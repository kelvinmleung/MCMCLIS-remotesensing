#!/bin/bash
#SBATCH --job-name=atmrunH15
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=atmrunH15.out
#SBATCH --error=atmrunH15.err
#SBATCH --exclusive
 
python atmRuns/runFileA1.py
python atmRuns/runFileA3.py
python atmRuns/runFileA5.py
python atmRuns/runFileA7.py
python atmRuns/runFileA9.py