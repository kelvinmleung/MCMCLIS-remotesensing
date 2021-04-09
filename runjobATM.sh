#!/bin/bash
#SBATCH --job-name=atmrunH25
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=atmrunH25.out
#SBATCH --error=atmrunH25.err
#SBATCH --exclusive

mv atmRuns/runFileA1.py .
python runFileA1.py
mv runFileA1.py atmRuns/

mv atmRuns/runFileA3.py .
python runFileA3.py
mv runFileA3.py atmRuns/

mv atmRuns/runFileA5.py .
python runFileA5.py
mv runFileA5.py atmRuns/