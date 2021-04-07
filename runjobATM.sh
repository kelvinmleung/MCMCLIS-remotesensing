#!/bin/bash
#SBATCH --job-name=atmrunH15
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=atmrunH15.out
#SBATCH --error=atmrunH15.err
#SBATCH --exclusive

mv atmRuns/runFileA1.py .
python atmRuns/runFileA1.py
mv runFileA1.py atmRuns/

mv atmRuns/runFileA3.py .
python atmRuns/runFileA3.py
mv runFileA3.py atmRuns/

mv atmRuns/runFileA5.py .
python atmRuns/runFileA5.py
mv runFileA5.py atmRuns/

mv atmRuns/runFileA7.py .
python atmRuns/runFileA7.py
mv runFileA7.py atmRuns/

mv atmRuns/runFileA9.py .
python atmRuns/runFileA9.py
mv runFileA9.py atmRuns/