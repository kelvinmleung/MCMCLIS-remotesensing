#!/bin/bash
#SBATCH --job-name=atmrunH2
#SBATCH --workdir=/master/home/kmleung/JPLproject/MCMCLIS-remotesensing
#SBATCH --output=atmrunH2.out
#SBATCH --error=atmrunH2.err
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

mv atmRuns/runFileA7.py .
python runFileA7.py
mv runFileA7.py atmRuns/

mv atmRuns/runFileA9.py .
python runFileA9.py
mv runFileA9.py atmRuns/