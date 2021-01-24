
!/bin/bash
#SBATCH --job-name=
SBATCH --workdir=/master/home/kmleung
SBATCH --output=host.out
SBATCH --error=host.err
 
python runFileSLURM.py