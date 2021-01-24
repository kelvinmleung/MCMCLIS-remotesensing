
#!/bin/bash
#SBATCH --job-name=testJob
#SBATCH --workdir=/master/home/kmleung
#SBATCH --output=testJob.out
#SBATCH --error=testJob.err
 
python runFileSLURM.py