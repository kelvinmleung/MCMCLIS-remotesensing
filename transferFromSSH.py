import subprocess
subprocess.call(["rsync", "-chavzP", "--stats", "--delete", "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/fromCluster"])

subprocess.call(["mv", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/fromCluster/MCMC_x.npy", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC"])
subprocess.call(["mv", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/fromCluster/logpos.npy", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC"])
subprocess.call(["rmdir", "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/fromCluster"])



