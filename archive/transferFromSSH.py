import subprocess

filename = 'C1_initMAP_rank175.tgz'

remoteDir = "kmleung@hypersonic.mit.edu:/master/home/kmleung/JPLproject/results/MCMC/"
localDir = "/Users/KelvinLeung/Documents/JPLproject/results/MCMC/"

subprocess.call(["rsync", "-chavzP", "--stats", "--delete", remoteDir + filename, localDir])


'''
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/N1 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/B9 /

./dropbox_uploader.sh upload ../JPLproject/results/MCMC/G1 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/G2 /
./dropbox_uploader.sh upload ../JPLproject/results/MCMC/G4 /


./dropbox_uploader.sh upload ../JPLproject/results/MCMC/G4 /


./dropbox_uploader.sh upload ../JPLproject/results/MCMC/H2A5 /


./dropbox_uploader.sh download /C7 ../JPLproject/results/MCMC
./dropbox_uploader.sh download /C8 ../JPLproject/results/MCMC
./dropbox_uploader.sh download /B7 ../JPLproject/results/MCMC
./dropbox_uploader.sh download /B8 ../JPLproject/results/MCMC

'''


## scrap code for zipping plots

# zip the plots
# listFiles = []
# listFiles = listFiles + ['reflMean.png', 'reflError.png', 'atmMean.png', 'atmError.png','reflVar.png', 'atmVar.png', 'errorRelCov.png', 'trace.png', 'autocorr.png', 'logpos.png', 'acceptance.png', 'runtime.txt']
# numPairs = int(len(indSet) / 2)
# for i in range(numPairs):
#     listFiles = listFiles + ['2D_' + str(indSet[2*i]) + '-' + str(indSet[2*i+1]) + '.png']
# subprocess.call(['tar', '--directory', setup.mcmcDir, '-zcvf', '../results/MCMC/' + mcmcfolder + '.tgz'] + listFiles)









