import numpy as np
from scipy.io import loadmat
from fsplit.filesplit import Filesplit


class FileProcessing:


    def __init__(self):
        print('\n')

    def loadWavelength(self, wvFile='setup/data/wavelengths.txt'):
        wvl, wv, wvr = np.loadtxt(wvFile).T
        self.wv = wv * 1000

    def loadReflectance(self, refFile ='setup/data/beckmanlawn/insitu.txt'):
        wvRaw, refRaw, refnoise = np.loadtxt(refFile).T
        self.ref = np.interp(self.wv, wvRaw, refRaw)

    def loadRadiance(self, datamatfile='setup/data/beckmanlawn/ang20171108t184227_data_v2p11_BeckmanLawn.mat'):
        datamatfile = datamatfile
        mat = loadmat(datamatfile)
        self.radiance = mat['meas'][0]
    
    def getFiles(self):
        return self.wv, self.ref, self.radiance

    def splitFile(self, filename, output):
        fs = Filesplit()
        def split_cb(f, s):
            print("file: {0}, size: {1}".format(f, s))
        fs.split(file=filename, split_size=9000000000, output_dir=output, callback=split_cb)

    def mergeFile(self, inputdir):
        fs = Filesplit()
        def merge_cb(f, s):
            print("file: {0}, size: {1}".format(f, s))
        fs.merge(input_dir=inputdir, callback=merge_cb)


        