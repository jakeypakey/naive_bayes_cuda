import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import csv
from pycuda.compiler import SourceModule
################KERNELS####################
from cudaFunctions import multiply_them, accumulate, normalize
###########################################
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
from pythonFunctions import readData, computeMeans

##################################################
##toy example from pyCuda
##################################################
a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

dest = np.zeros_like(a)
multiply_them(
        cuda.Out(dest), cuda.In(a), cuda.In(b),
        block=(400,1,1), grid=(1,1))
##################################################
##################################################


##start here




#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test
#start with JUST integers

cudaMeans = computeMeans(trainLabels,trainSamples)    

npMeans = np.zeros((NUM_CLASSES,len(trainSamples[0])),dtype=np.float32)
counts = np.array([0 for _ in range(NUM_CLASSES)],dtype=np.float32)

for s,l in zip(trainSamples,trainLabels):
    npMeans[l]+=s
    counts[l]+=1
for i in range(NUM_CLASSES):
    npMeans[i] /= counts[i]

for c,m in zip(cudaMeans,npMeans):
    if not np.array_equal(c,m):
        print('UH OH')



