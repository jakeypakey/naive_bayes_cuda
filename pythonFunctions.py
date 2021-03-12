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
trainingData = 'data/train.csv'
testData = 'data/test.csv'
def readData():
    with open(trainingData,'r') as fin:
        it = csv.reader(fin,delimiter=',')
        #skip first one
        next(it)

        train = [row for row in it]
        trainLabels = np.array([row[0] for row in train],dtype=np.int32)
        trainSamples = np.array([row[1:] for row in train],dtype=np.float32)
        train  = (trainLabels,trainSamples)

    with open(testData,'r') as fin:
        it = csv.reader(fin,delimiter=',')
        next(it)

        test = [row for row in it]
        testLabels = np.array([row[0] for row in test],dtype=np.int32)
        testSamples = np.array([row[1:] for row in test],dtype=np.float32)
        test  = (testLabels,testSamples)

    return (train,test)

def computeMeans(trainLabels,trainSamples):
    #create streams, one per class is a natural way to partition
    streams = []
    counts = np.array([0 for _ in range(NUM_CLASSES)],dtype=np.float32)
    for i in range(NUM_CLASSES):
        streams.append(cuda.Stream())


    #create means of zero, initialize, and move to GPU
    means = np.zeros((NUM_CLASSES,len(trainSamples[0])),dtype=np.float32)
    rets = []
    for i in range(NUM_CLASSES):
        rets.append(gpuarray.to_gpu_async(means[i],stream=streams[i]))

    ## accumulate everything on GPU
    for vector,label in zip(trainSamples,trainLabels):
        current = gpuarray.to_gpu_async(vector,stream=streams[label])
        counts[label]+=1
        ##THIS IS 28thread warp need to CHANGE once functional
        accumulate(rets[label],current,block=(28,1,1),grid=(28,1),stream=streams[label])


    #now, normalize the output
    #synchronize streams, as this is the completion of the mean calculation
    for i in range(NUM_CLASSES):
        normalize(rets[i],counts[i],block=(28,1,1),grid=(28,1),stream=streams[i])
        streams[i].synchronize()

    means = []
    for i in range(NUM_CLASSES):
        means.append(rets[i].get_async(stream=streams[i]))

    return means     
