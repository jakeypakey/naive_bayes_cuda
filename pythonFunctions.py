import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import csv
from pycuda.compiler import SourceModule
################KERNELS####################
from cudaFunctions import multiply_them, accumulate, scale
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

def computeMeans(streams,rets,vectors,scalars,trainLabels):
    #create streams, one per class is a natural way to partition
    for i in range(len(trainLabels)):
       accumulate(rets[trainLabels[i]],vectors[i],block=(28,1,1),grid=(28,1),stream=streams[trainLabels[i]])


    #now, normalize the output
    #synchronize streams, as this is the completion of the mean calculation
    for i in range(NUM_CLASSES):
        scale(rets[i],scalars[i],block=(28,1,1),grid=(28,1),stream=streams[i])

    means = []
    for i in range(NUM_CLASSES):
        means.append(rets[i].get_async(stream=streams[i]))

    return means     
def sendDataToGPU(samples,labels,streams=None):
    #create streams
    if streams==None:
        streams = []
        for _ in range(NUM_CLASSES):
            streams.append(cuda.Stream())

    means = []
    for i in range(NUM_CLASSES):
        means.append(gpuarray.to_gpu_async(np.zeros_like(samples[0]),stream=streams[i]))

    vectors = [] 
    count = np.array([[0] for _ in range(NUM_CLASSES)],dtype=np.float32)
    for s,l in zip(samples,labels):
        vectors.append(gpuarray.to_gpu_async(s,stream=streams[l]))
        count[l]+=1

    scalars = []
    for i in range(NUM_CLASSES):
        count[i] = 1/count[i]
        scalars.append(gpuarray.to_gpu_async(count[i],stream=streams[i]))
    return (streams,means,vectors,scalars)
        
#def computeCov(streams,means,vectors,priors):

