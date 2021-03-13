import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import csv
from pycuda.compiler import SourceModule
################KERNELS####################
from cudaFunctions import multiply_them, accumulate, scale, subtract, \
     accumulateCovs
###########################################
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
SAMPLE_BLOCK = (28,1,1)
SAMPLE_GRID = (28,1)
COV_BLOCK = (784,1,1)
COV_GRID = (784,1)
trainingData = 'data/train.csv'
testData = 'data/test.csv'

def checkMeans(cudaMeans,trainSamples,trainLabels,streams):
    npMeans = np.zeros((NUM_CLASSES,len(trainSamples[0])),dtype=np.float32)
    counts = np.array([0 for _ in range(NUM_CLASSES)],dtype=np.float32)
    #calclate means here
    for s,l in zip(trainSamples,trainLabels):
        npMeans[l]+=s
        counts[l]+=1
    for i in range(NUM_CLASSES):
        npMeans[i] /= counts[i]

    correct = True
    means = []
    for i in range(NUM_CLASSES):
        means.append(cudaMeans[i].get_async(stream=streams[i]))
    for c,m in zip(means,npMeans):
        if(np.linalg.norm(m-c) > .001):
            correct = False
        
    if not correct:
        print("ERROR - MEANS")

def checkCovs(samples,labels,cudaCovs,streams):

    correct = True
    covs = []
    for i in range(NUM_CLASSES):
        covs.append(cudaCovs[i].get_async(stream=streams[i]))
    data = [ [] for _ in range(NUM_CLASSES)]
    for label,sample in zip(labels,samples):
        data[label].append(sample)
    npCovs = []
    for i in range(NUM_CLASSES):
        npCovs.append(np.cov(np.transpose(data[i])))

    for c,m in zip(covs,npCovs):
        print(c)
        if(np.linalg.norm(m-c) > .001):
            print(np.linalg.norm(m-c))
            correct = False
    if not correct:
        print("ERROR - COVS")

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
       accumulate(rets[trainLabels[i]],vectors[i],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[trainLabels[i]])


    #now, normalize the output
    #synchronize streams, as this is the completion of the mean calculation
    for i in range(NUM_CLASSES):
        scale(rets[i],scalars[i],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[i])


    return rets     
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
        
def InitCovsGPU(streams=None,dim=784):
    assert not streams==None
    
    cudaCovs = []
    for i in range(NUM_CLASSES):
        cudaCovs.append(gpuarray.to_gpu_async(np.zeros((dim,dim)),stream=streams[i]))
    return cudaCovs


def computeCov(streams,covs,means,vectors,labels,scalars):
    #we are going to calculate each covarience serpeately
    #first we just get shifted vectors
    for i in range(len(labels)):
        subtract(vectors[i],means[labels[i]],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[labels[i]])


    for i in range(len(labels)):
        accumulateCovs(covs[labels[i]],vectors[i],block=COV_BLOCK,grid=COV_GRID,stream=streams[labels[i]])

    checks = []
    for i in range(NUM_CLASSES):
        checks.append(covs[i].get_async(stream=streams[i]))
    for i in range(10):
        for j in range(784):
            print(checks[i][j][j])
        print('next cov now')

    
"""
    for i in range(NUM_CLASSES):
        scale(covs[labels[i]],scalars[i],block=COV_BLOCK,grid=COV_GRID,stream=streams[i])

    return covs
        """

