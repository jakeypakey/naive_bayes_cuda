import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
import numpy as np
import csv
from pycuda.compiler import SourceModule
################KERNELS####################
from cudaFunctions import multiply_them, accumulate, scale, subtract, \
     accumulateCovs, extractInvDiag
###########################################
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
SAMPLE_BLOCK = (28,1,1)
SAMPLE_GRID = (28,1)
#SAMPLE_BLOCK = (2,1,1)
#SAMPLE_GRID = (1,1)
VECTOR_LEN = 784
#VECTOR_LEN = 2 COV_BLOCK = (VECTOR_LEN,1,1)
COV_GRID = (VECTOR_LEN,1)
COV_BLOCK = (VECTOR_LEN,1,1)
trainingData = 'data/train.csv'
testData = 'data/test.csv'

def checkParams(cudaMeans,cudaCovs,trainSamples,trainLabels,cudaPrecision,streams):
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
        if(np.linalg.norm(m-c)/np.linalg.norm(m) > .0001):
            break
            correct = False
    if not correct:
        print("ERROR - MEANS")
        return


    npData = [[] for _ in range(NUM_CLASSES)]
    for s,l in zip(trainSamples,trainLabels):
        npData[l].append(s)

    npCov = []
    for i in range(NUM_CLASSES):
        npCov.append(np.cov(np.transpose(npData[i])))

    covs = []
    for i in range(NUM_CLASSES):
        covs.append(cudaCovs[i].get_async(stream=streams[i]))

    for c,m in zip(covs,npCov):
        if (np.linalg.norm(m-c)/np.linalg.norm(m) > .001):
            break
            correct = False

    if not correct:
        print("ERROR - COVS")

    prec = []
    for i in range(NUM_CLASSES):
        prec.append(cudaPrecision[i].get_async(stream=streams[i]))

    npPrec = [-1*np.ones(VECTOR_LEN,dtype=np.float32) for _ in range(NUM_CLASSES)]

    for i in range(NUM_CLASSES):
        for j in range(VECTOR_LEN):
            if not npCov[i][j][j] == 0:
                npPrec[i][j] = 1/npCov[i][j][j]

    for c,m in zip(prec,npPrec):
        if (np.linalg.norm(m-c)/np.linalg.norm(m) > .001):
            break
            correct = False

    if not correct:
        print("ERROR - PRECISION")




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

def computeParams(streams,rets,vectors,scalars,trainLabels):
    #create streams, one per class is a natural way to partition
    for i in range(len(trainLabels)):
       accumulate(rets[trainLabels[i]],vectors[i],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[trainLabels[i]])

    #now, normalize the output
    for i in range(NUM_CLASSES): 
        scale(rets[i],scalars[i],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[i])


    #now we will shift the vectors for cov
    for i in range(len(trainLabels)):
        subtract(vectors[i],rets[trainLabels[i]],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[trainLabels[i]])

    #allocate covs here     
    cudaCovs = []
    for i in range(NUM_CLASSES):
        cudaCovs.append(gpuarray.to_gpu_async(np.zeros((VECTOR_LEN,VECTOR_LEN),dtype=np.float32),stream=streams[i]))

    for i in range(len(trainLabels)):
        accumulateCovs(cudaCovs[trainLabels[i]],vectors[i],block=COV_BLOCK,grid=COV_GRID,stream=streams[trainLabels[i]])
    
    for i in range(NUM_CLASSES):
        scale(cudaCovs[i],scalars[i],block=COV_BLOCK,grid=COV_GRID,stream=streams[i])


    return cudaCovs, rets     
        
def InitCovsGPU(streams=None):
    assert not streams==None
    
    cudaCovs = []
    for i in range(NUM_CLASSES):
        cudaCovs.append(gpuarray.to_gpu_async(np.zeros((VECTOR_LEN,VECTOR_LEN),dtype=np.float32),stream=streams[i]))
        
    return cudaCovs


def sendDataToGPU(samples,labels,streams=None,train=True):
    #create streams
    if streams==None:
        streams = []
        for _ in range(NUM_CLASSES):
            streams.append(cuda.Stream())


    vectors = [] 
    count = np.array([[0] for _ in range(NUM_CLASSES)],dtype=np.float32)
    for s,l in zip(samples,labels):
        vectors.append(gpuarray.to_gpu_async(s,stream=streams[l]))
        count[l]+=1
    if train:
        means = []
        scalars = []
        for i in range(NUM_CLASSES):
            count[i] = 1/count[i]
            scalars.append(gpuarray.to_gpu_async(count[i],stream=streams[i]))
            means.append(gpuarray.to_gpu_async(np.zeros_like(samples[0]),stream=streams[i]))
        return (streams,means,vectors,scalars)
    else:
        return streams,vectors

def getPrecision(covs,streams):
    gpuarray
    precision = []
    for i in range(NUM_CLASSES):
        precision.append(gpuarray.to_gpu_async(-1*np.ones(VECTOR_LEN),stream=streams[i]))
        extractInvDiag(precision[-1],covs[i],block=SAMPLE_BLOCK,grid=SAMPLE_GRID,stream=streams[i])
    return precision
