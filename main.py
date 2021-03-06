
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import csv
from pycuda.compiler import SourceModule
from cudaFunctions import multiply_them
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
trainingData = 'data/train.csv'
testData = 'data/test.csv'

a = np.random.randn(400).astype(np.float32)
b = np.random.randn(400).astype(np.float32)

dest = np.zeros_like(a)
multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400,1,1), grid=(1,1))


##start here

def readData():
    with open(trainingData,'r') as fin:
        it = csv.reader(fin,delimiter=',')
        #skip first one
        next(it)

        train = [row for row in it]
        trainLabels = np.array([row[0] for row in train],dtype=np.float32)
        trainSamples = np.array([row[1:] for row in train],dtype=np.float32)
        train  = trainLabels,trainSamples)

    with open(testData,'r') as fin:
        it = csv.read(fin,delimeter)
        next(it)

        test = [row for row in it]
        testLabels = np.array([row[0] for row in test],dtype=np.float32)
        testSamples = np.array([row[1:] for row in test],dtype=np.float32)
        test  = (testLabels,testSamples)

    return (train,test)



#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test

#init to zero, means
means = np.zeros((NUM_CLASSES,len(labels)),dtype=np.float32)

#malloc
cudaRet = cuda.mem_alloc(ret.nbytes)
cudaLabels = cuda.mem_alloc(trainLabels.nbytes)
cudaSamples = cuda.mem_alloc(trainSamples.nbytes)

#mem copy to GPU
cuda.memcpy_htod(cudaRet,means)
cuda.memcpy_htod(cudaLabels,trainLabels)
cuda.memcpy_htod(cudaSamples,trainSamples)

#call function here..

#bring mean back to RAM







cudaMean(ret,drv.In(samples),drv.In(labels),block=(,,),grid(,))
