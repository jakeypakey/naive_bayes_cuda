
import pycuda.autoinit
import pycuda.driver as cuda
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
        cuda.Out(dest), cuda.In(a), cuda.In(b),
        block=(400,1,1), grid=(1,1))


##start here

def readData():
    with open(trainingData,'r') as fin:
        it = csv.reader(fin,delimiter=',')
        #skip first one
        next(it)

        train = [row for row in it]
        trainLabels = np.array([row[0] for row in train],dtype=np.int32)
        trainSamples = np.array([row[1:] for row in train],dtype=np.int32)
        train  = (trainLabels,trainSamples)

    with open(testData,'r') as fin:
        it = csv.reader(fin,delimiter=',')
        next(it)

        test = [row for row in it]
        testLabels = np.array([row[0] for row in test],dtype=np.int32)
        testSamples = np.array([row[1:] for row in test],dtype=np.int32)
        test  = (testLabels,testSamples)

    return (train,test)



#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test
#start with JUST integers

#init to zero, means
means = np.zeros((NUM_CLASSES,len(trainSamples[0])),dtype=np.int32)
#cudaRet = cuda.mem_alloc(means.nbytes)
#cudaSamples = cuda.mem_alloc(trainSamples.nbytes)

#create streams
streams = []
inputs = []
rets = []
counts = [0 for _ in range(NUM_CLASSES)]
for i in range(NUM_CLASSES):
    streams.append(cuda.Stream())
    inputs.append(cuda.mem_alloc(trainSamples[0].nbytes))
    rets.append(cuda.mem_alloc(trainSamples[0].nbytes))


#send all means over
for i in range(NUM_CLASSES):
    cuda.memcpy_htod_async(rets[i],means[i],stream=streams[i])

## accumulate everything
for vector,label in zip(trainSamples,trainLabels):
    counts[label-1]+=1
    cuda.memcpy_htod_async(inputs[label-1],vector,stream=streams[label-1])
    #temporary, set back to 32 later
    accumulate(rets[label-1],inputs[label-1],block=(28,1,1),grid=(28,1),stream=streams[label-1])
    

for i in range(NUM_CLASSSES):
    cuda.memcpy_dtoh_async(means[i],ret[i],stream=streams[i])

for stream in streams:
    stream.synchronize()
for i in range(NUM_CLASSES):
    print(means[i])




    




#malloc
print(trainSamples.shape)
print(trainLabels.shape)

#mem copy to GPU
cuda.memcpy_htod(cudaRet,means)
cuda.memcpy_htod(cudaLabels,trainLabels)
cuda.memcpy_htod(cudaSamples,trainSamples)

#call function here..
#cudaMean(ret,drv.In(samples),drv.In(labels),block=(,,),grid(,))

#bring mean back to RAM







