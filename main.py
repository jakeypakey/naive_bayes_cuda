import numpy as np
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
from pythonFunctions import readData, computeMeans, sendDataToGPU

#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test

npMeans = np.zeros((NUM_CLASSES,len(trainSamples[0])),dtype=np.float32)
counts = np.array([0 for _ in range(NUM_CLASSES)],dtype=np.float32)
#calclate means here
for s,l in zip(trainSamples,trainLabels):
    npMeans[l]+=s
    counts[l]+=1
for i in range(NUM_CLASSES):
    npMeans[i] /= counts[i]

#and here
streams,cudaMeans,cudaVectors,cudaScalars = sendDataToGPU(trainSamples,trainLabels)
cudaMeans = computeMeans(streams,cudaMeans,cudaVectors,cudaScalars,trainLabels)

correct = True
for c,m in zip(cudaMeans,npMeans):
    if(np.linalg.norm(m-c) > .001):
        correct = False
        
if not correct:
    print("ERROR - MEANS")




