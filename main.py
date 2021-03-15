import numpy as np
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
from pythonFunctions import readData, computeParams, checkParams, sendDataToGPU

#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test


#move data over
streams,cudaMeans,cudaVectors,cudaScalars = sendDataToGPU(trainSamples,trainLabels)
#calculate params
cudaCovs, cudaMeans = computeParams(streams,cudaMeans,cudaVectors,cudaScalars,trainLabels)
checkParams(cudaMeans,cudaCovs,trainSamples,trainLabels,streams)


#exit()
#calculate covarience

