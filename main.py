import numpy as np
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
from pythonFunctions import readData, computeMeans, sendDataToGPU, computeCov, \
     checkMeans, InitCovsGPU, checkCovs

#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test


#move data over
streams,cudaMeans,cudaVectors,cudaScalars = sendDataToGPU(trainSamples,trainLabels)
#calculate means
cudaMeans = computeMeans(streams,cudaMeans,cudaVectors,cudaScalars,trainLabels)

#calculate covarience
cudaCovs = InitCovsGPU(streams)
cudaCovs = computeCov(streams,cudaCovs,cudaMeans,cudaVectors,trainLabels,cudaScalars)

checkCovs(trainSamples,trainLabels,cudaCovs,streams)

