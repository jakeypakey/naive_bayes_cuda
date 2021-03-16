import numpy as np
NUM_CLASSES = 10
WARPS_PER_BLOCK=4
from pythonFunctions import readData, computeParams, checkParams, sendDataToGPU, getPrecision, makePredictions

#read and unpack data
train,test = readData()
trainLabels, trainSamples = train
testLabels, testSamples = test


#move data over
streams,cudaMeans,cudaVectors,cudaScalars = sendDataToGPU(trainSamples,trainLabels)
#calculate params
cudaCovs, cudaMeans = computeParams(streams,cudaMeans,cudaVectors,cudaScalars,trainLabels)


cudaPrecisions = getPrecision(cudaCovs,streams)

#checkParams(cudaMeans,cudaCovs,trainSamples,trainLabels,cudaPrecision,streams)

streams, cudaVectorsTest = sendDataToGPU(testSamples,testLabels,streams,train=False)

results = makePredictions(cudaVectorsTest,cudaMeans,cudaPrecisions,streams)

#for the testing purposes, we make the assumption that all covaraince matrices are diagonal.
#first we need to convert our covariance matrices to diagonal precision matrices

errors = 0
for actual,predict in zip(testLabels,results):
    if actual != predict:
        errors+=1
print("Accuracy: {}".format((len(testLabels)-c)/len(testLabels)))



