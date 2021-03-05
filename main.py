
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import csv
from pycuda.compiler import SourceModule
from cudaFunctions import multiply_them
trainingData = 'train.csv'
testData = 'test.csv'

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
        train = [t for t in it]
    print(train)

print(readData())
