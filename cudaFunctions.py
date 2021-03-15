#!/usr/bin/env python import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

fmod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

amod = SourceModule("""
__global__ void accumulate(float *out, float *in)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    out[i] = out[i] + in[i];
} 
""")

dmod = SourceModule("""
__global__ void scale(float *out, float *scalar)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    out[i] = out[i]*scalar[0];

} 
""")

smod = SourceModule("""
__global__ void subtract(float *out, float *sub)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
//   printf("index: %d,values: out[i]: %f sub[i]: %f \\n",i,out[i],sub[i]);
    out[i] = out[i] - sub[i];
} 
""")

cmod = SourceModule("""
//#include <stdio.h>
__global__ void accumulateCovs(float *cov, float *in)
{
    const int i = threadIdx.x;
    const int j = blockIdx.x;

//    printf("indices: %d, %d,   values: in[i]: %f in[j]: %f \\n",i,j,in[i],in[j]);

    cov[i+j*blockDim.x] = cov[i+j*blockDim.x] + in[i]*in[j];
        //cov[i+j*blockDim.x] = 1;

        
} 
""")







multiply_them = fmod.get_function("multiply_them")
accumulate = amod.get_function("accumulate")
scale = dmod.get_function("scale")
subtract = smod.get_function("subtract")
accumulateCovs = cmod.get_function("accumulateCovs")
