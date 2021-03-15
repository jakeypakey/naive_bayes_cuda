#!/usr/bin/env python 
import pycuda.autoinit
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
    out[i] = out[i] - sub[i];
} 
""")

cmod = SourceModule("""
__global__ void accumulateCovs(float *cov, float *in)
{
    const int i = threadIdx.x;
    const int j = blockIdx.x;

    cov[i+j*blockDim.x] = cov[i+j*blockDim.x] + in[i]*in[j];
        
} 
""")

emod = SourceModule("""
__global__ void extractInvDiag(float *prec, float *cov)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(cov[i+i*blockDim.x] !=0)
        prec[i] = 1.0/cov[i+i*blockDim.x];
} 
""")




multiply_them = fmod.get_function("multiply_them")
accumulate = amod.get_function("accumulate")
scale = dmod.get_function("scale")
subtract = smod.get_function("subtract")
accumulateCovs = cmod.get_function("accumulateCovs")
extractInvDiag = emod.get_function("extractInvDiag")
