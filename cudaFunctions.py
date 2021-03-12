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








multiply_them = fmod.get_function("multiply_them")
accumulate = amod.get_function("accumulate")
scale = dmod.get_function("scale")
