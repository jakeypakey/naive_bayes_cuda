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
__global__ void normalize(float *out, float divisor)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    out[i] = out[i]/divisor;
} 
""")







multiply_them = fmod.get_function("multiply_them")
accumulate = amod.get_function("accumulate")
normalize = dmod.get_function("normalize")
