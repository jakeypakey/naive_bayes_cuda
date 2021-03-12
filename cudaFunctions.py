import pycuda.autoinit
import pycuda.driver as drv
import numpy as np

from pycuda.compiler import SourceModule
mmod = SourceModule("""
__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}
""")

mod = SourceModule("""
__global__ void accumulate(int *out, int *in)
{
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    out[i] = out[i] + in[i];
} 
""")




multiply_them = mmod.get_function("multiply_them")
accumulate = mod.get_function("accumulate")
