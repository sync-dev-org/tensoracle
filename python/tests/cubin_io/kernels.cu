#include <cuda_fp16.h>

// All tensors use the same packed HWC8 storage. Channels are padded by the caller.
extern "C" __global__ void half_io(__half const* x, __half const* residual,
    __half* output, int count, int mode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = mode == 0 ? __hadd(x[index], x[index])
        : mode == 1 ? __hadd(x[index], residual[index]) : residual[index];
}
