// Use half-precision PTX directly so the fixture needs no CCCL headers.
__device__ unsigned short half_add(unsigned short left, unsigned short right)
{
    unsigned short result;
    asm("add.rn.f16 %0, %1, %2;" : "=h"(result) : "h"(left), "h"(right));
    return result;
}

// All tensors use the same packed HWC8 storage. Channels are padded by the caller.
extern "C" __global__ void half_io(unsigned short const* x, unsigned short const* residual,
    unsigned short* output, int count, int mode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    output[index] = mode == 0 ? half_add(x[index], x[index])
        : mode == 1 ? half_add(x[index], residual[index]) : residual[index];
}
