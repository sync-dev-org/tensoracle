// Caller-owned CUDA C++ kernels. The tensoracle producer supplies compile constants.
#ifndef TF32_CONTROL
#define TF32_CONTROL 0
#endif
__device__ float input_value(float value) {
#if TF32_CONTROL
    unsigned int bits = __float_as_uint(value);
    if ((bits & 0x7f800000U) != 0x7f800000U)
        bits = (bits + 0xfffU + ((bits >> 13) & 1U)) & 0xffffe000U;
    return __uint_as_float(bits);
#else
    return value;
#endif
}
extern "C" __global__ void dot_kernel(const float *x_ptr, const float *weight_ptr, float *out_ptr,
                                      int rows) {
    for (int index = threadIdx.x; index < BLOCK_M * BLOCK_N; index += blockDim.x) {
        int row = blockIdx.x * BLOCK_M + index / BLOCK_N;
        int column = index % BLOCK_N;
        if (row >= rows)
            continue;
        float sum = 0.0f;
        for (int depth = 0; depth < BLOCK_K; ++depth)
            sum += input_value(x_ptr[row * BLOCK_K + depth]) *
                   input_value(weight_ptr[depth * BLOCK_N + column]);
        out_ptr[row * BLOCK_N + column] = sum;
    }
}
extern "C" __global__ void scale_kernel(const float *x_ptr, float *out_ptr, float factor,
                                        int count) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < count)
        out_ptr[offset] = x_ptr[offset] * factor;
}
extern "C" __global__ void bias_kernel(const float *x_ptr, const float *bias_ptr, float *out_ptr,
                                       int count) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < count)
        out_ptr[offset] = x_ptr[offset] + bias_ptr[offset % BLOCK_N];
}
