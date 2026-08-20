/*
 * rms_norm.cu — single-pass fused RMSNorm kernel.
 *
 * Replaces HuggingFace's unfused implementation (pow → mean → rsqrt → mul ×2,
 * five separate CUDA kernels) with a single block-per-row kernel.
 *
 * Algorithm (one block per row m):
 *   1. Threads stride over x[m, :] to accumulate partial sum_sq.
 *   2. Warp-level reduction (shfl_xor), then block-level via shared mem.
 *   3. Thread 0 computes scale = rsqrtf(sum_sq / H + eps).
 *   4. All threads write out[m, i] = x[m, i] * scale * weight[i].
 *
 * Supports float16 and bfloat16. H can be any size; threads stride in
 * steps of BLOCK when H > BLOCK.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define BLOCK    256
#define N_WARPS  (BLOCK / 32)   // 8 warps per block

template <typename scalar_t>
__global__ void _rms_norm_kernel(
    const scalar_t* __restrict__ x,       // [M, H]
    const scalar_t* __restrict__ weight,  // [H]
    scalar_t*       __restrict__ out,     // [M, H]
    int H, float eps
) {
    const long m = blockIdx.x;
    const scalar_t* xm  = x   + m * (long)H;
          scalar_t* outm = out + m * (long)H;

    // ------------------------------------------------------------------
    // Phase 1: accumulate sum of squares
    // ------------------------------------------------------------------
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < H; i += BLOCK) {
        float xi = (float)xm[i];
        ss += xi * xi;
    }

    // Warp-level reduction
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, off);

    // Store warp totals in shared memory
    __shared__ float smem[N_WARPS];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    if (lane == 0) smem[warp] = ss;
    __syncthreads();

    // Final reduction + scale computation (warp 0 only)
    __shared__ float scale_sh;
    if (warp == 0) {
        ss = (lane < N_WARPS) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            ss += __shfl_xor_sync(0xffffffffu, ss, off);
        if (lane == 0)
            scale_sh = rsqrtf(ss / (float)H + eps);
    }
    __syncthreads();
    const float scale = scale_sh;

    // ------------------------------------------------------------------
    // Phase 2: write scaled output
    // ------------------------------------------------------------------
    for (int i = (int)threadIdx.x; i < H; i += BLOCK)
        outm[i] = (scalar_t)((float)xm[i] * scale * (float)weight[i]);
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

torch::Tensor rms_norm(
    torch::Tensor x,       // [M, H]  half or bfloat16, CUDA
    torch::Tensor weight,  // [H]     same dtype as x
    double eps
) {
    TORCH_CHECK(x.is_cuda(),      "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(),     "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(),"weight must be contiguous");
    TORCH_CHECK(x.dim() == 2,     "x must be 2-D [M, H]");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1),
                "weight must be 1-D [H] matching x.size(1)");
    TORCH_CHECK(x.dtype() == weight.dtype(),
                "x and weight must share dtype");

    const int M = (int)x.size(0);
    const int H = (int)x.size(1);
    auto out = torch::empty_like(x);

    const dim3 grid(M);
    const dim3 block(BLOCK);
    const auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(), "rms_norm",
        [&] {
            _rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                H, (float)eps
            );
        }
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm,
          "Fused RMSNorm: out[m,i] = x[m,i] * rsqrt(mean(x[m,:]²)+eps) * weight[i]");
}
