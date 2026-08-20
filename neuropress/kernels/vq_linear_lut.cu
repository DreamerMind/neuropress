/*
 * vq_linear_lut.cu — LUT-based VQ linear gather kernel.
 *
 * Two-phase VQ forward pass:
 *
 *   Phase 1 (Python / cuBLAS):
 *     lut[m, g, k] = dot( x[m, g*VL:(g+1)*VL],  centroids[k] )
 *     Shape: [M, n_groups, NC]  — computed as one (M*n_groups, VL) × (VL, NC) GEMM.
 *
 *   Phase 2 (this kernel):
 *     y[m, j] = sum_g  lut[m, g, packed_indices[j*n_groups + g]]
 *
 * Each CUDA thread handles one (m, output_neuron) pair.  The inner loop
 * over n_groups unpacks the index and does a scalar L2-cached lookup.
 *
 * Why separate phases?
 *   The LUT GEMM is a small, standard dense matmul that cuBLAS handles
 *   optimally.  The gather phase has an irregular (data-dependent) access
 *   pattern over the LUT that benefits from a custom kernel.
 *
 * When to use:
 *   Most beneficial when out_features >> NC (large-vocab lm_head) so the
 *   LUT entries are reused many times.  For small layers (out_features ~ NC)
 *   the WMMA GEMM path is preferred.
 *
 * Supported: bits ∈ {4, 8, 10, 12, 14, 15, 16}.  Bit-stream layout is LSB-first,
 * identical to IndexPacking in vq_packing.py and vq_linear_wmma.cu.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Index unpacking — identical to vq_linear_wmma.cu (LSB-first bit-stream).
// ---------------------------------------------------------------------------

template <int BITS>
__device__ __forceinline__ int unpack_idx(const uint8_t* __restrict__ p, int flat);

template <>
__device__ __forceinline__ int unpack_idx<4>(const uint8_t* p, int flat) {
    const uint8_t b = __ldg(&p[flat >> 1]);
    return (flat & 1) ? (int)(b >> 4) : (int)(b & 0xF);
}

template <>
__device__ __forceinline__ int unpack_idx<8>(const uint8_t* p, int flat) {
    return (int)__ldg(&p[flat]);
}

template <>
__device__ __forceinline__ int unpack_idx<10>(const uint8_t* p, int flat) {
    const int bit_start  = flat * 10;
    const int byte_start = bit_start >> 3;
    const int bit_off    = bit_start & 7;
    const uint16_t val   = (uint16_t)__ldg(&p[byte_start])
                         | ((uint16_t)__ldg(&p[byte_start + 1]) << 8);
    return (int)((val >> bit_off) & 0x3FFu);
}

template <>
__device__ __forceinline__ int unpack_idx<12>(const uint8_t* p, int flat) {
    const int half = flat >> 1;
    if (flat & 1) {
        return (int)(__ldg(&p[half * 3 + 1]) >> 4)
             | ((int)__ldg(&p[half * 3 + 2]) << 4);
    } else {
        return (int)__ldg(&p[half * 3])
             | (((int)(__ldg(&p[half * 3 + 1]) & 0xF)) << 8);
    }
}

// 14-bit: NC=16384.  max window = bit_off(7)+14 = 21 bits → 3 bytes, uint32_t.
template <>
__device__ __forceinline__ int unpack_idx<14>(const uint8_t* p, int flat) {
    const int bit_start  = flat * 14;
    const int byte_start = bit_start >> 3;
    const int bit_off    = bit_start & 7;
    const uint32_t val = (uint32_t)__ldg(&p[byte_start])
                       | ((uint32_t)__ldg(&p[byte_start + 1]) << 8)
                       | ((uint32_t)__ldg(&p[byte_start + 2]) << 16);
    return (int)((val >> bit_off) & 0x3FFFu);
}

// 15-bit: NC=32768.  max window = bit_off(7)+15 = 22 bits → 3 bytes, uint32_t.
template <>
__device__ __forceinline__ int unpack_idx<15>(const uint8_t* p, int flat) {
    const int bit_start  = flat * 15;
    const int byte_start = bit_start >> 3;
    const int bit_off    = bit_start & 7;
    const uint32_t val = (uint32_t)__ldg(&p[byte_start])
                       | ((uint32_t)__ldg(&p[byte_start + 1]) << 8)
                       | ((uint32_t)__ldg(&p[byte_start + 2]) << 16);
    return (int)((val >> bit_off) & 0x7FFFu);
}

// 16-bit: NC=65536.  Exactly 2 bytes per index — alias buffer as uint16_t.
template <>
__device__ __forceinline__ int unpack_idx<16>(const uint8_t* p, int flat) {
    return (int)__ldg(&reinterpret_cast<const uint16_t*>(p)[flat]);
}

// ---------------------------------------------------------------------------
// Gather kernel
//
// Grid  : (ceil(N / BLOCK), M)
// Block : (BLOCK,)
//
// Thread (blockIdx.y, blockIdx.x * BLOCK + threadIdx.x) → output (m, j).
// Inner loop over n_groups: unpack index, scalar LUT lookup, float32 add.
// LUT is [M, n_groups, NC] float32; at NC=1024, n_groups=160 → 640 KB for
// M=1, comfortably L2-resident on any modern GPU.
// ---------------------------------------------------------------------------

#define BLOCK 256

template <int BITS>
__global__ void _lut_gather(
    const float*   __restrict__ lut,     // [M, n_groups, NC]  float32
    const uint8_t* __restrict__ packed,  // [N * n_groups]     packed
    float*         __restrict__ output,  // [M, N]             float32
    int N, int n_groups, int NC
) {
    const int j = (int)(blockIdx.x) * BLOCK + (int)(threadIdx.x);
    const int m = (int)(blockIdx.y);
    if (j >= N) return;

    const float* lut_m = lut + (long)m * n_groups * NC;
    float acc = 0.0f;

    const int base = j * n_groups;
    for (int g = 0; g < n_groups; ++g) {
        const int cidx = unpack_idx<BITS>(packed, base + g);
        acc += lut_m[(long)g * NC + cidx];
    }

    output[(long)m * N + j] = acc;
}

// ---------------------------------------------------------------------------
// Launcher — dispatches on bits.
// ---------------------------------------------------------------------------

static void dispatch_lut_gather(
    const float*   lut,
    const uint8_t* packed,
    float*         output,
    int M, int N, int n_groups, int NC, int bits,
    cudaStream_t stream
) {
    const dim3 grid(((unsigned)N + BLOCK - 1) / BLOCK, (unsigned)M);
    const dim3 block(BLOCK);

#define LAUNCH(B) \
    _lut_gather<B><<<grid, block, 0, stream>>>(lut, packed, output, N, n_groups, NC)

    switch (bits) {
        case  4: LAUNCH( 4); break;
        case  8: LAUNCH( 8); break;
        case 10: LAUNCH(10); break;
        case 12: LAUNCH(12); break;
        case 14: LAUNCH(14); break;
        case 15: LAUNCH(15); break;
        case 16: LAUNCH(16); break;
        default:
            AT_ERROR("vq_lut_gather: unsupported bits=", bits,
                     " (expected 4, 8, 10, 12, 14, 15, or 16)");
    }
#undef LAUNCH
}

// ---------------------------------------------------------------------------
// Python binding
// ---------------------------------------------------------------------------

torch::Tensor vq_lut_gather(
    torch::Tensor lut,      // [M, n_groups, NC] float32 CUDA
    torch::Tensor packed,   // 1-D uint8 CUDA
    int bits,
    int N                   // out_features
) {
    TORCH_CHECK(lut.is_cuda(),    "lut must be a CUDA tensor");
    TORCH_CHECK(packed.is_cuda(), "packed must be a CUDA tensor");
    TORCH_CHECK(lut.dtype() == torch::kFloat32, "lut must be float32");
    TORCH_CHECK(lut.dim() == 3, "lut must be 3-D [M, n_groups, NC]");

    const int M        = (int)lut.size(0);
    const int n_groups = (int)lut.size(1);
    const int NC       = (int)lut.size(2);

    auto output = torch::empty({M, N}, lut.options());

    dispatch_lut_gather(
        lut.data_ptr<float>(),
        packed.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        M, N, n_groups, NC, bits,
        at::cuda::getCurrentCUDAStream()
    );

    return output;  // [M, N] float32
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vq_lut_gather", &vq_lut_gather,
          "LUT gather phase of VQ linear: y[m,j] = sum_g lut[m,g,idx[j,g]]");
}
