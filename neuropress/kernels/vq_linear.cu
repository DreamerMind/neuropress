/*
 * Fused VQ-weight linear kernel.
 *
 * Each output element y[m, n] is computed as:
 *
 *   y[m, n] = sum_{g=0}^{G-1} sum_{v=0}^{VL-1}
 *                 x[m, g*VL + v] * centroids[idx(n,g), v]
 *
 * where idx(n, g) is the codebook index for output row n and vector group g,
 * unpacked from the packed_indices buffer.
 *
 * Index storage layout (matches IndexPacking in vq_packing.py)
 * -------------------------------------------------------------------
 *   bits=8: packed_indices[n * n_groups + g] = idx  (one uint8 per index)
 *   bits=4: packed_indices[(n * n_groups + g) / 2]  (two 4-bit indices
 *            per byte, LSB-first: even flat index -> low nibble,
 *                                  odd flat index  -> high nibble)
 *
 * Template parameters
 * -------------------
 *   BITS     : bits per index -- 4 or 8.
 *   VL       : vector length (elements per codebook entry) -- 4 or 8.
 *   scalar_t : __half, __nv_bfloat16, or float (input/output element type).
 *
 * Thread mapping -- one warp per output element y[m, n]:
 *   grid  : (ceil(N / WARPS_PER_BLOCK), M)
 *   block : (WARP_SIZE=32, WARPS_PER_BLOCK=8)
 *   Each lane handles g = lane, lane+32, lane+64, ...
 *   Inner VL dimension is unrolled at compile time.
 *   Final dot-product reduction via __shfl_down_sync.
 *
 * Centroid table layout
 * ---------------------
 *   centroids[c * VL + v]  -- row-major float16.
 *   For typical configs (NC=256, VL=8) the full table is 4 KiB -- fits in
 *   L1 cache and is broadcast efficiently by the GPU's constant cache.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <optional>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

namespace {

// ---------------------------------------------------------------------------
// Type-conversion helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<__half>(const __half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <> __device__ __forceinline__ float to_f32<float>(const float v) {
  return v;
}

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <> __device__ __forceinline__ float from_f32<float>(float v) {
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

/* Unpack one index from packed_indices.
 * flat = n * n_groups + g  (the linear position of this index).
 *
 * Bit-stream layout (LSB-first, identical to IndexPacking in vq_packing.py):
 *   bits=4:  two 4-bit indices per byte (nibble-packed).
 *   bits=8:  one 8-bit index per byte.
 *   bits=12: two 12-bit indices per 3 bytes.
 *            even flat → byte[flat*3/2],      low nibble of byte[flat*3/2+1]
 *            odd  flat → high nibble of byte[(flat>>1)*3+1], byte[(flat>>1)*3+2]
 */
template <int BITS>
__device__ __forceinline__ int unpack_idx(const uint8_t *__restrict__ packed,
                                          int flat) {
  if constexpr (BITS == 8) {
    return (int)__ldg(&packed[flat]);
  } else if constexpr (BITS == 4) {
    const uint8_t byte = __ldg(&packed[flat >> 1]);
    return (flat & 1) ? (int)(byte >> 4) : (int)(byte & 0xF);
  } else {
    // BITS == 12: 2 indices per 3 bytes, LSB-first bit-stream.
    const int half = flat >> 1;
    if (flat & 1) {
      // odd: high nibble of byte[half*3+1] gives bits 0-3,
      //      byte[half*3+2] gives bits 4-11.
      return (int)(__ldg(&packed[half * 3 + 1]) >> 4) |
             ((int)__ldg(&packed[half * 3 + 2]) << 4);
    } else {
      // even: byte[half*3] gives bits 0-7,
      //       low nibble of byte[half*3+1] gives bits 8-11.
      return (int)__ldg(&packed[half * 3]) |
             (((int)(__ldg(&packed[half * 3 + 1]) & 0xF)) << 8);
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Kernel: group-iteration with compile-time VL unrolling
// ---------------------------------------------------------------------------

template <int BITS, int VL, typename scalar_t>
__global__ void
_vq_linear(const scalar_t *__restrict__ x,            // [M, K]
           const uint8_t *__restrict__ packed_indices, // [N * n_groups] packed
           const __half *__restrict__ centroids,       // [num_centroids, VL]
           const float *__restrict__ bias,             // [N] or nullptr
           scalar_t *__restrict__ y,                   // [M, N]
           int M, int N, int n_groups) {
  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;

  float acc = 0.f;
  const int row_base = n * n_groups;
  const size_t x_base = (size_t)m * (size_t)n_groups * VL;

  // Each lane iterates over g = lane, lane+32, lane+64, ...
  // VL FMAs per iteration, unrolled at compile time.
  for (int g = threadIdx.x; g < n_groups; g += WARP_SIZE) {
    const int cidx = unpack_idx<BITS>(packed_indices, row_base + g);
    const __half *cent = &centroids[(size_t)cidx * VL];
    const scalar_t *xp = &x[x_base + (size_t)g * VL];

#pragma unroll
    for (int v = 0; v < VL; v++) {
      acc += to_f32(__ldg(&xp[v])) * __half2float(__ldg(&cent[v]));
    }
  }

  acc = warp_reduce_sum(acc);
  if (threadIdx.x == 0) {
    if (bias)
      acc += bias[n];
    y[(size_t)m * N + n] = from_f32<scalar_t>(acc);
  }
}

// ---------------------------------------------------------------------------
// Specialisation for VL=4 with half2 vectorized loads (fp16 inputs only)
// ---------------------------------------------------------------------------

template <int BITS>
__global__ void
_vq_linear_h2(const __half *__restrict__ x,            // [M, K]
              const uint8_t *__restrict__ packed_indices, // packed
              const __half *__restrict__ centroids,       // [NC, 4]
              const float *__restrict__ bias,             // [N] or nullptr
              __half *__restrict__ y,                     // [M, N]
              int M, int N, int n_groups) {
  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;

  float acc = 0.f;
  const int row_base = n * n_groups;
  const size_t x_base = (size_t)m * (size_t)n_groups * 4;

  for (int g = threadIdx.x; g < n_groups; g += WARP_SIZE) {
    const int cidx = unpack_idx<BITS>(packed_indices, row_base + g);

    // Vectorized half2 loads: 2 loads instead of 4 scalar loads.
    const half2 c01 =
        __ldg(reinterpret_cast<const half2 *>(&centroids[(size_t)cidx * 4]));
    const half2 c23 = __ldg(
        reinterpret_cast<const half2 *>(&centroids[(size_t)cidx * 4 + 2]));
    const half2 x01 =
        __ldg(reinterpret_cast<const half2 *>(&x[x_base + (size_t)g * 4]));
    const half2 x23 =
        __ldg(reinterpret_cast<const half2 *>(&x[x_base + (size_t)g * 4 + 2]));

    // Accumulate in fp32 for numerical stability.
    acc += __half2float(__low2half(x01)) * __half2float(__low2half(c01));
    acc += __half2float(__high2half(x01)) * __half2float(__high2half(c01));
    acc += __half2float(__low2half(x23)) * __half2float(__low2half(c23));
    acc += __half2float(__high2half(x23)) * __half2float(__high2half(c23));
  }

  acc = warp_reduce_sum(acc);
  if (threadIdx.x == 0) {
    if (bias)
      acc += bias[n];
    y[(size_t)m * N + n] = __float2half(acc);
  }
}

// ---------------------------------------------------------------------------
// Dispatch helper
// ---------------------------------------------------------------------------

static inline dim3 _vq_grid(int N, int M) {
  return dim3((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, (unsigned)M);
}
static const dim3 _VQ_BLOCK(WARP_SIZE, WARPS_PER_BLOCK);

template <typename scalar_t>
static void _vq_launch(const scalar_t *x, const uint8_t *packed,
                       const __half *centroids, const float *bias, scalar_t *y,
                       int M, int N, int n_groups, int vector_len, int bits,
                       cudaStream_t stream) {
  const dim3 grid = _vq_grid(N, M);

  // Dispatch on (bits, vector_len) with compile-time template parameters.
  // Unsupported combinations fall back to the generic kernel.
#define VQ_LAUNCH(B, V)                                                        \
  _vq_linear<B, V, scalar_t>                                                   \
      <<<grid, _VQ_BLOCK, 0, stream>>>(x, packed, centroids, bias, y, M, N,   \
                                       n_groups)

  if (vector_len == 4) {
    switch (bits) {
    case 4:
      VQ_LAUNCH(4, 4);
      return;
    case 8:
      VQ_LAUNCH(8, 4);
      return;
    case 12:
      VQ_LAUNCH(12, 4);
      return;
    }
  } else if (vector_len == 8) {
    switch (bits) {
    case 4:
      VQ_LAUNCH(4, 8);
      return;
    case 8:
      VQ_LAUNCH(8, 8);
      return;
    case 12:
      VQ_LAUNCH(12, 8);
      return;
    }
  }
#undef VQ_LAUNCH

  TORCH_CHECK(false, "vq_linear: unsupported bits=", bits,
              " vector_len=", vector_len, ".  Must be bits={4,8,12}, VL={4,8}.");
}

// Specialised half2 dispatch for fp16 + VL=4.
static void _vq_launch_h2(const __half *x, const uint8_t *packed,
                           const __half *centroids, const float *bias,
                           __half *y, int M, int N, int n_groups, int bits,
                           cudaStream_t stream) {
  const dim3 grid = _vq_grid(N, M);
  switch (bits) {
  case 4:
    _vq_linear_h2<4>
        <<<grid, _VQ_BLOCK, 0, stream>>>(x, packed, centroids, bias, y, M, N,
                                         n_groups);
    return;
  case 8:
    _vq_linear_h2<8>
        <<<grid, _VQ_BLOCK, 0, stream>>>(x, packed, centroids, bias, y, M, N,
                                         n_groups);
    return;
  case 12:
    _vq_linear_h2<12>
        <<<grid, _VQ_BLOCK, 0, stream>>>(x, packed, centroids, bias, y, M, N,
                                         n_groups);
    return;
  }
  TORCH_CHECK(false, "vq_linear: unsupported bits=", bits);
}

// ---------------------------------------------------------------------------
// Torch-facing function
// ---------------------------------------------------------------------------

torch::Tensor vq_linear(torch::Tensor x, torch::Tensor packed_indices,
                        torch::Tensor centroids, int bits, int vector_len,
                        std::optional<torch::Tensor> bias_opt) {
  TORCH_CHECK(x.is_cuda() && packed_indices.is_cuda() && centroids.is_cuda(),
              "vq_linear: all tensors must be on a CUDA device.");
  TORCH_CHECK(x.is_contiguous(), "vq_linear: x must be contiguous.");
  TORCH_CHECK(packed_indices.is_contiguous(),
              "vq_linear: packed_indices must be contiguous.");
  TORCH_CHECK(centroids.is_contiguous(),
              "vq_linear: centroids must be contiguous.");
  TORCH_CHECK(x.dim() == 2, "vq_linear: x must be 2-D [M, K].");
  TORCH_CHECK(centroids.dtype() == torch::kFloat16,
              "vq_linear: centroids must be float16.");
  TORCH_CHECK(bits == 4 || bits == 8 || bits == 12,
              "vq_linear: bits must be 4, 8, or 12, got ", bits, ".");
  TORCH_CHECK(vector_len == 4 || vector_len == 8,
              "vq_linear: vector_len must be 4 or 8, got ", vector_len, ".");

  const int M = x.size(0);
  const int K = x.size(1);
  TORCH_CHECK(K % vector_len == 0, "vq_linear: K=", K,
              " not divisible by vector_len=", vector_len);
  const int n_groups = K / vector_len;

  // Derive N (out_features) from packed_indices size.
  const int total_bytes = packed_indices.numel();
  // General formula: floor(total_bytes * 8 / bits).
  // Correct for bits=4 (×2), bits=8 (×1), and bits=12 (×2/3 with padding).
  const int total_indices = (int)(((long long)total_bytes * 8) / bits);
  TORCH_CHECK(total_indices % n_groups == 0,
              "vq_linear: packed_indices size mismatch.");
  const int N = total_indices / n_groups;

  auto y = torch::empty({M, N}, x.options());

  const float *bias_ptr = nullptr;
  torch::Tensor bias_f32;
  if (bias_opt.has_value()) {
    bias_f32 = bias_opt.value().to(torch::kFloat32).contiguous();
    bias_ptr = bias_f32.data_ptr<float>();
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const __half *cent_ptr =
      reinterpret_cast<const __half *>(centroids.data_ptr<at::Half>());
  const uint8_t *pack_ptr = packed_indices.data_ptr<uint8_t>();

  if (x.dtype() == torch::kFloat16) {
    const __half *x_ptr =
        reinterpret_cast<const __half *>(x.data_ptr<at::Half>());
    __half *y_ptr = reinterpret_cast<__half *>(y.data_ptr<at::Half>());

    // Use half2 vectorized kernel for VL=4 (most common path).
    if (vector_len == 4) {
      _vq_launch_h2(x_ptr, pack_ptr, cent_ptr, bias_ptr, y_ptr, M, N,
                     n_groups, bits, stream);
    } else {
      _vq_launch<__half>(x_ptr, pack_ptr, cent_ptr, bias_ptr, y_ptr, M, N,
                         n_groups, vector_len, bits, stream);
    }
  } else if (x.dtype() == torch::kBFloat16) {
    // Native bf16 — no conversion needed.
    const __nv_bfloat16 *x_ptr =
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>());
    __nv_bfloat16 *y_ptr =
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr<at::BFloat16>());
    _vq_launch<__nv_bfloat16>(x_ptr, pack_ptr, cent_ptr, bias_ptr, y_ptr, M, N,
                              n_groups, vector_len, bits, stream);
  } else {
    TORCH_CHECK(x.dtype() == torch::kFloat32,
                "vq_linear: x must be float16, bfloat16, or float32.");
    _vq_launch<float>(x.data_ptr<float>(), pack_ptr, cent_ptr, bias_ptr,
                      y.data_ptr<float>(), M, N, n_groups, vector_len, bits,
                      stream);
  }

  return y;
}

// ---------------------------------------------------------------------------
// Pybind11 bindings
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Dequantize: materialise W[N, K] fp16 from packed indices + centroid table.
//
// Handles any bit width in [4, 20] via a generic bitstream unpacker.
// Requires packed_indices to have ≥4 bytes of zero-padding past the last
// valid byte (safe for the 4-byte window read used by the generic path).
//
// Thread mapping: identical to _vq_linear — one warp per output row n.
// Write pattern: lane g writes W[n, g*VL:(g+1)*VL] → coalesced at VL=4 or 8.
// ---------------------------------------------------------------------------

// Generic bitstream unpacker: index `flat` occupies bits [flat*bits, (flat+1)*bits).
// Reads at most 4 bytes; caller must guarantee ≥4 bytes of padding past end.
__device__ __forceinline__ int
_unpack_any(const uint8_t *__restrict__ packed, int flat, int bits) {
  const int bit_start = flat * bits;
  const int byte_start = bit_start >> 3;
  const int bit_off = bit_start & 7;
  uint32_t val = (uint32_t)__ldg(&packed[byte_start])
               | ((uint32_t)__ldg(&packed[byte_start + 1]) << 8)
               | ((uint32_t)__ldg(&packed[byte_start + 2]) << 16);
  // For bits > 16 the span can exceed 24 bits; load a 4th byte.
  // Branch is uniform across a warp (bits is the same for all threads).
  if (bits > 16)
    val |= ((uint32_t)__ldg(&packed[byte_start + 3]) << 24);
  return (int)((val >> bit_off) & ((1u << bits) - 1));
}

template <int VL>
__global__ void
_dequantize_vq_weight(const uint8_t *__restrict__ packed,
                      const __half *__restrict__ centroids, // [NC, VL]
                      __half *__restrict__ W,               // [N, n_groups*VL]
                      int N, int n_groups, int bits) {
  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  if (n >= N)
    return;

  const int row_base = n * n_groups;
  __half *__restrict__ Wn = W + (size_t)n * n_groups * VL;

  for (int g = threadIdx.x; g < n_groups; g += WARP_SIZE) {
    const int cidx = _unpack_any(packed, row_base + g, bits);
    const __half *cent = centroids + (size_t)cidx * VL;
    __half *dst = Wn + (size_t)g * VL;
#pragma unroll
    for (int v = 0; v < VL; v++)
      dst[v] = __ldg(&cent[v]);
  }
}

// ---------------------------------------------------------------------------
// Torch-facing dequantize function
// ---------------------------------------------------------------------------

torch::Tensor dequantize_vq_weight(torch::Tensor packed_indices,
                                   torch::Tensor centroids, int bits,
                                   int vector_len, int out_features,
                                   int in_features) {
  TORCH_CHECK(packed_indices.is_cuda() && centroids.is_cuda(),
              "dequantize_vq_weight: tensors must be on CUDA.");
  TORCH_CHECK(packed_indices.is_contiguous() && centroids.is_contiguous(),
              "dequantize_vq_weight: tensors must be contiguous.");
  TORCH_CHECK(centroids.dtype() == torch::kFloat16,
              "dequantize_vq_weight: centroids must be float16.");
  TORCH_CHECK(vector_len == 4 || vector_len == 8,
              "dequantize_vq_weight: vector_len must be 4 or 8, got ",
              vector_len);
  TORCH_CHECK(bits >= 4 && bits <= 20,
              "dequantize_vq_weight: bits must be in [4, 20], got ", bits);
  TORCH_CHECK(in_features % vector_len == 0,
              "dequantize_vq_weight: in_features not divisible by vector_len.");

  const int N = out_features;
  const int n_groups = in_features / vector_len;

  auto W = torch::empty({N, in_features}, centroids.options());

  const dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1);
  const auto stream = at::cuda::getCurrentCUDAStream();

  const uint8_t *pack_ptr = packed_indices.data_ptr<uint8_t>();
  const __half *cent_ptr =
      reinterpret_cast<const __half *>(centroids.data_ptr<at::Half>());
  __half *W_ptr = reinterpret_cast<__half *>(W.data_ptr<at::Half>());

  if (vector_len == 4)
    _dequantize_vq_weight<4>
        <<<grid, _VQ_BLOCK, 0, stream>>>(pack_ptr, cent_ptr, W_ptr, N,
                                         n_groups, bits);
  else
    _dequantize_vq_weight<8>
        <<<grid, _VQ_BLOCK, 0, stream>>>(pack_ptr, cent_ptr, W_ptr, N,
                                         n_groups, bits);
  return W;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_linear", &vq_linear,
        "Fused VQ-weight linear (index unpack + centroid gather + matmul).",
        py::arg("x"), py::arg("packed_indices"), py::arg("centroids"),
        py::arg("bits"), py::arg("vector_len"), py::arg("bias") = py::none());
  m.def("dequantize_vq_weight", &dequantize_vq_weight,
        "Materialise fp16 weight matrix from packed VQ indices + centroids. "
        "packed_indices must have ≥4 bytes of zero-padding past the last byte.",
        py::arg("packed_indices"), py::arg("centroids"), py::arg("bits"),
        py::arg("vector_len"), py::arg("out_features"), py::arg("in_features"));
}
