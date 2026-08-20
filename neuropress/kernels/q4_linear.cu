/*
 * Generic fused group-quantized-weight linear kernel.
 *
 * Supports any bit-width packed by BitPacking (1, 2, 4, 8) and both
 * symmetric (scale-only) and asymmetric (scale + zero_point) schemes.
 *
 * Template parameters
 * -------------------
 * NBITS    : bits per value (1 | 2 | 4 | 8).  Determines:
 *              VPB  = 8 / NBITS  (values packed per byte)
 *              MASK = (1 << NBITS) - 1
 * ASYM     : true  → per-group float16 zero_point read from memory.
 *            false → symmetric zero_point = 1 << (NBITS-1) baked in as
 * constant. scalar_t : __half or float (input/output element type).
 *
 * Weight storage
 * --------------
 *   w_pack[n, k / VPB]  — packed byte for row n, value index k.
 *   shift = (k % VPB) * NBITS
 *   nibble = (w_pack[...] >> shift) & MASK
 *   w_float = (nibble - zero_point) * scale[n, k / group_size]
 *
 * Thread mapping — one warp per output element y[m, n]:
 *   grid  : (⌈N / WARPS_PER_BLOCK⌉, M)
 *   block : (WARP_SIZE=32, WARPS_PER_BLOCK=8)
 *   Each lane handles values i = lane, lane+32, lane+64, …
 *   Final reduction via __shfl_down_sync.
 *
 * For NBITS=4, group_size=32 (the common Q4_0 case):
 *   – All 32 lanes share the same scale per iteration → broadcast.
 *   – Weight accesses cover 16 consecutive bytes per iteration → coalesced.
 *   – Input accesses cover 32 consecutive elements per iteration → coalesced.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <optional>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

// ---------------------------------------------------------------------------
// Helper: uniform float conversion for __half and float
// ---------------------------------------------------------------------------

namespace {

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32<__half>(const __half v) {
  return __half2float(v);
}
template <> __device__ __forceinline__ float to_f32<float>(const float v) {
  return v;
}

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ __half from_f32<__half>(float v) {
  return __float2half(v);
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

} // namespace

// ---------------------------------------------------------------------------
// Generic kernel
// ---------------------------------------------------------------------------

template <int NBITS, bool ASYM, typename scalar_t>
__global__ void
_q_linear(const scalar_t *__restrict__ x,        // [M, K]
          const uint8_t *__restrict__ w_pack,    // [N, K / VPB]
          const __half *__restrict__ scale,      // [N * (K / group_size)]
          const __half *__restrict__ zero_point, // [same] or nullptr when !ASYM
          const float *__restrict__ bias,        // [N] or nullptr
          scalar_t *__restrict__ y,              // [M, N]
          int M, int N, int K, int group_size) {
  constexpr int VPB = 8 / NBITS;
  constexpr int MASK = (1 << NBITS) - 1;

  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  const int m = blockIdx.y;
  if (n >= N || m >= M)
    return;

  const int n_groups = K / group_size;
  const int K_bytes = K / VPB; // packed bytes per weight row
  float acc = 0.f;

  for (int i = threadIdx.x; i < K; i += WARP_SIZE) {
    // Unpack: locate byte and extract the slot for value i.
    const int byte_idx = i / VPB;
    const int shift = (i % VPB) * NBITS;
    const uint8_t p = __ldg(&w_pack[(size_t)n * K_bytes + byte_idx]);
    const int q = (int)(p >> shift) & MASK;

    // Dequantize.
    const int g = i / group_size;
    const int sidx = n * n_groups + g;
    const float sc = __half2float(__ldg(&scale[sidx]));

    float w;
    if constexpr (ASYM) {
      const float zp = __half2float(__ldg(&zero_point[sidx]));
      w = ((float)q - zp) * sc;
    } else {
      constexpr float SYM_ZP = (float)(1 << (NBITS - 1));
      w = ((float)q - SYM_ZP) * sc;
    }

    acc += w * to_f32(__ldg(&x[(size_t)m * K + i]));
  }

  acc = warp_reduce_sum(acc);
  if (threadIdx.x == 0) {
    if (bias)
      acc += bias[n];
    y[(size_t)m * N + n] = from_f32<scalar_t>(acc);
  }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

static inline dim3 _grid(int N, int M) {
  return dim3((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, (unsigned)M);
}
static const dim3 _BLOCK(WARP_SIZE, WARPS_PER_BLOCK);

// Launch the right template instantiation for given (nbits, asym, fp16).
// zero_point may be nullptr when !asym.
template <typename scalar_t>
static void _launch(const scalar_t *x, const uint8_t *w, const __half *sc,
                    const __half *zp, const float *bias, scalar_t *y, int M,
                    int N, int K, int group_size, int nbits, bool asym,
                    cudaStream_t stream) {
  const dim3 grid = _grid(N, M);

#define LAUNCH(NB, AS)                                                         \
  _q_linear<NB, AS, scalar_t><<<grid, _BLOCK, 0, stream>>>(                    \
      x, w, sc, zp, bias, y, M, N, K, group_size)

  switch (nbits) {
  case 1:
    asym ? LAUNCH(1, true) : LAUNCH(1, false);
    break;
  case 2:
    asym ? LAUNCH(2, true) : LAUNCH(2, false);
    break;
  case 4:
    asym ? LAUNCH(4, true) : LAUNCH(4, false);
    break;
  case 8:
    asym ? LAUNCH(8, true) : LAUNCH(8, false);
    break;
  default:
    TORCH_CHECK(false, "q_linear: unsupported nbits=", nbits,
                ".  Must be one of {1, 2, 4, 8}.");
  }
#undef LAUNCH
}

// ---------------------------------------------------------------------------
// Torch-facing function
// ---------------------------------------------------------------------------

torch::Tensor q_linear(torch::Tensor x, torch::Tensor w_pack,
                       torch::Tensor scale, int nbits, int group_size,
                       std::optional<torch::Tensor> zero_point_opt,
                       std::optional<torch::Tensor> bias_opt) {
  // --- Validation ---
  TORCH_CHECK(x.is_cuda() && w_pack.is_cuda() && scale.is_cuda(),
              "q_linear: x, w_pack, and scale must be on CUDA");
  TORCH_CHECK(w_pack.dtype() == torch::kUInt8,
              "q_linear: w_pack must be uint8");
  TORCH_CHECK(scale.dtype() == torch::kFloat16,
              "q_linear: scale must be float16");
  TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32,
              "q_linear: x must be float16 or float32");
  TORCH_CHECK(nbits == 1 || nbits == 2 || nbits == 4 || nbits == 8,
              "q_linear: nbits must be one of {1, 2, 4, 8}, got ", nbits);

  const int vpb = 8 / nbits;
  const int K = (int)x.size(-1);
  const int N = (int)w_pack.size(0);
  const int M = (int)(x.numel() / K);

  TORCH_CHECK(K % group_size == 0, "q_linear: K=", K,
              " not divisible by group_size=", group_size);
  TORCH_CHECK((int)w_pack.size(1) == K / vpb,
              "q_linear: w_pack.size(1)=", w_pack.size(1),
              " != K/vpb=", K / vpb, " (nbits=", nbits, ")");

  const bool asym =
      zero_point_opt.has_value() && zero_point_opt.value().defined();
  if (asym) {
    TORCH_CHECK(zero_point_opt.value().is_cuda() &&
                    zero_point_opt.value().dtype() == torch::kFloat16,
                "q_linear: zero_point must be float16 on CUDA");
  }

  // --- Prepare pointers ---
  const torch::Tensor xc = x.contiguous();
  const torch::Tensor wc = w_pack.contiguous();
  const torch::Tensor sc = scale.contiguous().reshape({-1});
  torch::Tensor y = torch::empty({M, N}, xc.options());

  const __half *sc_ptr = reinterpret_cast<const __half *>(sc.data_ptr());
  const uint8_t *wp_ptr = wc.data_ptr<uint8_t>();
  const __half *zp_ptr = nullptr;
  torch::Tensor zpc;
  if (asym) {
    zpc = zero_point_opt.value().contiguous().reshape({-1});
    zp_ptr = reinterpret_cast<const __half *>(zpc.data_ptr());
  }

  const float *bias_ptr = nullptr;
  torch::Tensor bias_f32;
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    bias_f32 = bias_opt.value().contiguous().to(torch::kFloat32);
    bias_ptr = bias_f32.data_ptr<float>();
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (x.dtype() == torch::kFloat16) {
    _launch<__half>(reinterpret_cast<const __half *>(xc.data_ptr()), wp_ptr,
                    sc_ptr, zp_ptr, bias_ptr,
                    reinterpret_cast<__half *>(y.data_ptr()), M, N, K,
                    group_size, nbits, asym, stream);
  } else {
    _launch<float>(xc.data_ptr<float>(), wp_ptr, sc_ptr, zp_ptr, bias_ptr,
                   y.data_ptr<float>(), M, N, K, group_size, nbits, asym,
                   stream);
  }

  // Restore leading batch dims.
  auto out_shape = x.sizes().vec();
  out_shape.back() = N;
  return y.view(out_shape);
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "neuropress generic group-quantized linear CUDA kernel";
  m.def("q_linear", &q_linear,
        "Generic fused group-quant dequant+linear (CUDA).\n\n"
        "  x        : [..., K]  float16 or float32\n"
        "  w_pack   : [N, K/vpb]  uint8 packed weights\n"
        "  scale    : [N*(K/group_size)]  float16 per-group scales\n"
        "  nbits    : 1 | 2 | 4 | 8\n"
        "  group_size : must divide K\n"
        "  zero_point : optional [same shape as scale]  float16 (asymmetric)\n"
        "  bias     : optional [N]  float\n"
        "Returns [..., N] matching x dtype.",
        py::arg("x"), py::arg("w_pack"), py::arg("scale"), py::arg("nbits"),
        py::arg("group_size"), py::arg("zero_point") = py::none(),
        py::arg("bias") = py::none());
}
