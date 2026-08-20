/*
 * Fused VQ-weight linear kernel — zero-escape 4-tier flat 2-bit prefix.
 *
 * Each VQ index has a 2-bit prefix in a flat prefix stream:
 *   00 = zero   (centroid 0, no payload)
 *   01 = hot    (4-bit payload → centroid 1–16)
 *   10 = warm   (8-bit payload → centroid 17–256)
 *   11 = cold   (cold_bits-bit payload → centroid 257+)
 *
 * Unlike the n-tier CBS kernel (cascaded prefix), every index's tier is
 * determined by a single 2-bit read — no sequential dependency.  This
 * makes the decode simpler and slightly faster per lane.
 *
 * Wire format (split by the Python caller)
 * ----------------------------------------
 *   prefix_stream  : uint8 bitstream, 2 bits per index (packed LSB-first)
 *   hot_stream     : uint8 packed 4-bit remapped hot indices (0-15)
 *   warm_stream    : uint8 packed 8-bit remapped warm indices (0-239)
 *   cold_stream    : uint8 packed cold_bits-bit remapped cold indices
 *   row_offsets[4] : int32[N+1] cumulative tier-k count per row
 *                    (zero, hot, warm, cold)
 *
 * Thread mapping — same as vq_linear_tiered/ntier:
 *   grid  : (ceil(N / WARPS_PER_BLOCK), M)
 *   block : (WARP_SIZE=32, WARPS_PER_BLOCK)
 *
 * Shared memory:
 *   1. Centroid cache: first cache_cents centroids (tier-0 through warm,
 *      ≤257 for standard zero-escape).  Covers ~90-95% of lookups for
 *      Zipf-distributed codebook usage.
 *   2. Per-warp prefix cache: 2 bits per group, loaded cooperatively.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <optional>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8

// Maximum groups for which we cache prefix bits in smem per warp.
#define MAX_PREFIX_SMEM_GROUPS 8192
#define PREFIX_SMEM_BYTES(ng) (((ng) * 2 + 7) / 8)

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

template <typename T> struct is_half_t {
  static constexpr bool value = false;
};
template <> struct is_half_t<__half> {
  static constexpr bool value = true;
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

/* Compile-time unpack. */
template <int BITS>
__device__ __forceinline__ int unpack_bits(const uint8_t *__restrict__ buf,
                                           int pos) {
  if constexpr (BITS == 8) {
    return (int)__ldg(&buf[pos]);
  }
  if constexpr (BITS == 4) {
    const uint8_t b = __ldg(&buf[pos >> 1]);
    return (pos & 1) ? (int)(b >> 4) : (int)(b & 0xF);
  }
  if constexpr (BITS == 16) {
    const int byte0 = (int)__ldg(&buf[pos * 2]);
    const int byte1 = (int)__ldg(&buf[pos * 2 + 1]);
    return byte0 | (byte1 << 8);
  }
  // Generic bit-stream decode.
  const int bit_start = pos * BITS;
  int result = 0;
#pragma unroll
  for (int b = 0; b < BITS; b++) {
    const int abs_bit = bit_start + b;
    result |= ((int)((__ldg(&buf[abs_bit >> 3]) >> (abs_bit & 7)) & 1)) << b;
  }
  return result;
}

/* Runtime-dispatched unpack for cold stream. */
__device__ __forceinline__ int unpack_bits_rt(const uint8_t *__restrict__ buf,
                                              int pos, int bits) {
  switch (bits) {
  case 4:
    return unpack_bits<4>(buf, pos);
  case 8:
    return unpack_bits<8>(buf, pos);
  case 10: {
    const int bit_start = pos * 10;
    int result = 0;
    for (int b = 0; b < 10; b++) {
      const int abs_bit = bit_start + b;
      result |= ((int)((__ldg(&buf[abs_bit >> 3]) >> (abs_bit & 7)) & 1)) << b;
    }
    return result;
  }
  case 12: {
    const int bit_start = pos * 12;
    int result = 0;
    for (int b = 0; b < 12; b++) {
      const int abs_bit = bit_start + b;
      result |= ((int)((__ldg(&buf[abs_bit >> 3]) >> (abs_bit & 7)) & 1)) << b;
    }
    return result;
  }
  case 16:
    return unpack_bits<16>(buf, pos);
  }
  // Generic fallback.
  const int bit_start = pos * bits;
  int result = 0;
  for (int b = 0; b < bits; b++) {
    const int abs_bit = bit_start + b;
    result |= ((int)((__ldg(&buf[abs_bit >> 3]) >> (abs_bit & 7)) & 1)) << b;
  }
  return result;
}

/* Read 2-bit prefix from global memory. */
__device__ __forceinline__ int read_prefix_2bit(const uint8_t *__restrict__ buf,
                                                int pos) {
  const int bit_start = pos * 2;
  const int byte_idx = bit_start >> 3;
  const int bit_off = bit_start & 7;
  return (int)((__ldg(&buf[byte_idx]) >> bit_off) & 0x3);
}

/* Read 2-bit prefix from shared memory. */
__device__ __forceinline__ int read_prefix_2bit_smem(const uint8_t *buf,
                                                     int pos) {
  const int bit_start = pos * 2;
  const int byte_idx = bit_start >> 3;
  const int bit_off = bit_start & 7;
  return (int)((buf[byte_idx] >> bit_off) & 0x3);
}

// ---------------------------------------------------------------------------
// Vectorized dot-product helpers
// ---------------------------------------------------------------------------

template <int VL>
__device__ __forceinline__ float
dot_h2_ldg(const __half *__restrict__ x_row, int base_k,
           const __half *__restrict__ cent, int cidx) {
  float acc = 0.f;
  const half2 *x2 = reinterpret_cast<const half2 *>(&x_row[base_k]);
  const half2 *c2 = reinterpret_cast<const half2 *>(&cent[(size_t)cidx * VL]);
#pragma unroll
  for (int v = 0; v < VL / 2; v++) {
    const half2 xv = __ldg(&x2[v]);
    const half2 wv = __ldg(&c2[v]);
    const float2 xf = __half22float2(xv);
    const float2 wf = __half22float2(wv);
    acc += xf.x * wf.x + xf.y * wf.y;
  }
  return acc;
}

template <int VL>
__device__ __forceinline__ float
dot_h2_smem(const __half *__restrict__ x_row, int base_k,
            const __half *__restrict__ smem_cent, int cidx) {
  float acc = 0.f;
  const half2 *x2 = reinterpret_cast<const half2 *>(&x_row[base_k]);
  const half2 *c2 = reinterpret_cast<const half2 *>(&smem_cent[cidx * VL]);
#pragma unroll
  for (int v = 0; v < VL / 2; v++) {
    const half2 xv = __ldg(&x2[v]);
    const half2 wv = c2[v];
    const float2 xf = __half22float2(xv);
    const float2 wf = __half22float2(wv);
    acc += xf.x * wf.x + xf.y * wf.y;
  }
  return acc;
}

} // namespace

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/*
 * Kernel arguments struct.
 * The 4 tiers are always: zero(0), hot(1), warm(2), cold(3).
 */
struct ZescArgs {
  const uint8_t *prefix_ptr;   // 2-bit prefix stream
  const uint8_t *hot_ptr;      // 4-bit hot data stream
  const uint8_t *warm_ptr;     // 8-bit warm data stream
  const uint8_t *cold_ptr;     // cold_bits-bit cold data stream
  const int32_t *row_off_zero; // [N+1] cumulative zero counts
  const int32_t *row_off_hot;  // [N+1] cumulative hot counts
  const int32_t *row_off_warm; // [N+1] cumulative warm counts
  const int32_t *row_off_cold; // [N+1] cumulative cold counts
  int cold_bits;
  int cache_cents; // number of centroids cached in smem
};

template <int VECTOR_LEN, typename scalar_t>
__global__ void _vq_linear_zero_escape(
    const scalar_t *__restrict__ x,       // [M, K]
    const __half *__restrict__ centroids, // [num_centroids, VECTOR_LEN]
    const float *__restrict__ bias,       // [N] or nullptr
    scalar_t *__restrict__ y,             // [M, N]
    int M, int N, int n_groups, ZescArgs args) {
  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  const int m = blockIdx.y;
  const int lane = threadIdx.x;
  const int K = n_groups * VECTOR_LEN;
  const int tid = threadIdx.y * WARP_SIZE + lane;
  const int block_threads = WARPS_PER_BLOCK * WARP_SIZE;
  const int cache_cents = args.cache_cents;

  // --- Shared memory partitioning ---
  extern __shared__ uint8_t smem_raw[];
  const int cache_bytes = cache_cents * VECTOR_LEN * (int)sizeof(__half);
  __half *cached_cent = reinterpret_cast<__half *>(smem_raw);

  const int prefix_bytes = PREFIX_SMEM_BYTES(n_groups);
  const bool use_prefix_smem = (n_groups <= MAX_PREFIX_SMEM_GROUPS);
  uint8_t *prefix_base = smem_raw + cache_bytes;
  uint8_t *my_prefix =
      use_prefix_smem ? (prefix_base + threadIdx.y * prefix_bytes) : nullptr;

  // --- Cooperative load: cached centroids into shared memory ---
  const int cache_elems = cache_cents * VECTOR_LEN;
  for (int i = tid; i < cache_elems; i += block_threads) {
    cached_cent[i] = __ldg(&centroids[i]);
  }

  // Early exit AFTER centroid load (all threads must participate in sync).
  if (n >= N || m >= M) {
    __syncthreads();
    return;
  }

  // --- Per-warp prefix load ---
  if (use_prefix_smem) {
    // Each index occupies 2 bits; row n starts at bit n*n_groups*2.
    const int row_base_byte = (n * n_groups * 2) >> 3;
    for (int i = lane; i < prefix_bytes; i += WARP_SIZE) {
      my_prefix[i] = __ldg(&args.prefix_ptr[row_base_byte + i]);
    }
  }

  __syncthreads();
  if (use_prefix_smem)
    __syncwarp();

  // --- Load row-level base offsets (lane-0 broadcast) ---
  int off_zero, off_hot, off_warm, off_cold;
  if (lane == 0) {
    off_zero = __ldg(&args.row_off_zero[n]);
    off_hot = __ldg(&args.row_off_hot[n]);
    off_warm = __ldg(&args.row_off_warm[n]);
    off_cold = __ldg(&args.row_off_cold[n]);
  }
  off_zero = __shfl_sync(0xFFFFFFFF, off_zero, 0);
  off_hot = __shfl_sync(0xFFFFFFFF, off_hot, 0);
  off_warm = __shfl_sync(0xFFFFFFFF, off_warm, 0);
  off_cold = __shfl_sync(0xFFFFFFFF, off_cold, 0);

  const scalar_t *x_row = &x[(size_t)m * K];
  float acc = 0.f;

  for (int g_base = 0; g_base < n_groups; g_base += WARP_SIZE) {
    const int g = g_base + lane;
    const bool valid = (g < n_groups);

    // ------ Read 2-bit prefix ------
    int tier = 0;
    if (valid) {
      if (use_prefix_smem) {
        tier = read_prefix_2bit_smem(my_prefix, g);
      } else {
        tier = read_prefix_2bit(args.prefix_ptr, n * n_groups + g);
      }
    }

    // ------ Per-tier ballots ------
    uint32_t ballot_zero = __ballot_sync(0xFFFFFFFF, valid && tier == 0);
    uint32_t ballot_hot = __ballot_sync(0xFFFFFFFF, valid && tier == 1);
    uint32_t ballot_warm = __ballot_sync(0xFFFFFFFF, valid && tier == 2);
    uint32_t ballot_cold = __ballot_sync(0xFFFFFFFF, valid && tier == 3);

    const uint32_t lane_mask = (1u << lane) - 1u;

    // ------ Decode centroid index ------
    int cidx = 0;
    if (valid) {
      switch (tier) {
      case 0:
        // Zero tier: centroid index is always 0.
        cidx = 0;
        break;
      case 1: {
        // Hot tier: 4-bit payload, remapped +1.
        int pos = off_hot + (int)__popc(ballot_hot & lane_mask);
        cidx = unpack_bits<4>(args.hot_ptr, pos) + 1;
        break;
      }
      case 2: {
        // Warm tier: 8-bit payload, remapped +17.
        int pos = off_warm + (int)__popc(ballot_warm & lane_mask);
        cidx = unpack_bits<8>(args.warm_ptr, pos) + 17;
        break;
      }
      case 3: {
        // Cold tier: cold_bits-bit payload, remapped +257.
        int pos = off_cold + (int)__popc(ballot_cold & lane_mask);
        cidx = unpack_bits_rt(args.cold_ptr, pos, args.cold_bits) + 257;
        break;
      }
      }

      // --- Dot product ---
      const int base_k = g * VECTOR_LEN;

      if constexpr (is_half_t<scalar_t>::value && VECTOR_LEN >= 2) {
        if (cidx < cache_cents) {
          acc +=
              dot_h2_smem<VECTOR_LEN>(reinterpret_cast<const __half *>(x_row),
                                      base_k, cached_cent, cidx);
        } else {
          acc += dot_h2_ldg<VECTOR_LEN>(reinterpret_cast<const __half *>(x_row),
                                        base_k, centroids, cidx);
        }
      } else {
#pragma unroll
        for (int v = 0; v < VECTOR_LEN; v++) {
          float wt;
          if (cidx < cache_cents) {
            wt = __half2float(cached_cent[cidx * VECTOR_LEN + v]);
          } else {
            wt = __half2float(__ldg(&centroids[(size_t)cidx * VECTOR_LEN + v]));
          }
          acc += to_f32(__ldg(&x_row[base_k + v])) * wt;
        }
      }
    }

    // ------ Advance running offsets ------
    const int batch_n = min(WARP_SIZE, n_groups - g_base);
    const uint32_t vmask =
        (batch_n < WARP_SIZE) ? ((1u << batch_n) - 1u) : 0xFFFFFFFFu;

    off_zero += (int)__popc(ballot_zero & vmask);
    off_hot += (int)__popc(ballot_hot & vmask);
    off_warm += (int)__popc(ballot_warm & vmask);
    off_cold += (int)__popc(ballot_cold & vmask);
  }

  // Warp reduction.
  acc = warp_reduce_sum(acc);
  if (lane == 0) {
    if (bias)
      acc += bias[n];
    y[(size_t)m * N + n] = from_f32<scalar_t>(acc);
  }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

static inline dim3 _zesc_grid(int N, int M) {
  return dim3((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, (unsigned)M);
}
static const dim3 _ZESC_BLOCK(WARP_SIZE, WARPS_PER_BLOCK);

template <typename scalar_t>
static void _zesc_launch(const scalar_t *x, const __half *centroids,
                         const float *bias, scalar_t *y, int M, int N,
                         int n_groups, int vector_len, ZescArgs args,
                         cudaStream_t stream) {
  const dim3 grid = _zesc_grid(N, M);

  const size_t cache_bytes =
      (size_t)args.cache_cents * vector_len * sizeof(__half);
  const size_t prefix_bytes_per_warp =
      (n_groups <= MAX_PREFIX_SMEM_GROUPS) ? PREFIX_SMEM_BYTES(n_groups) : 0;
  const size_t smem_bytes =
      cache_bytes + (size_t)WARPS_PER_BLOCK * prefix_bytes_per_warp;

#define ZESCL(VL)                                                              \
  _vq_linear_zero_escape<VL, scalar_t>                                         \
      <<<grid, _ZESC_BLOCK, smem_bytes, stream>>>(x, centroids, bias, y, M, N, \
                                                  n_groups, args)

  switch (vector_len) {
  case 2:
    ZESCL(2);
    break;
  case 4:
    ZESCL(4);
    break;
  case 8:
    ZESCL(8);
    break;
  case 16:
    ZESCL(16);
    break;
  default:
    TORCH_CHECK(false,
                "vq_linear_zero_escape: unsupported vector_len=", vector_len,
                ".  Must be in {2,4,8,16}.");
  }
#undef ZESCL
}

// ---------------------------------------------------------------------------
// Torch-facing function
// ---------------------------------------------------------------------------

/*
 * vq_linear_zero_escape(x, prefix_stream, hot_stream, warm_stream,
 *                       cold_stream, row_off_zero, row_off_hot,
 *                       row_off_warm, row_off_cold, centroids,
 *                       cold_bits, vector_len, bias=None)
 *
 * All tensors must be on the same CUDA device.
 * x             : [M, K]  float16 or float32
 * prefix_stream : uint8 — 2-bit-per-index prefix bitstream
 * hot_stream    : uint8 — 4-bit packed hot indices
 * warm_stream   : uint8 — 8-bit warm indices
 * cold_stream   : uint8 — cold_bits-bit packed cold indices
 * row_off_*     : int32[N+1] — cumulative tier counts per row
 * centroids     : [num_centroids, vector_len] float16
 * cold_bits     : int — bit width of cold tier
 * vector_len    : int — VQ vector length (2, 4, 8, or 16)
 * bias          : optional [N] float
 *
 * Returns [M, N] matching x dtype.
 */
torch::Tensor
vq_linear_zero_escape(torch::Tensor x, torch::Tensor prefix_stream,
                      torch::Tensor hot_stream, torch::Tensor warm_stream,
                      torch::Tensor cold_stream, torch::Tensor row_off_zero,
                      torch::Tensor row_off_hot, torch::Tensor row_off_warm,
                      torch::Tensor row_off_cold, torch::Tensor centroids,
                      int cold_bits, int vector_len,
                      std::optional<torch::Tensor> bias_opt) {
  TORCH_CHECK(x.is_cuda(), "vq_linear_zero_escape: x must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "vq_linear_zero_escape: x must be 2-D [M, K].");
  TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32,
              "vq_linear_zero_escape: x must be float16 or float32.");
  TORCH_CHECK(centroids.dtype() == torch::kFloat16,
              "vq_linear_zero_escape: centroids must be float16.");

  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  TORCH_CHECK(K % vector_len == 0, "vq_linear_zero_escape: K=", K,
              " not divisible by vector_len=", vector_len);
  const int n_groups = K / vector_len;
  const int N = (int)row_off_zero.size(0) - 1;
  TORCH_CHECK(N > 0, "vq_linear_zero_escape: row_off_zero too short.");

  // Ensure contiguous + on CUDA.
  auto xc = x.contiguous();
  auto pfx_c = prefix_stream.contiguous().cuda();
  auto hot_c = hot_stream.contiguous().cuda();
  auto warm_c = warm_stream.contiguous().cuda();
  auto cold_c = cold_stream.contiguous().cuda();
  auto roz_c = row_off_zero.contiguous().to(torch::kInt32).cuda();
  auto roh_c = row_off_hot.contiguous().to(torch::kInt32).cuda();
  auto row_c = row_off_warm.contiguous().to(torch::kInt32).cuda();
  auto roc_c = row_off_cold.contiguous().to(torch::kInt32).cuda();
  auto cent_c = centroids.contiguous().cuda();

  // Build args.
  ZescArgs args;
  args.prefix_ptr = pfx_c.data_ptr<uint8_t>();
  args.hot_ptr = hot_c.data_ptr<uint8_t>();
  args.warm_ptr = warm_c.data_ptr<uint8_t>();
  args.cold_ptr = cold_c.data_ptr<uint8_t>();
  args.row_off_zero = roz_c.data_ptr<int32_t>();
  args.row_off_hot = roh_c.data_ptr<int32_t>();
  args.row_off_warm = row_c.data_ptr<int32_t>();
  args.row_off_cold = roc_c.data_ptr<int32_t>();
  args.cold_bits = cold_bits;

  // Cache centroids 0–256 (zero+hot+warm range) in smem.
  // Cap at 32KB / (vector_len * 2 bytes).
  const int num_centroids = (int)centroids.size(0);
  const int max_cache = (32 * 1024) / (vector_len * (int)sizeof(at::Half));
  int cache_cents = 257; // TIER_2_MAX: covers zero(1) + hot(16) + warm(240)
  cache_cents = std::min(cache_cents, num_centroids);
  cache_cents = std::min(cache_cents, max_cache);
  args.cache_cents = cache_cents;

  torch::Tensor y = torch::empty({M, N}, xc.options());

  const float *bias_ptr = nullptr;
  torch::Tensor bias_f32;
  if (bias_opt.has_value() && bias_opt.value().defined()) {
    bias_f32 = bias_opt.value().contiguous().to(torch::kFloat32).cuda();
    bias_ptr = bias_f32.data_ptr<float>();
  }

  const auto stream = at::cuda::getCurrentCUDAStream();
  const __half *cent_ptr =
      reinterpret_cast<const __half *>(cent_c.data_ptr<at::Half>());

  if (x.dtype() == torch::kFloat16) {
    _zesc_launch<__half>(
        reinterpret_cast<const __half *>(xc.data_ptr<at::Half>()), cent_ptr,
        bias_ptr, reinterpret_cast<__half *>(y.data_ptr<at::Half>()), M, N,
        n_groups, vector_len, args, stream);
  } else {
    _zesc_launch<float>(xc.data_ptr<float>(), cent_ptr, bias_ptr,
                        y.data_ptr<float>(), M, N, n_groups, vector_len, args,
                        stream);
  }

  return y;
}

// ---------------------------------------------------------------------------
// Pybind11 bindings
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_linear_zero_escape", &vq_linear_zero_escape,
        "Fused VQ zero-escape linear (flat 2-bit prefix + ballot decode + "
        "centroid gather + matmul).",
        py::arg("x"), py::arg("prefix_stream"), py::arg("hot_stream"),
        py::arg("warm_stream"), py::arg("cold_stream"), py::arg("row_off_zero"),
        py::arg("row_off_hot"), py::arg("row_off_warm"),
        py::arg("row_off_cold"), py::arg("centroids"), py::arg("cold_bits"),
        py::arg("vector_len"), py::arg("bias") = py::none());
}
