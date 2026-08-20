/*
 * Fused VQ-weight linear kernel — two-tier prefix-coded index decoding.
 *
 * The weight indices are stored in a tiered format (see TieredIndexPacking):
 *   - A 1-bit prefix selects hot (4-bit, centroids 0–15) vs cold
 * (COLD_BITS-bit).
 *   - Centroids must be frequency-sorted so low indices are most-assigned.
 *
 * Wire format expected by this kernel
 * ------------------------------------
 * The Python caller (NPVQLinear._forward_cuda_tiered) splits the monolithic
 * tiered buffer into four separate tensors passed as pointers:
 *
 *   prefix_stream  : uint8[ceil(N * n_groups / 8)]
 *   hot_stream     : uint8[ceil(n_hot_total * 4 / 8)]   (4-bit packed)
 *   cold_stream    : uint8[ceil(n_cold_total * COLD_BITS / 8)]
 *   hot_row_off    : int32[N + 1]   cumulative hot  count ending before row i
 *   cold_row_off   : int32[N + 1]   cumulative cold count ending before row i
 *
 * Thread mapping — one warp per output element y[m, n]:
 *   grid  : (ceil(N / WARPS_PER_BLOCK), M)
 *   block : (WARP_SIZE=32, WARPS_PER_BLOCK=8)
 *
 * Within each warp, lanes process GROUPS in batches of WARP_SIZE:
 *   - Lane l handles group g = g_base + l for each batch.
 *   - __ballot_sync identifies hot vs cold lanes.
 *   - popcount(ballot & lane_mask) gives lane's position in hot/cold stream.
 *   - Each lane accumulates vector_len dot-products for its group.
 *   - __shfl_down_sync reduces all lanes' partial sums to lane 0.
 *
 * Template parameters
 * -------------------
 *   COLD_BITS  : bit-width of cold-tier indices (4 | 5 | 6 | 7 | 8 | 10 | 12 |
 * 14 | 16). VECTOR_LEN : VQ vector length (2 | 4 | 8 | 16).  Compile-time
 * constant enables full loop unrolling and vectorized memory loads. scalar_t :
 * __half or float (input/output element type).
 *
 * Optimizations
 * -------------
 *   - Hot centroid cache: first 2^HOT_BITS centroids cached in shared memory.
 *   - Prefix cache: per-warp prefix bytes cached in shared memory.
 *   - Vectorized x loads: half2 loads when scalar_t is __half (2× fewer
 *     global memory transactions for the input tensor).
 *   - Full unroll: VECTOR_LEN as template param enables compile-time unrolling
 *     of the inner dot-product loop.
 *
 * Performance note
 * ----------------
 * Warp utilisation = min(WARP_SIZE, n_groups) / WARP_SIZE per batch.
 * For n_groups >= 32 (K >= 128 at VL=4) all lanes are always busy.
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
// Shared helpers
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

// Type trait to identify __half without needing <type_traits>.
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

/* Decode one BITS-wide index at position `pos` in a bit-stream buffer.
 * Bit layout: index i spans bits [i*BITS, (i+1)*BITS), LSB first.
 * Used for both hot and cold streams.
 */
template <int BITS>
__device__ __forceinline__ int unpack_bits(const uint8_t *__restrict__ buf,
                                           int pos) {
  // Optimised fast paths for power-of-2 widths.
  if constexpr (BITS == 8) {
    return (int)__ldg(&buf[pos]);
  }
  if constexpr (BITS == 4) {
    // Two per byte, LSB-first.
    const uint8_t b = __ldg(&buf[pos >> 1]);
    return (pos & 1) ? (int)(b >> 4) : (int)(b & 0xF);
  }
  if constexpr (BITS == 16) {
    // Little-endian 16-bit.
    const int byte0 = (int)__ldg(&buf[pos * 2]);
    const int byte1 = (int)__ldg(&buf[pos * 2 + 1]);
    return byte0 | (byte1 << 8);
  }
  // Generic bit-stream decode for 5, 6, 7, 10, 12, 14, 17, 18, 20.
  const int bit_start = pos * BITS;
  int result = 0;
#pragma unroll
  for (int b = 0; b < BITS; b++) {
    const int abs_bit = bit_start + b;
    const int byte_i = abs_bit >> 3;
    const int bit_i = abs_bit & 7;
    result |= ((int)((__ldg(&buf[byte_i]) >> bit_i) & 1)) << b;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Vectorized dot-product helpers
// ---------------------------------------------------------------------------

/* Accumulate the dot product of x[base_k .. base_k+VL) with centroid values.
 *
 * Two specialisations:
 *   __half  → half2 loads (2× fewer global/shared memory transactions)
 *   float   → scalar loads with full unroll
 *
 * `cent_ptr` points either into shared memory (hot) or global memory (cold).
 * `use_ldg` selects __ldg (global) vs plain dereference (shared).
 */
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
  // Shared memory — no __ldg, plain load.
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
 * Shared memory layout (all dynamic, via extern __shared__):
 *
 *   1. Hot centroid cache:  (1 << HOT_BITS) * VECTOR_LEN * sizeof(__half) bytes
 *      Shared across ALL warps in the block.  The first 2^HOT_BITS centroids
 *      are loaded once and reused for every row processed by this block.
 *      Since hot indices are the majority by design, this converts most
 *      centroid gathers from random global memory reads (~100 cycles) into
 *      shared memory reads (~1 cycle).
 *
 *      HOT_BITS=4 → 16 centroids cached (default, small codebooks)
 *      HOT_BITS=8 → 256 centroids cached (128K+ codebooks)
 *
 *   2. Per-warp prefix cache: WARPS_PER_BLOCK * ceil(n_groups/8) bytes
 *      Each warp loads its row's prefix bytes from global memory once
 *      and reads them from shared memory on every warp batch.
 *
 * Typical sizes (n_groups=1024, VECTOR_LEN=4, HOT_BITS=4):
 *   hot centroids: 16 * 4 * 2 = 128 bytes
 *   prefix:        8 * 128    = 1024 bytes
 *   total:         ~1.2 KB — well within the 48 KB limit.
 *
 * With HOT_BITS=8, VECTOR_LEN=4:
 *   hot centroids: 256 * 4 * 2 = 2048 bytes
 *   prefix:        8 * 128     = 1024 bytes
 *   total:         ~3 KB — still well within the 48 KB limit.
 */

// Max groups we support with shared-memory prefix caching.
// Beyond this we fall back to global __ldg reads (still correct).
#define MAX_PREFIX_SMEM_GROUPS 8192
#define PREFIX_SMEM_BYTES(ng) (((ng) + 7) / 8)

template <int HOT_BITS, int COLD_BITS, int VECTOR_LEN, typename scalar_t>
__global__ void _vq_linear_tiered(
    const scalar_t *__restrict__ x,            // [M, K]
    const uint8_t *__restrict__ prefix_stream, // [ceil(N*n_groups/8)]
    const uint8_t *__restrict__ hot_stream,    // [ceil(n_hot_total*4/8)]
    const uint8_t *__restrict__ cold_stream, // [ceil(n_cold_total*COLD_BITS/8)]
    const int32_t *__restrict__ hot_row_off, // [N+1]
    const int32_t *__restrict__ cold_row_off, // [N+1]
    const __half *__restrict__ centroids,     // [num_centroids, VECTOR_LEN]
    const float *__restrict__ bias,           // [N] or nullptr
    scalar_t *__restrict__ y,                 // [M, N]
    int M, int N, int n_groups) {
  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  const int m = blockIdx.y;

  const int lane = threadIdx.x;
  const int K = n_groups * VECTOR_LEN;
  const int tid = threadIdx.y * WARP_SIZE + lane;
  const int block_threads = WARPS_PER_BLOCK * WARP_SIZE;

  // --- Shared memory partitioning ---
  extern __shared__ uint8_t smem_raw[];
  // Region 1: hot centroid cache (shared by all warps).
  constexpr int HOT_CENTROIDS = 1 << HOT_BITS;
  constexpr int hot_cache_bytes =
      HOT_CENTROIDS * VECTOR_LEN * (int)sizeof(__half);
  __half *hot_cent = reinterpret_cast<__half *>(smem_raw);

  // Region 2: per-warp prefix cache (starts after hot centroids).
  const int prefix_bytes = PREFIX_SMEM_BYTES(n_groups);
  const bool use_prefix_smem = (n_groups <= MAX_PREFIX_SMEM_GROUPS);
  uint8_t *prefix_base = smem_raw + hot_cache_bytes;
  uint8_t *my_prefix =
      use_prefix_smem ? (prefix_base + threadIdx.y * prefix_bytes) : nullptr;

  // --- Cooperative load: hot centroids (all threads participate) ---
  constexpr int hot_elems = HOT_CENTROIDS * VECTOR_LEN;
  for (int i = tid; i < hot_elems; i += block_threads) {
    hot_cent[i] = __ldg(&centroids[i]);
  }

  // Early exit AFTER hot centroid load (all threads must participate).
  if (n >= N || m >= M) {
    __syncthreads();
    return;
  }

  // --- Per-warp prefix load ---
  if (use_prefix_smem) {
    const int row_base_byte = (n * n_groups) >> 3;
    for (int i = lane; i < prefix_bytes; i += WARP_SIZE) {
      my_prefix[i] = __ldg(&prefix_stream[row_base_byte + i]);
    }
  }

  // Sync: hot centroid cache must be visible to all warps.
  __syncthreads();
  // Prefix is per-warp, so __syncwarp suffices.
  if (use_prefix_smem)
    __syncwarp();

  // Load row-level base offsets via lane-0 broadcast (saves 31 __ldg).
  int hot_off, cold_off;
  if (lane == 0) {
    hot_off = __ldg(&hot_row_off[n]);
    cold_off = __ldg(&cold_row_off[n]);
  }
  hot_off = __shfl_sync(0xFFFFFFFF, hot_off, 0);
  cold_off = __shfl_sync(0xFFFFFFFF, cold_off, 0);

  // Pointer to this row of x.
  const scalar_t *x_row = &x[(size_t)m * K];

  float acc = 0.f;

  for (int g_base = 0; g_base < n_groups; g_base += WARP_SIZE) {
    const int g = g_base + lane;

    // Each lane reads its prefix bit (0 = hot, 1 = cold).
    int is_cold = 0;
    if (g < n_groups) {
      if (use_prefix_smem) {
        const int byte_i = g >> 3;
        const int bit_i = g & 7;
        is_cold = (int)((my_prefix[byte_i] >> bit_i) & 1);
      } else {
        const int flat = n * n_groups + g;
        const int byte_i = flat >> 3;
        const int bit_i = flat & 7;
        is_cold = (int)((__ldg(&prefix_stream[byte_i]) >> bit_i) & 1);
      }
    }

    // Ballot: bit l set  ↔  lane l is cold.
    const uint32_t cold_ballot = __ballot_sync(0xFFFFFFFF, is_cold);
    const uint32_t hot_ballot = ~cold_ballot;
    const uint32_t lane_mask = (1u << lane) - 1u;

    if (g < n_groups) {
      int cidx;
      if (!is_cold) {
        // Hot — HOT_BITS-bit stream.
        const int h = hot_off + (int)__popc(hot_ballot & lane_mask);
        cidx = unpack_bits<HOT_BITS>(hot_stream, h);
      } else {
        // Cold — COLD_BITS-bit bit-stream.
        const int c = cold_off + (int)__popc(cold_ballot & lane_mask);
        cidx = unpack_bits<COLD_BITS>(cold_stream, c);
      }

      // Accumulate dot product for this group.
      // half path: vectorized half2 loads (2× fewer memory txns).
      // float path: scalar loads with full compile-time unroll.
      const int base_k = g * VECTOR_LEN;

      if constexpr (is_half_t<scalar_t>::value && VECTOR_LEN >= 2) {
        // --- half2 vectorised path ---
        if (cidx < HOT_CENTROIDS) {
          acc += dot_h2_smem<VECTOR_LEN>(
              reinterpret_cast<const __half *>(x_row), base_k, hot_cent, cidx);
        } else {
          acc += dot_h2_ldg<VECTOR_LEN>(reinterpret_cast<const __half *>(x_row),
                                        base_k, centroids, cidx);
        }
      } else {
        // --- scalar path (float, or odd VECTOR_LEN) ---
#pragma unroll
        for (int v = 0; v < VECTOR_LEN; v++) {
          float wt;
          if (cidx < HOT_CENTROIDS) {
            wt = __half2float(hot_cent[cidx * VECTOR_LEN + v]);
          } else {
            wt = __half2float(__ldg(&centroids[(size_t)cidx * VECTOR_LEN + v]));
          }
          const int k = base_k + v;
          acc += to_f32(__ldg(&x_row[k])) * wt;
        }
      }
    }

    // Advance base offsets by this batch's hot/cold counts.
    const int batch_n = min(WARP_SIZE, n_groups - g_base);
    const uint32_t vmask =
        (batch_n < WARP_SIZE) ? ((1u << batch_n) - 1u) : 0xFFFFFFFFu;
    hot_off += (int)__popc(hot_ballot & vmask);
    cold_off += (int)__popc(cold_ballot & vmask);
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
// Dispatch helper
// ---------------------------------------------------------------------------

static inline dim3 _tgrid(int N, int M) {
  return dim3((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, (unsigned)M);
}
static const dim3 _T_BLOCK(WARP_SIZE, WARPS_PER_BLOCK);

template <int HOT_BITS, typename scalar_t>
static void
_tiered_launch_hb(const scalar_t *x, const uint8_t *prefix, const uint8_t *hot,
                  const uint8_t *cold, const int32_t *hot_off,
                  const int32_t *cold_off, const __half *centroids,
                  const float *bias, scalar_t *y, int M, int N, int n_groups,
                  int vector_len, int cold_bits, cudaStream_t stream) {
  const dim3 grid = _tgrid(N, M);

  // Shared memory: hot centroid cache + per-warp prefix byte buffer.
  constexpr int hot_centroids = 1 << HOT_BITS;
  const size_t hot_cache = (size_t)hot_centroids * vector_len * sizeof(__half);
  const size_t prefix_bytes_per_warp =
      (n_groups <= MAX_PREFIX_SMEM_GROUPS) ? PREFIX_SMEM_BYTES(n_groups) : 0;
  const size_t smem_bytes =
      hot_cache + (size_t)WARPS_PER_BLOCK * prefix_bytes_per_warp;

  // 3-D dispatch: HOT_BITS × COLD_BITS × VECTOR_LEN.
  // Supported VECTOR_LEN values: 2, 4, 8, 16.
#define TL(CB, VL)                                                             \
  _vq_linear_tiered<HOT_BITS, CB, VL, scalar_t>                                \
      <<<grid, _T_BLOCK, smem_bytes, stream>>>(x, prefix, hot, cold, hot_off,  \
                                               cold_off, centroids, bias, y,   \
                                               M, N, n_groups)

#define TL_VL(CB)                                                              \
  switch (vector_len) {                                                        \
  case 2:                                                                      \
    TL(CB, 2);                                                                 \
    break;                                                                     \
  case 4:                                                                      \
    TL(CB, 4);                                                                 \
    break;                                                                     \
  case 8:                                                                      \
    TL(CB, 8);                                                                 \
    break;                                                                     \
  case 16:                                                                     \
    TL(CB, 16);                                                                \
    break;                                                                     \
  default:                                                                     \
    TORCH_CHECK(false,                                                         \
                "vq_linear_tiered: unsupported vector_len=", vector_len,       \
                ".  Must be in {2,4,8,16}.");                                  \
  }

  switch (cold_bits) {
  case 4:
    TL_VL(4);
    break;
  case 5:
    TL_VL(5);
    break;
  case 6:
    TL_VL(6);
    break;
  case 7:
    TL_VL(7);
    break;
  case 8:
    TL_VL(8);
    break;
  case 9:
    TL_VL(9);
    break;
  case 10:
    TL_VL(10);
    break;
  case 11:
    TL_VL(11);
    break;
  case 12:
    TL_VL(12);
    break;
  case 13:
    TL_VL(13);
    break;
  case 14:
    TL_VL(14);
    break;
  case 15:
    TL_VL(15);
    break;
  case 16:
    TL_VL(16);
    break;
  case 17:
    TL_VL(17);
    break;
  case 18:
    TL_VL(18);
    break;
  case 19:
    TL_VL(19);
    break;
  case 20:
    TL_VL(20);
    break;
  default:
    TORCH_CHECK(false, "vq_linear_tiered: unsupported cold_bits=", cold_bits,
                ".  Must be in [4, 20].");
  }
#undef TL_VL
#undef TL
}

template <typename scalar_t>
static void _tiered_launch(const scalar_t *x, const uint8_t *prefix,
                           const uint8_t *hot, const uint8_t *cold,
                           const int32_t *hot_off, const int32_t *cold_off,
                           const __half *centroids, const float *bias,
                           scalar_t *y, int M, int N, int n_groups,
                           int vector_len, int cold_bits, int hot_bits,
                           cudaStream_t stream) {
  switch (hot_bits) {
  case 4:
    _tiered_launch_hb<4, scalar_t>(x, prefix, hot, cold, hot_off, cold_off,
                                   centroids, bias, y, M, N, n_groups,
                                   vector_len, cold_bits, stream);
    break;
  case 8:
    _tiered_launch_hb<8, scalar_t>(x, prefix, hot, cold, hot_off, cold_off,
                                   centroids, bias, y, M, N, n_groups,
                                   vector_len, cold_bits, stream);
    break;
  default:
    TORCH_CHECK(false, "vq_linear_tiered: unsupported hot_bits=", hot_bits,
                ".  Must be 4 or 8.");
  }
}

// ---------------------------------------------------------------------------
// Torch-facing function
// ---------------------------------------------------------------------------

/*
 * vq_linear_tiered(x, prefix_stream, hot_stream, cold_stream,
 *                  hot_row_off, cold_row_off,
 *                  centroids, cold_bits, hot_bits, vector_len, bias=None)
 *
 * All tensors must be on the same CUDA device.
 * x            : [M, K]  float16 or float32
 * prefix_stream: [ceil(N*n_groups/8)]  uint8
 * hot_stream   : [ceil(n_hot_total*hot_bits/8)]  uint8
 * cold_stream  : [ceil(n_cold_total*cold_bits/8)]  uint8
 * hot_row_off  : [N+1]  int32
 * cold_row_off : [N+1]  int32
 * centroids    : [num_centroids, vector_len]  float16
 * cold_bits    : int — cold-tier bit width (4..20)
 * hot_bits     : int — hot-tier bit width (4 or 8)
 * vector_len   : int — VQ vector length (2, 4, 8, or 16)
 * bias         : optional [N]  float
 *
 * Returns [M, N] matching x dtype.
 */
torch::Tensor
vq_linear_tiered(torch::Tensor x, torch::Tensor prefix_stream,
                 torch::Tensor hot_stream, torch::Tensor cold_stream,
                 torch::Tensor hot_row_off, torch::Tensor cold_row_off,
                 torch::Tensor centroids, int cold_bits, int hot_bits,
                 int vector_len, std::optional<torch::Tensor> bias_opt) {
  TORCH_CHECK(x.is_cuda(), "vq_linear_tiered: x must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "vq_linear_tiered: x must be 2-D [M, K].");
  TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32,
              "vq_linear_tiered: x must be float16 or float32.");
  TORCH_CHECK(centroids.dtype() == torch::kFloat16,
              "vq_linear_tiered: centroids must be float16.");
  TORCH_CHECK(prefix_stream.dtype() == torch::kUInt8,
              "vq_linear_tiered: prefix_stream must be uint8.");
  TORCH_CHECK(hot_stream.dtype() == torch::kUInt8,
              "vq_linear_tiered: hot_stream must be uint8.");
  TORCH_CHECK(cold_stream.dtype() == torch::kUInt8,
              "vq_linear_tiered: cold_stream must be uint8.");
  TORCH_CHECK(hot_row_off.dtype() == torch::kInt32,
              "vq_linear_tiered: hot_row_off must be int32.");
  TORCH_CHECK(cold_row_off.dtype() == torch::kInt32,
              "vq_linear_tiered: cold_row_off must be int32.");

  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  TORCH_CHECK(K % vector_len == 0, "vq_linear_tiered: K=", K,
              " not divisible by vector_len=", vector_len);
  const int n_groups = K / vector_len;
  // N = hot_row_off.size(0) - 1
  const int N = (int)hot_row_off.size(0) - 1;
  TORCH_CHECK(N > 0, "vq_linear_tiered: hot_row_off too short.");

  auto xc = x.contiguous();
  auto pref_c = prefix_stream.contiguous().cuda();
  auto hot_c = hot_stream.contiguous().cuda();
  auto cold_c = cold_stream.contiguous().cuda();
  auto hoff_c = hot_row_off.contiguous().cuda();
  auto coff_c = cold_row_off.contiguous().cuda();
  auto cent_c = centroids.contiguous().cuda();

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
  const uint8_t *pref_ptr = pref_c.data_ptr<uint8_t>();
  const uint8_t *hot_ptr =
      hot_c.numel() > 0 ? hot_c.data_ptr<uint8_t>() : nullptr;
  const uint8_t *cold_ptr =
      cold_c.numel() > 0 ? cold_c.data_ptr<uint8_t>() : nullptr;
  const int32_t *hoff_ptr = hoff_c.data_ptr<int32_t>();
  const int32_t *coff_ptr = coff_c.data_ptr<int32_t>();

  if (x.dtype() == torch::kFloat16) {
    _tiered_launch<__half>(
        reinterpret_cast<const __half *>(xc.data_ptr<at::Half>()), pref_ptr,
        hot_ptr, cold_ptr, hoff_ptr, coff_ptr, cent_ptr, bias_ptr,
        reinterpret_cast<__half *>(y.data_ptr<at::Half>()), M, N, n_groups,
        vector_len, cold_bits, hot_bits, stream);
  } else {
    _tiered_launch<float>(xc.data_ptr<float>(), pref_ptr, hot_ptr, cold_ptr,
                          hoff_ptr, coff_ptr, cent_ptr, bias_ptr,
                          y.data_ptr<float>(), M, N, n_groups, vector_len,
                          cold_bits, hot_bits, stream);
  }

  return y;
}

// ---------------------------------------------------------------------------
// Pybind11 bindings
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_linear_tiered", &vq_linear_tiered,
        "Fused VQ tiered-index linear (warp-ballot decode + centroid gather + "
        "matmul).",
        py::arg("x"), py::arg("prefix_stream"), py::arg("hot_stream"),
        py::arg("cold_stream"), py::arg("hot_row_off"), py::arg("cold_row_off"),
        py::arg("centroids"), py::arg("cold_bits"), py::arg("hot_bits"),
        py::arg("vector_len"), py::arg("bias") = py::none());
}
