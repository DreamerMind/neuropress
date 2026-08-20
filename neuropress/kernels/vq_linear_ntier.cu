/*
 * Fused VQ-weight linear kernel — N-tier cascaded binary splitting (CBS).
 *
 * Generalises vq_linear_tiered.cu from 2 tiers to D tiers (2 ≤ D ≤ 4).
 * Each level k has a 1-bit prefix stream that splits the current population
 * into "tier-k" (encoded in tier_bits[k] bits) and "rest" (fed to level k+1).
 * The last tier has no prefix — it is whatever remains.
 *
 * Wire format (split by the Python caller)
 * ----------------------------------------
 * The Python caller (NPVQLinear._forward_cuda_ntier) splits the monolithic
 * n-tier buffer produced by NTieredIndexPacking.pack() into:
 *
 *   prefix_streams[0..D-2] : uint8 bitstreams, one per cascade level
 *   tier_streams[0..D-1]   : uint8 packed index data, one per tier
 *   row_offsets[0..D-1]    : int32[N+1] cumulative tier-k count per row
 *
 * Thread mapping — identical to vq_linear_tiered.cu:
 *   grid  : (ceil(N / WARPS_PER_BLOCK), M)
 *   block : (WARP_SIZE=32, WARPS_PER_BLOCK)
 *
 * Within each warp, lanes process groups in batches of WARP_SIZE:
 *   - Lane l handles group g = g_base + l.
 *   - Cascaded ballot_sync determines each lane's tier.
 *   - Per-tier popcount gives the lane's position in that tier's data stream.
 *   - Each lane accumulates VECTOR_LEN dot-products with the looked-up
 * centroid.
 *   - __shfl_down_sync reduces all lanes' partial sums to lane 0.
 *
 * Template parameters
 * -------------------
 *   TIER0_BITS : bit-width of tier-0 (hot) indices (compile-time unpack).
 *   VECTOR_LEN : VQ vector length (2 | 4 | 8 | 16).
 *   scalar_t   : __half or float.
 *
 *   args.cache_cents (runtime): number of centroids cached in shared
 *     memory.  Typically tier_thresholds[0] + tier_thresholds[1], covering
 *     ~80-85% of all lookups at Zipf s=1.2.  Capped at 32KB / (VL*2).
 *
 * Non-tier-0 bit widths are dispatched at runtime via unpack_bits_rt().
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>
#include <optional>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define MAX_TIERS 4

// ---------------------------------------------------------------------------
// Shared helpers (identical to vq_linear_tiered.cu)
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

/* Compile-time unpack (same as vq_linear_tiered.cu). */
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

/* Runtime-dispatched unpack for non-tier-0 streams. */
__device__ __forceinline__ int unpack_bits_rt(const uint8_t *__restrict__ buf,
                                              int pos, int bits) {
  // Fast paths for common widths.
  switch (bits) {
  case 4:
    return unpack_bits<4>(buf, pos);
  case 8:
    return unpack_bits<8>(buf, pos);
  case 16:
    return unpack_bits<16>(buf, pos);
  }
  // Generic bit-stream decode (runtime bits).
  const int bit_start = pos * bits;
  int result = 0;
  for (int b = 0; b < bits; b++) {
    const int abs_bit = bit_start + b;
    result |= ((int)((__ldg(&buf[abs_bit >> 3]) >> (abs_bit & 7)) & 1)) << b;
  }
  return result;
}

/* Read one bit from a 1-bit-per-element bitstream. */
__device__ __forceinline__ int read_prefix_bit(const uint8_t *__restrict__ buf,
                                               int pos) {
  return (int)((__ldg(&buf[pos >> 3]) >> (pos & 7)) & 1);
}

/* Read one bit from shared-memory prefix cache. */
__device__ __forceinline__ int read_prefix_bit_smem(const uint8_t *buf,
                                                    int pos) {
  return (int)((buf[pos >> 3] >> (pos & 7)) & 1);
}

// ---------------------------------------------------------------------------
// Vectorized dot-product helpers (same as vq_linear_tiered.cu)
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
 * Shared memory layout:
 *   1. Centroid cache: cache_cents * VECTOR_LEN * sizeof(__half)
 *      Caches the first cache_cents centroids (tier-0 + tier-1) in shared
 *      memory.  cache_cents = tier_thresholds[0] + tier_thresholds[1],
 *      covering ~80-85% of all lookups at Zipf s=1.2.  Capped at 32KB.
 *   2. Per-warp prefix_0 cache: WARPS_PER_BLOCK * ceil(n_groups / 8)
 */

#define MAX_PREFIX_SMEM_GROUPS 8192
#define PREFIX_SMEM_BYTES(ng) (((ng) + 7) / 8)

/*
 * Kernel arguments struct — avoids long parameter lists.
 * D-indexed arrays are sized MAX_TIERS; only [0..n_tiers-1] are used.
 */
struct NTierArgs {
  const uint8_t *prefix_ptrs[MAX_TIERS - 1]; // D-1 prefix streams
  int prefix_counts[MAX_TIERS - 1];          // n_remaining per prefix level
  const uint8_t *tier_ptrs[MAX_TIERS];       // D tier data streams
  int tier_bits[MAX_TIERS];                  // bit widths per tier
  const int32_t *row_off_ptrs[MAX_TIERS];    // D row-offset arrays [N+1]
  int tier_thresholds[MAX_TIERS];            // cumulative centroid thresholds
  int n_tiers;
  int cache_cents; // number of centroids cached in smem (tier-0 + tier-1)
};

template <int N_TIERS, int TIER0_BITS, int VECTOR_LEN, typename scalar_t>
__global__ void _vq_linear_ntier(
    const scalar_t *__restrict__ x,       // [M, K]
    const __half *__restrict__ centroids, // [num_centroids, VECTOR_LEN]
    const float *__restrict__ bias,       // [N] or nullptr
    scalar_t *__restrict__ y,             // [M, N]
    int M, int N, int n_groups, NTierArgs args) {
  const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;
  const int m = blockIdx.y;
  const int lane = threadIdx.x;
  const int K = n_groups * VECTOR_LEN;
  const int tid = threadIdx.y * WARP_SIZE + lane;
  const int block_threads = WARPS_PER_BLOCK * WARP_SIZE;
  constexpr int D = N_TIERS;
  const int cache_cents = args.cache_cents;

  // --- Shared memory partitioning ---
  extern __shared__ uint8_t smem_raw[];
  const int cache_bytes = cache_cents * VECTOR_LEN * (int)sizeof(__half);
  __half *cached_cent = reinterpret_cast<__half *>(smem_raw);

  const int prefix_bytes = PREFIX_SMEM_BYTES(n_groups);
  const bool use_prefix_smem = (n_groups <= MAX_PREFIX_SMEM_GROUPS);
  uint8_t *prefix0_base = smem_raw + cache_bytes;
  uint8_t *my_prefix0 =
      use_prefix_smem ? (prefix0_base + threadIdx.y * prefix_bytes) : nullptr;

  // --- Cooperative load: cached centroids into shared memory ---
  // Caches tier-0 + tier-1 (or more) centroids for fast smem lookups.
  const int cache_elems = cache_cents * VECTOR_LEN;
  for (int i = tid; i < cache_elems; i += block_threads) {
    cached_cent[i] = __ldg(&centroids[i]);
  }

  // Early exit AFTER hot centroid load (all threads must participate).
  if (n >= N || m >= M) {
    __syncthreads();
    return;
  }

  // --- Per-warp prefix_0 load ---
  if (use_prefix_smem) {
    const int row_base_byte = (n * n_groups) >> 3;
    for (int i = lane; i < prefix_bytes; i += WARP_SIZE) {
      my_prefix0[i] = __ldg(&args.prefix_ptrs[0][row_base_byte + i]);
    }
  }

  __syncthreads();
  if (use_prefix_smem)
    __syncwarp();

  // --- Load row-level base offsets (lane-0 broadcast) ---
  int tier_off[MAX_TIERS];
  if (lane == 0) {
    for (int k = 0; k < D; k++) {
      tier_off[k] = __ldg(&args.row_off_ptrs[k][n]);
    }
  }
  for (int k = 0; k < D; k++) {
    tier_off[k] = __shfl_sync(0xFFFFFFFF, tier_off[k], 0);
  }

  // Running offset into prefix streams 1..D-2.
  // prefix_off[k] tracks position in prefix_ptrs[k] for k >= 1.
  int prefix_off[MAX_TIERS - 1];
  // prefix_off[0] is tracked implicitly (flat indexing by n*n_groups+g).
  // For k >= 1: initialise from cumulative rest counts before this row.
  // rest_before_row[k] = sum of tier_{k..D-1} counts in rows 0..n-1
  //                    = sum_{j=k}^{D-1} row_off_ptrs[j][n]
  for (int k = 1; k < D - 1; k++) {
    int cumulative_rest = 0;
    for (int j = k; j < D; j++) {
      cumulative_rest += tier_off[j];
    }
    prefix_off[k] = cumulative_rest;
  }

  const scalar_t *x_row = &x[(size_t)m * K];
  float acc = 0.f;

  // Running rest-population offsets for cascaded prefix advancing.
  // After each warp batch we advance prefix_off[k] by the count of
  // rest-k-1 indices in that batch.
  int rest_count[MAX_TIERS]; // rest_count[k] = remaining after removing tiers
                             // 0..k-1 in this batch

  for (int g_base = 0; g_base < n_groups; g_base += WARP_SIZE) {
    const int g = g_base + lane;
    const bool valid = (g < n_groups);

    // ------ Cascaded tier assignment ------
    // Each lane determines its tier_id via cascaded prefix bits.
    int tier_id = D - 1; // default: last tier (coldest)

    // Level 0: prefix_0 at flat position n*n_groups + g.
    int is_rest = 0;
    if (valid) {
      if (use_prefix_smem) {
        is_rest = read_prefix_bit_smem(my_prefix0, g);
      } else {
        is_rest = read_prefix_bit(args.prefix_ptrs[0], n * n_groups + g);
      }
    }
    if (!is_rest && valid)
      tier_id = 0;

    uint32_t rest_ballot = __ballot_sync(0xFFFFFFFF, is_rest && valid);
    const uint32_t lane_mask = (1u << lane) - 1u;

    // Levels 1..D-2: cascaded prefix among remaining population.
    for (int k = 1; k < D - 1; k++) {
      // Only lanes that are still "rest" read prefix_k.
      int is_tier_k = 0;
      if (is_rest && valid) {
        int pos = prefix_off[k] + (int)__popc(rest_ballot & lane_mask);
        int pval = read_prefix_bit(args.prefix_ptrs[k], pos);
        is_tier_k = (pval == 0); // 0 = this tier, 1 = deeper rest
      }
      if (is_tier_k) {
        tier_id = k;
        is_rest = 0; // no longer in rest population
      }

      // Update rest_ballot: only lanes still in rest.
      rest_ballot = __ballot_sync(0xFFFFFFFF, is_rest && valid);
    }

    // ------ Per-tier ballots (computed ONCE, reused below) ------
    uint32_t tier_ballot[MAX_TIERS];
    for (int k = 0; k < D; k++) {
      tier_ballot[k] = __ballot_sync(0xFFFFFFFF, valid && tier_id == k);
    }

    // ------ Per-tier stream position via popcount ------
    int cidx = 0;
    if (valid) {
      int my_pos =
          tier_off[tier_id] + (int)__popc(tier_ballot[tier_id] & lane_mask);

      // Unpack centroid index from the appropriate tier stream.
      if (tier_id == 0) {
        cidx = unpack_bits<TIER0_BITS>(args.tier_ptrs[0], my_pos);
      } else {
        int lo = args.tier_thresholds[tier_id - 1];
        cidx = unpack_bits_rt(args.tier_ptrs[tier_id], my_pos,
                              args.tier_bits[tier_id]) +
               lo;
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

    // ------ Advance running offsets (reuse cached ballots) ------
    const int batch_n = min(WARP_SIZE, n_groups - g_base);
    const uint32_t vmask =
        (batch_n < WARP_SIZE) ? ((1u << batch_n) - 1u) : 0xFFFFFFFFu;

    // Advance tier stream offsets.
    for (int k = 0; k < D; k++) {
      tier_off[k] += (int)__popc(tier_ballot[k] & vmask);
    }

    // Advance prefix stream offsets for levels 1..D-2.
    for (int k = 1; k < D - 1; k++) {
      int rest_k = 0;
      for (int j = k; j < D; j++) {
        rest_k += (int)__popc(tier_ballot[j] & vmask);
      }
      prefix_off[k] += rest_k;
    }
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

static inline dim3 _ntgrid(int N, int M) {
  return dim3((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, (unsigned)M);
}
static const dim3 _NT_BLOCK(WARP_SIZE, WARPS_PER_BLOCK);

template <int N_TIERS, int TIER0_BITS, typename scalar_t>
static void _ntier_launch_dt0(const scalar_t *x, const __half *centroids,
                              const float *bias, scalar_t *y, int M, int N,
                              int n_groups, int vector_len, NTierArgs args,
                              cudaStream_t stream) {
  const dim3 grid = _ntgrid(N, M);

  const size_t cache_bytes =
      (size_t)args.cache_cents * vector_len * sizeof(__half);
  const size_t prefix_bytes_per_warp =
      (n_groups <= MAX_PREFIX_SMEM_GROUPS) ? PREFIX_SMEM_BYTES(n_groups) : 0;
  const size_t smem_bytes =
      cache_bytes + (size_t)WARPS_PER_BLOCK * prefix_bytes_per_warp;

#define NTL(VL)                                                                \
  _vq_linear_ntier<N_TIERS, TIER0_BITS, VL, scalar_t>                          \
      <<<grid, _NT_BLOCK, smem_bytes, stream>>>(x, centroids, bias, y, M, N,   \
                                                n_groups, args)

  switch (vector_len) {
  case 2:
    NTL(2);
    break;
  case 4:
    NTL(4);
    break;
  case 8:
    NTL(8);
    break;
  case 16:
    NTL(16);
    break;
  default:
    TORCH_CHECK(false, "vq_linear_ntier: unsupported vector_len=", vector_len,
                ".  Must be in {2,4,8,16}.");
  }
#undef NTL
}

template <int N_TIERS, typename scalar_t>
static void _ntier_launch_d(const scalar_t *x, const __half *centroids,
                            const float *bias, scalar_t *y, int M, int N,
                            int n_groups, int vector_len, NTierArgs args,
                            cudaStream_t stream) {
  int t0_bits = args.tier_bits[0];
  switch (t0_bits) {
  case 4:
    _ntier_launch_dt0<N_TIERS, 4, scalar_t>(x, centroids, bias, y, M, N,
                                            n_groups, vector_len, args, stream);
    break;
  case 5:
    _ntier_launch_dt0<N_TIERS, 5, scalar_t>(x, centroids, bias, y, M, N,
                                            n_groups, vector_len, args, stream);
    break;
  case 6:
    _ntier_launch_dt0<N_TIERS, 6, scalar_t>(x, centroids, bias, y, M, N,
                                            n_groups, vector_len, args, stream);
    break;
  case 7:
    _ntier_launch_dt0<N_TIERS, 7, scalar_t>(x, centroids, bias, y, M, N,
                                            n_groups, vector_len, args, stream);
    break;
  case 8:
    _ntier_launch_dt0<N_TIERS, 8, scalar_t>(x, centroids, bias, y, M, N,
                                            n_groups, vector_len, args, stream);
    break;
  default:
    TORCH_CHECK(false, "vq_linear_ntier: unsupported tier0_bits=", t0_bits,
                ".  Must be in {4,5,6,7,8}.");
  }
}

template <typename scalar_t>
static void _ntier_launch(const scalar_t *x, const __half *centroids,
                          const float *bias, scalar_t *y, int M, int N,
                          int n_groups, int vector_len, NTierArgs args,
                          cudaStream_t stream) {
  switch (args.n_tiers) {
  case 2:
    _ntier_launch_d<2, scalar_t>(x, centroids, bias, y, M, N, n_groups,
                                 vector_len, args, stream);
    break;
  case 3:
    _ntier_launch_d<3, scalar_t>(x, centroids, bias, y, M, N, n_groups,
                                 vector_len, args, stream);
    break;
  case 4:
    _ntier_launch_d<4, scalar_t>(x, centroids, bias, y, M, N, n_groups,
                                 vector_len, args, stream);
    break;
  default:
    TORCH_CHECK(false, "vq_linear_ntier: unsupported n_tiers=", args.n_tiers,
                ".  Must be in {2,3,4}.");
  }
}

// ---------------------------------------------------------------------------
// Torch-facing function
// ---------------------------------------------------------------------------

/*
 * vq_linear_ntier(x, prefix_streams, tier_streams, row_offsets,
 *                 centroids, tier_bits, vector_len, bias=None)
 *
 * All tensors must be on the same CUDA device.
 * x               : [M, K]  float16 or float32
 * prefix_streams  : list of D-1 uint8 tensors (cascaded prefix bitstreams)
 * tier_streams    : list of D uint8 tensors (packed tier data)
 * row_offsets     : list of D int32 tensors, each [N+1]
 * centroids       : [num_centroids, vector_len]  float16
 * tier_bits       : list of D ints (bit widths per tier)
 * vector_len      : int — VQ vector length (2, 4, 8, or 16)
 * bias            : optional [N] float
 *
 * Returns [M, N] matching x dtype.
 */
torch::Tensor
vq_linear_ntier(torch::Tensor x, std::vector<torch::Tensor> prefix_streams,
                std::vector<torch::Tensor> tier_streams,
                std::vector<torch::Tensor> row_offsets, torch::Tensor centroids,
                std::vector<int64_t> tier_bits_vec, int vector_len,
                std::optional<torch::Tensor> bias_opt) {
  TORCH_CHECK(x.is_cuda(), "vq_linear_ntier: x must be on CUDA.");
  TORCH_CHECK(x.dim() == 2, "vq_linear_ntier: x must be 2-D [M, K].");
  TORCH_CHECK(x.dtype() == torch::kFloat16 || x.dtype() == torch::kFloat32,
              "vq_linear_ntier: x must be float16 or float32.");
  TORCH_CHECK(centroids.dtype() == torch::kFloat16,
              "vq_linear_ntier: centroids must be float16.");

  const int D = (int)tier_streams.size();
  TORCH_CHECK(D >= 2 && D <= MAX_TIERS, "vq_linear_ntier: n_tiers=", D,
              " must be in [2, ", MAX_TIERS, "].");
  TORCH_CHECK((int)prefix_streams.size() == D - 1, "vq_linear_ntier: need ",
              D - 1, " prefix streams, got ", prefix_streams.size(), ".");
  TORCH_CHECK((int)row_offsets.size() == D, "vq_linear_ntier: need ", D,
              " row_offsets, got ", row_offsets.size(), ".");
  TORCH_CHECK((int)tier_bits_vec.size() == D, "vq_linear_ntier: need ", D,
              " tier_bits, got ", tier_bits_vec.size(), ".");

  const int M = (int)x.size(0);
  const int K = (int)x.size(1);
  TORCH_CHECK(K % vector_len == 0, "vq_linear_ntier: K=", K,
              " not divisible by vector_len=", vector_len);
  const int n_groups = K / vector_len;
  const int N = (int)row_offsets[0].size(0) - 1;
  TORCH_CHECK(N > 0, "vq_linear_ntier: row_offsets[0] too short.");

  // Build NTierArgs.
  NTierArgs args;
  args.n_tiers = D;

  for (int k = 0; k < D - 1; k++) {
    auto ps = prefix_streams[k].contiguous().cuda();
    TORCH_CHECK(ps.dtype() == torch::kUInt8, "vq_linear_ntier: prefix_streams[",
                k, "] must be uint8.");
    args.prefix_ptrs[k] = ps.data_ptr<uint8_t>();
    // Compute n_remaining at this level: sum of tier_k..D-1 counts.
    // We derive it from row_offsets: total = row_offsets[k][N] for tiers
    // k..D-1. Actually, prefix_counts[k] = sum of row_offsets[j][N] for j >= k.
    int n_remaining = 0;
    for (int j = k; j < D; j++) {
      n_remaining +=
          (int)row_offsets[j][(int)row_offsets[j].size(0) - 1].item<int32_t>();
    }
    args.prefix_counts[k] = n_remaining;
  }

  for (int k = 0; k < D; k++) {
    auto ts = tier_streams[k].contiguous().cuda();
    TORCH_CHECK(ts.dtype() == torch::kUInt8, "vq_linear_ntier: tier_streams[",
                k, "] must be uint8.");
    args.tier_ptrs[k] = ts.data_ptr<uint8_t>();

    auto ro = row_offsets[k].contiguous().cuda();
    TORCH_CHECK(ro.dtype() == torch::kInt32, "vq_linear_ntier: row_offsets[", k,
                "] must be int32.");
    args.row_off_ptrs[k] = ro.data_ptr<int32_t>();

    args.tier_bits[k] = (int)tier_bits_vec[k];
  }

  // Compute tier thresholds: threshold[k] = 2^tier_bits[k].
  for (int k = 0; k < D; k++) {
    args.tier_thresholds[k] = 1 << args.tier_bits[k];
  }

  // Cache tier-0 + tier-1 centroids in smem.
  // cache_cents = tier_thresholds[0] + tier_thresholds[1] (upper bound of
  // centroid indices reachable by tiers 0 and 1).
  // Cap at 32KB / (vector_len * 2 bytes) to fit within smem budget.
  const int num_centroids = (int)centroids.size(0);
  const int max_cache = (32 * 1024) / (vector_len * (int)sizeof(at::Half));
  int cache_cents = args.tier_thresholds[0];
  if (D >= 2) {
    cache_cents += args.tier_thresholds[1];
  }
  cache_cents = std::min(cache_cents, num_centroids);
  cache_cents = std::min(cache_cents, max_cache);
  args.cache_cents = cache_cents;

  auto xc = x.contiguous();
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

  // IMPORTANT: we must keep contiguous/cuda tensors alive until after the
  // kernel finishes.  Store them in a vector to prevent early destruction.
  // (The args struct holds raw pointers into these tensors.)
  std::vector<torch::Tensor> keepalive;
  for (int k = 0; k < D - 1; k++)
    keepalive.push_back(prefix_streams[k].contiguous().cuda());
  for (int k = 0; k < D; k++) {
    keepalive.push_back(tier_streams[k].contiguous().cuda());
    keepalive.push_back(row_offsets[k].contiguous().cuda());
  }

  // Re-extract pointers from keepalive (the first set may have been
  // temporaries).
  int ki = 0;
  for (int k = 0; k < D - 1; k++)
    args.prefix_ptrs[k] = keepalive[ki++].data_ptr<uint8_t>();
  for (int k = 0; k < D; k++) {
    args.tier_ptrs[k] = keepalive[ki++].data_ptr<uint8_t>();
    args.row_off_ptrs[k] = keepalive[ki++].data_ptr<int32_t>();
  }

  if (x.dtype() == torch::kFloat16) {
    _ntier_launch<__half>(
        reinterpret_cast<const __half *>(xc.data_ptr<at::Half>()), cent_ptr,
        bias_ptr, reinterpret_cast<__half *>(y.data_ptr<at::Half>()), M, N,
        n_groups, vector_len, args, stream);
  } else {
    _ntier_launch<float>(xc.data_ptr<float>(), cent_ptr, bias_ptr,
                         y.data_ptr<float>(), M, N, n_groups, vector_len, args,
                         stream);
  }

  return y;
}

// ---------------------------------------------------------------------------
// Pybind11 bindings
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_linear_ntier", &vq_linear_ntier,
        "Fused VQ n-tier CBS linear (cascaded ballot decode + centroid gather "
        "+ matmul).",
        py::arg("x"), py::arg("prefix_streams"), py::arg("tier_streams"),
        py::arg("row_offsets"), py::arg("centroids"), py::arg("tier_bits"),
        py::arg("vector_len"), py::arg("bias") = py::none());
}
