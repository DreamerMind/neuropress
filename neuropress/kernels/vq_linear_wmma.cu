/*
 * vq_linear_wmma.cu — Fused VQ linear with tensor-core GEMM and smem-cached
 * GEMV.  VL=4 fixed.  Dispatches automatically on batch dimension M:
 *
 *   M <= GEMV_THRESHOLD (8):  _vq_gemv  — warp-level dot, smem codebook.
 *   M >  GEMV_THRESHOLD (8):  _vq_gemm  — wmma 16×16×16, smem codebook +
 *                               per-K-step weight gather.  No intermediate
 *                               VRAM write of the weight matrix.
 *
 * Supported: bits ∈ {4, 8, 10, 12, 14, 15, 16} and any NC.
 * For NC ≤ NC_SMEM_MAX (4096): codebook is staged in shared memory (fast).
 * For NC > NC_SMEM_MAX: codebook accessed from global memory (L2-cached); no
 * hard limit on NC.  Shared-memory budget shrinks accordingly so smem_b / smem_a
 * / smem_out still fit.
 *
 * Porting to Rust / C++
 * ---------------------
 * The kernel logic is independent of PyTorch.  To use from a Rust inference
 * library, replace the torch::Tensor wrapper below with a plain extern-C
 * launcher that accepts raw device pointers + shape ints + cudaStream_t.
 * The kernels themselves (_vq_gemv, _vq_gemm) and their helpers need no
 * modification.
 *
 * Shared memory budget (per block):
 *   smem_cb  : NC × 4 × 2  bytes  (codebook; NC=1024→8 KB, NC=4096→32 KB)
 *   smem_b   : 16 × 64 × 2 bytes  (weight tile for GEMM; 2 KB)
 *   smem_a   : 16 × 16 × 2 bytes  (input tile for GEMM dtype conversion; 512 B)
 *   smem_out : 4 × 16 × 16 × 4 bytes  (fp32 accumulator staging; 4 KB)
 * Total (NC=4096): ~38.5 KB — fits sm_80+ (100 KB available).
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>
#include <torch/extension.h>
#include <optional>

using namespace nvcuda;

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

#define GEMV_THRESHOLD  8    // M <= this → GEMV kernel; M > this → GEMM kernel

// WMMA tile dimensions (fp16 input, fp32 accumulation)
#define WMMA_M  16
#define WMMA_N  16
#define WMMA_K  16

// GEMM block tile: one warp handles one [WMMA_M × WMMA_N] sub-tile
#define BLOCK_M    16          // = WMMA_M (one tile row per block)
#define BLOCK_N    64          // = 4 × WMMA_N (four warps, one per 16-col slice)
#define WARPS_GEMM  4

// GEMV block config: one warp per output element
#define WARP_SIZE   32
#define WARPS_GEMV   8

// Codebook size limit for shared memory caching.
// NC=4096, VL=4 → 32 KB, comfortably within sm_80+ budget.
#define NC_SMEM_MAX  4096

// VL is fixed at 4 throughout.
#define VL  4

namespace {

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

template <typename T> __device__ __forceinline__ float to_f32(T v);
template <> __device__ __forceinline__ float to_f32(__half v)         { return __half2float(v); }
template <> __device__ __forceinline__ float to_f32(__nv_bfloat16 v)  { return __bfloat162float(v); }
template <> __device__ __forceinline__ float to_f32(float v)          { return v; }

template <typename T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ __half        from_f32(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
template <> __device__ __forceinline__ float         from_f32(float v) { return v; }

// Convert any supported input type to fp16 (for wmma fragments)
template <typename T> __device__ __forceinline__ __half to_half(T v)       { return from_f32<__half>(to_f32(v)); }
template <> __device__ __forceinline__ __half           to_half(__half v)   { return v; }

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    val += __shfl_down_sync(0xFFFFFFFFu, val, off);
  return val;
}

// ---------------------------------------------------------------------------
// Index unpacking — BITS as compile-time template parameter.
// Bit-stream layout matches IndexPacking in vq_packing.py (LSB-first).
// ---------------------------------------------------------------------------

template <int BITS>
__device__ __forceinline__ int unpack_idx(const uint8_t* __restrict__ packed,
                                          int flat);

// 4-bit: two indices per byte, low-nibble first.
template <>
__device__ __forceinline__ int unpack_idx<4>(const uint8_t* p, int flat) {
  const uint8_t b = __ldg(&p[flat >> 1]);
  return (flat & 1) ? (int)(b >> 4) : (int)(b & 0xF);
}

// 8-bit: one index per byte.
template <>
__device__ __forceinline__ int unpack_idx<8>(const uint8_t* p, int flat) {
  return (int)__ldg(&p[flat]);
}

// 10-bit: covers NC=1024.  4 indices per 5 bytes; bit_off ∈ {0,2,4,6} so
// each index always spans exactly 2 consecutive bytes — clean 16-bit read.
template <>
__device__ __forceinline__ int unpack_idx<10>(const uint8_t* p, int flat) {
  const int bit_start  = flat * 10;
  const int byte_start = bit_start >> 3;
  const int bit_off    = bit_start & 7;
  const uint16_t val   = (uint16_t)__ldg(&p[byte_start])
                       | ((uint16_t)__ldg(&p[byte_start + 1]) << 8);
  return (int)((val >> bit_off) & 0x3FFu);
}

// 12-bit: two indices per 3 bytes, identical layout to vq_linear.cu.
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
// GEMV kernel — one warp per output element y[m, n].
//
// Thread mapping:
//   grid  : (⌈N / WARPS_GEMV⌉, M)
//   block : (WARP_SIZE=32, WARPS_GEMV=8)
//   lane g iterates over K-groups: g, g+32, g+64, …
//
// The entire codebook is loaded into shared memory once per block (all
// warps cooperate).  Centroid lookups then hit L1 smem rather than L2/VRAM.
// Half2 vectorized loads from x (VL=4 → two half2 per group).
// ---------------------------------------------------------------------------

template <int BITS, typename scalar_t>
__global__ void _vq_gemv(
    const scalar_t* __restrict__ x,          // [M, K]
    const uint8_t*  __restrict__ packed,     // [N * n_groups] packed
    const __half*   __restrict__ centroids,  // [NC, 4]
    const float*    __restrict__ bias,       // [N] or nullptr
    scalar_t*       __restrict__ y,          // [M, N]
    int M, int N, int n_groups, int NC
) {
  // Dynamic smem: NC × 4 fp16 when NC <= NC_SMEM_MAX; 0 bytes otherwise.
  extern __shared__ __half smem_cb[];

  const __half* cb;
  if (NC <= NC_SMEM_MAX) {
    // Cooperatively load codebook into shared memory.
    const int block_threads = blockDim.x * blockDim.y;  // 256
    const int tid_flat = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = tid_flat; i < NC * VL; i += block_threads)
      smem_cb[i] = __ldg(&centroids[i]);
    __syncthreads();
    cb = smem_cb;
  } else {
    // Large NC: codebook stays in global memory, L2-cached across warps.
    cb = centroids;
  }

  const int n = blockIdx.x * WARPS_GEMV + threadIdx.y;
  const int m = blockIdx.y;
  if (n >= N || m >= M) return;

  float acc = 0.f;
  const int    row_base = n * n_groups;
  const size_t x_base   = (size_t)m * n_groups * VL;

  for (int g = threadIdx.x; g < n_groups; g += WARP_SIZE) {
    const int    cidx = unpack_idx<BITS>(packed, row_base + g);
    const __half* cent = cb + (size_t)cidx * VL;

    const half2 c01 = *reinterpret_cast<const half2*>(cent);
    const half2 c23 = *reinterpret_cast<const half2*>(cent + 2);

    const scalar_t* xp = x + x_base + (size_t)g * VL;
    acc += to_f32(xp[0]) * __half2float(__low2half(c01));
    acc += to_f32(xp[1]) * __half2float(__high2half(c01));
    acc += to_f32(xp[2]) * __half2float(__low2half(c23));
    acc += to_f32(xp[3]) * __half2float(__high2half(c23));
  }

  acc = warp_reduce_sum(acc);
  if (threadIdx.x == 0) {
    if (bias) acc += bias[n];
    y[(size_t)m * N + n] = from_f32<scalar_t>(acc);
  }
}

// ---------------------------------------------------------------------------
// GEMM kernel — wmma 16×16×16 tiles, BLOCK_M=16, BLOCK_N=64, 4 warps.
//
// Each block computes a [16, 64] output tile.  Each of the 4 warps owns one
// [16, 16] wmma sub-tile along N.
//
// Per K-step (stride WMMA_K=16):
//   1. All 128 threads cooperatively gather the weight tile smem_b [16, 64]
//      from the packed index buffer + smem codebook.  No VRAM write.
//   2. All 128 threads load the input tile smem_a [16, 16], converting to fp16.
//   3. Each warp calls wmma::mma_sync, accumulating into fp32 fragment frag_c.
//
// After all K-steps: frag_c is staged through smem_out and written to y with
// boundary checking and optional bias addition.
//
// Bank conflicts: the gather writes smem_b with stride BLOCK_N=64 fp16 words,
// giving 2-way conflicts in the worst case — acceptable for v1.
// ---------------------------------------------------------------------------

template <int BITS, typename scalar_t>
__global__ void _vq_gemm(
    const scalar_t* __restrict__ x,          // [M, K]
    const uint8_t*  __restrict__ packed,     // [N * n_groups] packed
    const __half*   __restrict__ centroids,  // [NC, 4]
    const float*    __restrict__ bias,       // [N] or nullptr
    scalar_t*       __restrict__ y,          // [M, N]
    int M, int N, int n_groups, int K, int NC
) {
  // Dynamic smem layout (allocated by caller):
  //   [smem_cb  NC × 4 fp16]           only when NC <= NC_SMEM_MAX, else 0 bytes
  //   smem_b   [WMMA_K, BLOCK_N]       fp16 — weight tile (gathered each K-step)
  //   smem_a   [BLOCK_M, WMMA_K]       fp16 — input tile (x, dtype-converted)
  //   smem_out [WARPS_GEMM, WMMA_M, WMMA_N] fp32 — accumulator staging
  extern __shared__ __half smem[];
  const int   cb_elems = (NC <= NC_SMEM_MAX) ? NC * VL : 0;
  __half*     smem_b   = smem + cb_elems;
  __half*     smem_a   = smem_b  + WMMA_K * BLOCK_N;
  float*      smem_out = reinterpret_cast<float*>(smem_a + BLOCK_M * WMMA_K);
  const __half* cb     = (NC <= NC_SMEM_MAX) ? smem : centroids;

  const int block_threads = WARPS_GEMM * WARP_SIZE;  // 128
  const int tid_flat      = threadIdx.y * WARP_SIZE + threadIdx.x;
  const int warp_id       = threadIdx.y;              // 0..3

  const int m_start = blockIdx.y * BLOCK_M;
  const int n_start = blockIdx.x * BLOCK_N;

  // Load codebook into smem when it fits; otherwise use global memory (L2).
  if (NC <= NC_SMEM_MAX) {
    for (int i = tid_flat; i < NC * VL; i += block_threads)
      smem[i] = __ldg(&centroids[i]);
    __syncthreads();
  }

  // Each warp accumulates its [WMMA_M, WMMA_N] output tile in fp32.
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c;
  wmma::fill_fragment(frag_c, 0.f);

  // K-loop: one wmma tile (16 input features) per step.
  for (int k = 0; k < K; k += WMMA_K) {
    const int k_group_start = k / VL;  // first K-group index in this step

    // ------------------------------------------------------------------
    // Step 1: gather weight tile smem_b [WMMA_K=16][BLOCK_N=64].
    //
    // smem_b[k_row][n_col] = W^T[k_start+k_row][n_start+n_col]
    //                      = centroids[idx[n_start+n_col, k_group_start + k_row/VL]][k_row%VL]
    //
    // 4 K-groups × 64 N-cols = 256 index lookups, 128 threads → 2 rounds.
    // Thread flat owns flat_idx = flat + round*128.
    //   n_local      = flat_idx % BLOCK_N   — which N-column [0, 64)
    //   k_group_local = flat_idx / BLOCK_N   — which K-group [0,  4)
    // (This ordering keeps consecutive threads on consecutive N-cols, giving
    //  better coalescing on the packed index reads.)
    // ------------------------------------------------------------------
#pragma unroll
    for (int round = 0; round < 2; ++round) {
      const int flat_idx     = tid_flat + round * block_threads;
      const int n_local      = flat_idx % BLOCK_N;   // [0, 64)
      const int k_group_local = flat_idx / BLOCK_N;   // [0,  4)
      const int n_global     = n_start + n_local;
      const int k_group_global = k_group_start + k_group_local;

      if (n_global < N) {
        const int cidx  = unpack_idx<BITS>(packed, n_global * n_groups + k_group_global);
        const __half* cent = cb + (size_t)cidx * VL;
        // Write VL=4 centroid values into smem_b at rows [k_group_local*4 : k_group_local*4+4],
        // column n_local.  smem_b is row-major [WMMA_K][BLOCK_N].
        __half* dst = smem_b + k_group_local * VL * BLOCK_N + n_local;
        dst[0 * BLOCK_N] = cent[0];
        dst[1 * BLOCK_N] = cent[1];
        dst[2 * BLOCK_N] = cent[2];
        dst[3 * BLOCK_N] = cent[3];
      } else {
        __half* dst = smem_b + k_group_local * VL * BLOCK_N + n_local;
        dst[0 * BLOCK_N] = dst[1 * BLOCK_N] =
        dst[2 * BLOCK_N] = dst[3 * BLOCK_N] = __float2half(0.f);
      }
    }

    // ------------------------------------------------------------------
    // Step 2: load input tile smem_a [BLOCK_M=16][WMMA_K=16].
    // Convert to fp16 in smem (handles bf16 input transparently).
    // 16×16 = 256 elements, 128 threads → 2 elements per thread.
    // ------------------------------------------------------------------
    for (int i = tid_flat; i < BLOCK_M * WMMA_K; i += block_threads) {
      const int row      = i / WMMA_K;
      const int col      = i % WMMA_K;
      const int m_global = m_start + row;
      const int k_global = k + col;
      smem_a[i] = (m_global < M && k_global < K)
                  ? to_half(x[(size_t)m_global * K + k_global])
                  : __float2half(0.f);
    }
    __syncthreads();  // smem_b and smem_a fully written

    // ------------------------------------------------------------------
    // Step 3: wmma multiply-accumulate.
    // frag_a — shared A tile (same for all 4 warps).
    // frag_b — each warp loads its [WMMA_K, WMMA_N] column slice of smem_b.
    // ------------------------------------------------------------------
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> frag_b;

    wmma::load_matrix_sync(frag_a, smem_a, WMMA_K);
    // Warp w owns columns [w*WMMA_N, (w+1)*WMMA_N) of smem_b.
    wmma::load_matrix_sync(frag_b, smem_b + warp_id * WMMA_N, BLOCK_N);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);

    __syncthreads();  // All warps done; safe to overwrite smem next iteration.
  }

  // -----------------------------------------------------------------------
  // Store: stage fp32 accumulator through smem_out, then write to y.
  // Each warp's tile is at smem_out[warp_id * WMMA_M * WMMA_N].
  // -----------------------------------------------------------------------
  wmma::store_matrix_sync(smem_out + warp_id * WMMA_M * WMMA_N,
                           frag_c, WMMA_N, wmma::mem_row_major);
  __syncthreads();

  // Each warp copies its [WMMA_M, WMMA_N] tile to global y.
  for (int i = threadIdx.x; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
    const int row      = i / WMMA_N;
    const int col      = i % WMMA_N;
    const int m_global = m_start + row;
    const int n_global = n_start + warp_id * WMMA_N + col;
    if (m_global < M && n_global < N) {
      float val = smem_out[warp_id * WMMA_M * WMMA_N + i];
      if (bias) val += bias[n_global];
      y[(size_t)m_global * N + n_global] = from_f32<scalar_t>(val);
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

template <int BITS, typename scalar_t>
static void _launch_gemv(
    const scalar_t* x, const uint8_t* packed, const __half* cb,
    const float* bias, scalar_t* y,
    int M, int N, int n_groups, int NC, cudaStream_t stream
) {
  const dim3 grid((N + WARPS_GEMV - 1) / WARPS_GEMV, (unsigned)M);
  const dim3 block(WARP_SIZE, WARPS_GEMV);
  const size_t smem = (NC <= NC_SMEM_MAX) ? (size_t)NC * VL * sizeof(__half) : 0;
  _vq_gemv<BITS, scalar_t>
      <<<grid, block, smem, stream>>>(x, packed, cb, bias, y, M, N, n_groups, NC);
}

template <int BITS, typename scalar_t>
static void _launch_gemm(
    const scalar_t* x, const uint8_t* packed, const __half* cb,
    const float* bias, scalar_t* y,
    int M, int N, int n_groups, int K, int NC, cudaStream_t stream
) {
  const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
  const dim3 block(WARP_SIZE, WARPS_GEMM);
  const size_t smem_cb_bytes = (NC <= NC_SMEM_MAX) ? (size_t)NC * VL * sizeof(__half) : 0;
  const size_t smem =
      smem_cb_bytes                                              // smem_cb (0 for large NC)
    + (size_t)WMMA_K * BLOCK_N * sizeof(__half)                // smem_b
    + (size_t)BLOCK_M * WMMA_K * sizeof(__half)                // smem_a
    + (size_t)WARPS_GEMM * WMMA_M * WMMA_N * sizeof(float);   // smem_out
  _vq_gemm<BITS, scalar_t>
      <<<grid, block, smem, stream>>>(x, packed, cb, bias, y, M, N, n_groups, K, NC);
}

// Dispatch on (bits, M<=threshold?GEMV:GEMM)
template <typename scalar_t>
static void _vq_dispatch(
    const scalar_t* x, const uint8_t* packed, const __half* cb,
    const float* bias, scalar_t* y,
    int M, int N, int K, int NC, int bits, cudaStream_t stream
) {
  const int n_groups = K / VL;

#define DISPATCH(B)                                                           \
  do {                                                                        \
    if (M <= GEMV_THRESHOLD)                                                  \
      _launch_gemv<B>(x, packed, cb, bias, y, M, N, n_groups, NC, stream);   \
    else                                                                      \
      _launch_gemm<B>(x, packed, cb, bias, y, M, N, n_groups, K, NC, stream);\
  } while (0)

  switch (bits) {
    case  4: DISPATCH( 4); return;
    case  8: DISPATCH( 8); return;
    case 10: DISPATCH(10); return;
    case 12: DISPATCH(12); return;
    case 14: DISPATCH(14); return;
    case 15: DISPATCH(15); return;
    case 16: DISPATCH(16); return;
  }
#undef DISPATCH
  TORCH_CHECK(false, "vq_linear_wmma: unsupported bits=", bits,
              ". Must be 4, 8, 10, 12, 14, 15, or 16.");
}

// ---------------------------------------------------------------------------
// Torch-facing entry point
// ---------------------------------------------------------------------------

torch::Tensor vq_linear_wmma(
    torch::Tensor                x,               // [M, K]  fp16 or bf16
    torch::Tensor                packed_indices,  // [packed bytes] uint8
    torch::Tensor                centroids,       // [NC, 4] fp16
    int                          bits,
    std::optional<torch::Tensor> bias_opt
) {
  TORCH_CHECK(x.is_cuda() && packed_indices.is_cuda() && centroids.is_cuda(),
              "vq_linear_wmma: all tensors must be on CUDA.");
  TORCH_CHECK(x.is_contiguous() && packed_indices.is_contiguous()
              && centroids.is_contiguous(),
              "vq_linear_wmma: all tensors must be contiguous.");
  TORCH_CHECK(x.dim() == 2, "vq_linear_wmma: x must be 2-D [M, K].");
  TORCH_CHECK(centroids.dtype() == torch::kFloat16,
              "vq_linear_wmma: centroids must be float16.");
  TORCH_CHECK(bits == 4 || bits == 8 || bits == 10 || bits == 12
              || bits == 14 || bits == 15 || bits == 16,
              "vq_linear_wmma: bits must be 4, 8, 10, 12, 14, 15, or 16, got ", bits, ".");

  const int M = x.size(0);
  const int K = x.size(1);
  const int NC = centroids.size(0);

  TORCH_CHECK(K % VL == 0,
              "vq_linear_wmma: K=", K, " not divisible by VL=", VL, ".");
  TORCH_CHECK(M > GEMV_THRESHOLD || K % WMMA_K == 0 || true,  // GEMV has no K restriction
              "vq_linear_wmma: K=", K, " must be divisible by WMMA_K=", WMMA_K, " for GEMM.");
  if (M > GEMV_THRESHOLD)
    TORCH_CHECK(K % WMMA_K == 0,
                "vq_linear_wmma: GEMM path requires K%", WMMA_K, "==0, got K=", K, ".");
  // NC > NC_SMEM_MAX: codebook accessed from global memory (L2-cached).
  // No hard limit — any NC is valid.

  const long long total_bits = (long long)packed_indices.numel() * 8LL;
  const int n_groups = K / VL;
  const int total_indices = (int)(total_bits / bits);
  TORCH_CHECK(total_indices % n_groups == 0,
              "vq_linear_wmma: packed_indices size does not evenly divide into rows.");
  const int N = total_indices / n_groups;

  auto y = torch::empty({M, N}, x.options());

  const float* bias_ptr = nullptr;
  torch::Tensor bias_f32;
  if (bias_opt.has_value()) {
    bias_f32 = bias_opt.value().to(torch::kFloat32).contiguous();
    bias_ptr = bias_f32.data_ptr<float>();
  }

  const auto    stream   = at::cuda::getCurrentCUDAStream();
  const __half* cb_ptr   = reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>());
  const uint8_t* pk_ptr  = packed_indices.data_ptr<uint8_t>();

  if (x.dtype() == torch::kFloat16) {
    _vq_dispatch(
        reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
        pk_ptr, cb_ptr, bias_ptr,
        reinterpret_cast<__half*>(y.data_ptr<at::Half>()),
        M, N, K, NC, bits, stream);
  } else if (x.dtype() == torch::kBFloat16) {
    _vq_dispatch(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        pk_ptr, cb_ptr, bias_ptr,
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        M, N, K, NC, bits, stream);
  } else {
    TORCH_CHECK(false,
                "vq_linear_wmma: x must be float16 or bfloat16, got ",
                x.dtype(), ".");
  }

  return y;
}

// ---------------------------------------------------------------------------
// Python bindings
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_linear_wmma", &vq_linear_wmma,
        "Fused VQ linear: GEMV (M≤8, smem codebook, warp reduction) or "
        "GEMM (M>8, wmma 16×16×16, no VRAM weight materialisation). "
        "VL=4 fixed. bits ∈ {4,8,10,12,14,15,16}. "
        "For NC>4096 (bits>12) route to LUT kernel instead (WMMA smem limit).",
        py::arg("x"),
        py::arg("packed_indices"),
        py::arg("centroids"),
        py::arg("bits"),
        py::arg("bias") = py::none());
}
