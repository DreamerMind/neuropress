/*
 * Fused k-means E-step kernel.
 *
 * For each query vector x[n] in [N, D], computes:
 *   labels[n] = argmin_{c=0..k-1}  ||x[n] - centroids[c]||^2
 *
 * Thread mapping — one warp per query:
 *   grid  : (ceil(N / WARPS_PER_BLOCK), 1)
 *   block : (WARP_SIZE=32, WARPS_PER_BLOCK=8)
 *
 * Centroids are loaded cooperatively into shared memory in TILE-centroid
 * chunks; all warps in the block process the same tile for different queries.
 * TILE=2048 at D=4: 2048×4×4=32 KB — within 48 KB shmem limit, reduces
 * pass count to 2 at k=4k, 8 at k=16k, 32 at k=64k.
 *
 * Template parameters
 * -------------------
 *   D  : vector dimensionality — 4 (production VL). D=8 would need TILE=1024
 *        to stay within 48 KB; not instantiated since it is not used.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <float.h>
#include <optional>
#include <stdint.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
// TILE=2048 at D=4: 2048×4×4=32 KB shmem — within 48 KB limit, halves pass
// count vs TILE=1024. Reduces to 8 passes at k=16k, 2 at k=4k.
#define TILE 2048

namespace {

template <int D>
__global__ void _kmeans_assign(
    const float* __restrict__ x,          // [N, D]
    const float* __restrict__ centroids,  // [k, D]
    int* __restrict__ labels,             // [N]
    int N, int k,
    const float* __restrict__ bias)       // [k] or nullptr — subtracted from dist before argmin
{
    const int n = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;

    // Shared memory: TILE centroids × D floats, shared across all warps.
    __shared__ float shmem[TILE * D];

    // Load query vector into registers (D is small, fully unrolled).
    float xv[D];
    if (n < N) {
#pragma unroll
        for (int d = 0; d < D; d++)
            xv[d] = __ldg(&x[n * D + d]);
    }

    float best_dist = FLT_MAX;
    int   best_idx  = 0;

    const int tid           = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int block_threads = WARPS_PER_BLOCK * WARP_SIZE;  // 256

    for (int tile_start = 0; tile_start < k; tile_start += TILE) {
        const int tile_end  = min(tile_start + TILE, k);
        const int tile_size = tile_end - tile_start;

        // Cooperatively load tile into shared memory.
        for (int i = tid; i < tile_size * D; i += block_threads)
            shmem[i] = __ldg(&centroids[(tile_start + i / D) * D + (i % D)]);
        __syncthreads();

        if (n < N) {
            // Each lane strides over tile centroids with step WARP_SIZE.
            for (int c = threadIdx.x; c < tile_size; c += WARP_SIZE) {
                float dist = 0.f;
#pragma unroll
                for (int d = 0; d < D; d++) {
                    float diff = xv[d] - shmem[c * D + d];
                    dist += diff * diff;
                }
                if (bias != nullptr)
                    dist -= __ldg(&bias[tile_start + c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx  = tile_start + c;
                }
            }
        }
        __syncthreads();
    }

    if (n >= N) return;

    // Warp-reduce: find global argmin across lanes.
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
        int   other_idx  = __shfl_down_sync(0xFFFFFFFF, best_idx,  offset);
        if (other_dist < best_dist) {
            best_dist = other_dist;
            best_idx  = other_idx;
        }
    }

    if (threadIdx.x == 0)
        labels[n] = best_idx;
}

// ---------------------------------------------------------------------------
// Grouped kernel: handles all G groups in a single launch using blockIdx.y
// as the group dimension.
//
// Grid:  (ceil(R / WARPS_PER_BLOCK), G)
// Block: (WARP_SIZE=32, WARPS_PER_BLOCK=8)
//
// x   : [R, G, D] contiguous float32
// C   : [NC, D]   contiguous float32
// imp : [G, D]    contiguous float32 - sqrt(importance) per group-feature
// labels : [R, G] output int32
// ---------------------------------------------------------------------------

template <int D>
__global__ void _kmeans_assign_grouped(
    const float* __restrict__ x,          // [R, G, D]
    const float* __restrict__ C,          // [NC, D]
    const float* __restrict__ imp,        // [G, D]
    int* __restrict__ labels,             // [R, G]
    int R, int G, int NC,
    const float* __restrict__ bias)       // [NC] or nullptr
{
    const int g   = blockIdx.y;
    const int row = blockIdx.x * WARPS_PER_BLOCK + threadIdx.y;

    __shared__ float shmem[TILE * D];

    // Load query vector scaled by importance for this group.
    float xv[D];
    if (row < R) {
#pragma unroll
        for (int d = 0; d < D; d++)
            xv[d] = __ldg(&x[(row * G + g) * D + d]) * __ldg(&imp[g * D + d]);
    }

    float best_dist = FLT_MAX;
    int   best_idx  = 0;

    const int tid           = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int block_threads = WARPS_PER_BLOCK * WARP_SIZE;  // 256

    for (int tile_start = 0; tile_start < NC; tile_start += TILE) {
        const int tile_end  = min(tile_start + TILE, NC);
        const int tile_size = tile_end - tile_start;

        // Cooperatively load tile into shared memory, scaling by imp[g].
        for (int i = tid; i < tile_size * D; i += block_threads) {
            const int ci = i / D;
            const int di = i % D;
            shmem[i] = __ldg(&C[(tile_start + ci) * D + di]) * __ldg(&imp[g * D + di]);
        }
        __syncthreads();

        if (row < R) {
            for (int c = threadIdx.x; c < tile_size; c += WARP_SIZE) {
                float dist = 0.f;
#pragma unroll
                for (int d = 0; d < D; d++) {
                    float diff = xv[d] - shmem[c * D + d];
                    dist += diff * diff;
                }
                if (bias != nullptr)
                    dist -= __ldg(&bias[tile_start + c]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx  = tile_start + c;
                }
            }
        }
        __syncthreads();
    }

    if (row >= R) return;

    // Warp-reduce: find global argmin across lanes.
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_dist = __shfl_down_sync(0xFFFFFFFF, best_dist, offset);
        int   other_idx  = __shfl_down_sync(0xFFFFFFFF, best_idx,  offset);
        if (other_dist < best_dist) {
            best_dist = other_dist;
            best_idx  = other_idx;
        }
    }

    if (threadIdx.x == 0)
        labels[row * G + g] = best_idx;
}

}  // namespace

// ---------------------------------------------------------------------------
// Torch-facing function
// ---------------------------------------------------------------------------

torch::Tensor kmeans_assign(
    torch::Tensor x,                            // [N, D] float32, CUDA
    torch::Tensor centroids,                    // [k, D] float32, CUDA
    int vector_len,                             // D: 4
    std::optional<torch::Tensor> bias = std::nullopt)  // [k] float32, CUDA, or None
{
    TORCH_CHECK(x.is_cuda() && centroids.is_cuda(),
                "kmeans_assign: tensors must be on CUDA.");
    TORCH_CHECK(x.is_contiguous() && centroids.is_contiguous(),
                "kmeans_assign: tensors must be contiguous.");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && centroids.dtype() == torch::kFloat32,
                "kmeans_assign: tensors must be float32.");
    TORCH_CHECK(x.dim() == 2 && centroids.dim() == 2,
                "kmeans_assign: tensors must be 2-D.");
    TORCH_CHECK(x.size(1) == centroids.size(1),
                "kmeans_assign: x and centroids must have the same D.");
    TORCH_CHECK(vector_len == 4,
                "kmeans_assign: only vector_len=4 is supported (D=8 exceeds "
                "shared-memory limits at TILE=2048), got ", vector_len);

    const int N = x.size(0);
    const int k = centroids.size(0);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto& b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.dtype() == torch::kFloat32,
                    "kmeans_assign: bias must be a contiguous float32 CUDA tensor.");
        TORCH_CHECK(b.numel() == k,
                    "kmeans_assign: bias must have k=", k, " elements, got ", b.numel());
        bias_ptr = b.data_ptr<float>();
    }

    auto labels = torch::empty({N}, x.options().dtype(torch::kInt32));

    const dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1);
    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    const auto stream = at::cuda::getCurrentCUDAStream();

    _kmeans_assign<4><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), centroids.data_ptr<float>(),
        labels.data_ptr<int>(), N, k, bias_ptr);

    return labels.to(torch::kInt64);
}

// ---------------------------------------------------------------------------
// Torch-facing grouped function
// ---------------------------------------------------------------------------

torch::Tensor kmeans_assign_grouped(
    torch::Tensor x,          // [R, G, D] float32 CUDA contiguous
    torch::Tensor C,          // [NC, D]   float32 CUDA contiguous
    torch::Tensor imp,        // [G, D]    float32 CUDA contiguous
    int vector_len,           // must be 4
    std::optional<torch::Tensor> bias = std::nullopt)  // [NC] or None
{
    TORCH_CHECK(x.is_cuda() && C.is_cuda() && imp.is_cuda(),
                "kmeans_assign_grouped: tensors must be on CUDA.");
    TORCH_CHECK(x.is_contiguous() && C.is_contiguous() && imp.is_contiguous(),
                "kmeans_assign_grouped: tensors must be contiguous.");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && C.dtype() == torch::kFloat32
                && imp.dtype() == torch::kFloat32,
                "kmeans_assign_grouped: tensors must be float32.");
    TORCH_CHECK(vector_len == 4,
                "kmeans_assign_grouped: only vector_len=4 is supported, got ",
                vector_len);
    TORCH_CHECK(x.dim() == 3,
                "kmeans_assign_grouped: x must be 3-D [R, G, D], got dim=",
                x.dim());
    TORCH_CHECK(C.dim() == 2,
                "kmeans_assign_grouped: C must be 2-D [NC, D], got dim=",
                C.dim());
    TORCH_CHECK(imp.dim() == 2,
                "kmeans_assign_grouped: imp must be 2-D [G, D], got dim=",
                imp.dim());

    const int D = vector_len;
    TORCH_CHECK(x.size(2) == D,
                "kmeans_assign_grouped: x.size(2)=", x.size(2),
                " must equal vector_len=", D);
    TORCH_CHECK(C.size(1) == D,
                "kmeans_assign_grouped: C.size(1)=", C.size(1),
                " must equal vector_len=", D);

    const int R  = (int)x.size(0);
    const int G  = (int)x.size(1);
    const int NC = (int)C.size(0);

    TORCH_CHECK(imp.size(0) == G,
                "kmeans_assign_grouped: imp.size(0)=", imp.size(0),
                " must equal G=", G);
    TORCH_CHECK(imp.size(1) == D,
                "kmeans_assign_grouped: imp.size(1)=", imp.size(1),
                " must equal vector_len=", D);

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto& b = bias.value();
        TORCH_CHECK(b.is_cuda() && b.is_contiguous() && b.dtype() == torch::kFloat32,
                    "kmeans_assign_grouped: bias must be a contiguous float32 CUDA tensor.");
        TORCH_CHECK(b.numel() == NC,
                    "kmeans_assign_grouped: bias must have NC=", NC,
                    " elements, got ", b.numel());
        bias_ptr = b.data_ptr<float>();
    }

    auto labels = torch::empty({R, G}, x.options().dtype(torch::kInt32));

    const dim3 grid((R + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, G);
    const dim3 block(WARP_SIZE, WARPS_PER_BLOCK);
    const auto stream = at::cuda::getCurrentCUDAStream();

    _kmeans_assign_grouped<4><<<grid, block, 0, stream>>>(
        x.data_ptr<float>(), C.data_ptr<float>(), imp.data_ptr<float>(),
        labels.data_ptr<int>(), R, G, NC, bias_ptr);

    return labels.to(torch::kInt64);
}

// ---------------------------------------------------------------------------
// Bit-stream index packer: replaces the Python for-loop in _pack_bitstream.
// Each thread packs one index into a LSB-first bit-stream.
// out[] must be zeroed before launch; concurrent ORs are safe because
// atomicOr only sets bits and CUDA's 32-bit atomicOr covers individual bytes
// via the aligned-word trick.
// ---------------------------------------------------------------------------

__global__ void _pack_indices_kernel(
    const int64_t* __restrict__ indices,
    uint8_t* __restrict__ out,
    int n_indices,
    int bits)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_indices) return;

    const int bit_start  = i * bits;
    const int byte_start = bit_start >> 3;
    const int bit_off    = bit_start & 7;

    // Shift the index value into its bit-position within a 3-byte window.
    const uint32_t val    = (uint32_t)(indices[i] & ((1u << bits) - 1u));
    const uint32_t shifted = val << bit_off;

    // Write up to 3 bytes using 32-bit atomicOr on the aligned containing word.
    // CUDA lacks native byte-level atomics; the aligned-word trick is standard.
    // Inlined helper: atomicOr one byte into the aligned 32-bit word.
    #define OR_BYTE(bidx, bval) do { \
        if (bval) { \
            uint32_t* _word = (uint32_t*)((uintptr_t)(out + (bidx)) & ~3UL); \
            const uint32_t _shift = ((uintptr_t)(out + (bidx)) & 3u) * 8u; \
            atomicOr(_word, ((bval) & 0xFFu) << _shift); \
        } \
    } while (0)

    OR_BYTE(byte_start,     (shifted)       );
    if (bits + bit_off >  8) OR_BYTE(byte_start + 1, (shifted >>  8));
    if (bits + bit_off > 16) OR_BYTE(byte_start + 2, (shifted >> 16));
    #undef OR_BYTE
}

torch::Tensor pack_indices(
    torch::Tensor indices,  // [n_indices] int64 CUDA
    int n_indices,
    int bits)
{
    TORCH_CHECK(indices.is_cuda() && indices.is_contiguous()
                && indices.dtype() == torch::kInt64,
                "pack_indices: indices must be contiguous int64 CUDA tensor.");
    TORCH_CHECK(bits >= 4 && bits <= 16,
                "pack_indices: bits must be in [4, 16], got ", bits);
    TORCH_CHECK(n_indices >= 0, "pack_indices: n_indices must be >= 0.");
    TORCH_CHECK(indices.numel() >= n_indices,
                "pack_indices: indices.numel() < n_indices.");

    const int n_bytes = (n_indices * bits + 7) / 8;
    // zeros() guarantees the output is clean before atomic ORs.
    auto out = torch::zeros({n_bytes},
                            indices.options().dtype(torch::kUInt8));
    if (n_indices == 0) return out;

    const dim3 block(256);
    const dim3 grid((n_indices + 255) / 256);
    const auto stream = at::cuda::getCurrentCUDAStream();

    _pack_indices_kernel<<<grid, block, 0, stream>>>(
        indices.data_ptr<int64_t>(),
        out.data_ptr<uint8_t>(),
        n_indices, bits);

    return out;
}

// ---------------------------------------------------------------------------
// Bit-stream index unpacker: replaces the Python for-loop in _unpack_bitstream.
// Each thread unpacks one index from a LSB-first bit-stream.
// Requires packed to have ≥3 zero-padded bytes past the last valid byte.
// ---------------------------------------------------------------------------

__global__ void _unpack_bitstream_kernel(
    const uint8_t* __restrict__ packed,
    int32_t* __restrict__ out,
    int n_indices,
    int bits)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_indices) return;
    const int bit_start  = i * bits;
    const int byte_start = bit_start >> 3;
    const int bit_off    = bit_start & 7;
    // 3-byte window covers any bits-wide index at any byte alignment (bits ≤ 16).
    const uint32_t window =
          (uint32_t)packed[byte_start]
        | ((uint32_t)packed[byte_start + 1] << 8)
        | ((uint32_t)packed[byte_start + 2] << 16);
    const uint32_t mask = (1u << bits) - 1u;
    out[i] = (int32_t)((window >> bit_off) & mask);
}

torch::Tensor unpack_indices(
    torch::Tensor packed,   // [n_bytes] uint8 CUDA, must have ≥3 zero-pad bytes at end
    int n_indices,
    int bits)
{
    TORCH_CHECK(packed.is_cuda() && packed.is_contiguous()
                && packed.dtype() == torch::kUInt8,
                "unpack_indices: packed must be contiguous uint8 CUDA tensor.");
    TORCH_CHECK(bits >= 4 && bits <= 16,
                "unpack_indices: bits must be in [4, 16], got ", bits);
    TORCH_CHECK(n_indices >= 0, "unpack_indices: n_indices must be >= 0.");

    auto out = torch::empty({n_indices},
                            packed.options().dtype(torch::kInt32));
    if (n_indices == 0) return out.to(torch::kInt64);

    const dim3 block(256);
    const dim3 grid((n_indices + 255) / 256);
    const auto stream = at::cuda::getCurrentCUDAStream();

    _unpack_bitstream_kernel<<<grid, block, 0, stream>>>(
        packed.data_ptr<uint8_t>(),
        out.data_ptr<int32_t>(),
        n_indices, bits);

    return out.to(torch::kInt64);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kmeans_assign", &kmeans_assign,
          "Fused k-means E-step: nearest-centroid assignment.",
          py::arg("x"), py::arg("centroids"), py::arg("vector_len"),
          py::arg("bias") = py::none());
    m.def("kmeans_assign_grouped", &kmeans_assign_grouped,
          "Fused grouped k-means E-step: all G groups in one launch.",
          py::arg("x"), py::arg("C"), py::arg("imp"), py::arg("vector_len"),
          py::arg("bias") = py::none());
    m.def("pack_indices", &pack_indices,
          "Pack N-bit indices into a LSB-first bit-stream (4 ≤ bits ≤ 16).",
          py::arg("indices"), py::arg("n_indices"), py::arg("bits"));
    m.def("unpack_indices", &unpack_indices,
          "Unpack a LSB-first bit-stream of N-bit indices (4 ≤ bits ≤ 16).",
          py::arg("packed"), py::arg("n_indices"), py::arg("bits"));
}
