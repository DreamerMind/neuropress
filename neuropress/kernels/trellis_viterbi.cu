/*
 * trellis_viterbi.cu - tail-biting Viterbi encoder for bitshift trellis codes.
 *
 * Ported from exllamav3 (exllamav3_ext/quant/quantize_tiles_kernel.cuh,
 * quantize_tiles_frac_kernel.cuh and codebook.cuh), under its MIT license:
 *
 *   MIT License
 *
 *   Copyright (c) 2025 Turboderp
 *
 *   Permission is hereby granted, free of charge, to any person obtaining a copy
 *   of this software and associated documentation files (the "Software"), to deal
 *   in the Software without restriction, including without limitation the rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:
 *
 *   The above copyright notice and this permission notice shall be included in all
 *   copies or substantial portions of the Software.
 *
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *   SOFTWARE.
 *
 * Changes from the original: one kernel covers integer and fractional rates
 * (the step width D(i) = KA + bit (i mod 16) of MASK, MASK = 0 for integer K),
 * any tile length L divisible by 32, a fourth codebook read from a 65536-entry
 * table (learned codebooks), and the traceback history stores the branch index
 * (one byte) instead of the 16-bit in-node. The cost arithmetic, the candidate
 * order and the final argmin tie order are those of the original, so with the
 * same codebook the selected states are identical.
 *
 * Code. A tile of L values is a closed walk through a 16-bit shift register:
 * position i shifts D(i) new bits in, and its 16-bit window (the register
 * content) decodes to a value through the codebook. The node between positions
 * i and i+1 is the (16 - D(i+1))-bit overlap of their windows. Viterbi runs two
 * passes: the first, rolled by L/2, finds a good state at position 0; the second
 * is pinned to it, so the walk closes on itself (tail-biting).
 *
 * Codebooks (CB): 0 = 3INST, 1 = MCG, 2 = MUL1 (QTIP / EXL3 hash codebooks),
 * 3 = LUT (lut[state], fp16).
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define VT_THREADS 512
#define VT_H_INF __ushort_as_half(0x7c00)

// ---------------------------------------------------------------------------
// Codebooks
// ---------------------------------------------------------------------------

union vt_half2_u32 { uint32_t u; half2 h; __device__ vt_half2_u32(uint32_t v) : u(v) {} };
union vt_half_u16 { uint16_t u; half h; __device__ vt_half_u16(uint16_t v) : u(v) {} };

// (x & 0x8fff8fff) ^ 0x3b603b60: two fp16 bit patterns in [-~1.9, ~1.9]
__device__ __forceinline__ uint32_t vt_lop3(uint32_t x)
{
    asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x));
    return x;
}

__device__ __forceinline__ half vt_sum_halves(uint32_t x)
{
    vt_half2_u32 xu(vt_lop3(x));
    return __hadd(__low2half(xu.h), __high2half(xu.h));
}

template <int CB>
__device__ __forceinline__ half vt_decode(uint32_t x, const half* __restrict__ lut)
{
    if constexpr (CB == 0)
        return vt_sum_halves(x * 89226354u + 64248484u);
    else if constexpr (CB == 1)
        return vt_sum_halves(x * 0xCBAC1FEDu);
    else if constexpr (CB == 2)
    {
        // Byte sum of the product, read as fp16 1024 + sum, then an affine map to ~unit variance
        const uint32_t sum = __dp4a(x * 0x83DCD12Du, 0x01010101u, 0x6400u);
        vt_half_u16 h((uint16_t) sum);
        return __hfma(h.h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
    }
    else
        return lut[x & 0xffff];
}

// Hash multiplier of the MCG / MUL1 codebooks, 0 for the others
template <int CB>
__host__ __device__ constexpr uint32_t vt_mult() { return CB == 1 ? 0xCBAC1FEDu : CB == 2 ? 0x83DCD12Du : 0u; }

// Two adjacent states from their hash products p = state * vt_mult<CB>() (MCG / MUL1), else from the
// states themselves. Products are linear in the state modulo 2^32, so the candidate loop advances
// them by one addition instead of a multiplication per state
template <int CB>
__device__ __forceinline__ half2 vt_decode2(uint32_t x0, uint32_t x1, const half* __restrict__ lut)
{
    if constexpr (CB == 1)
    {
        vt_half2_u32 a(vt_lop3(x0)), b(vt_lop3(x1));
        return __hadd2(__lows2half2(a.h, b.h), __highs2half2(a.h, b.h));
    }
    else if constexpr (CB == 2)
    {
        vt_half_u16 h0((uint16_t) __dp4a(x0, 0x01010101u, 0x6400u));
        vt_half_u16 h1((uint16_t) __dp4a(x1, 0x01010101u, 0x6400u));
        return __hfma2(__halves2half2(h0.h, h1.h), __half2half2(__ushort_as_half(0x1eee)),
                       __half2half2(__ushort_as_half(0xc931)));
    }
    else
        return __halves2half2(vt_decode<CB>(x0, lut), vt_decode<CB>(x1, lut));
}

__device__ __forceinline__ uint32_t vt_warp_min(uint32_t v)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __reduce_min_sync(0xffffffff, v);
#else
    #pragma unroll
    for (int o = 16; o; o >>= 1) v = min(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
#endif
}

// ---------------------------------------------------------------------------
// One trellis step: in-node width 16 - KIN, out-node width 16 - KOUT
// ---------------------------------------------------------------------------

// For every out node, the best of its 2^KOUT incoming windows. Window state s = (k << (16 - KOUT)) | out,
// its in-node s >> KIN. History stores k.
template <int KIN, int KOUT, int CB>
__device__ __forceinline__ void vt_step
(
    const half w,
    const half* __restrict__ costs_in,
    half* __restrict__ costs_out,
    uint8_t* __restrict__ hist_row,
    const bool first,
    const int pre_state,
    const bool write_hist,
    const half* __restrict__ lut
)
{
    constexpr int Kr = 16 - KOUT;
    constexpr int max_q = 1 << KOUT;
    constexpr int edges_out = 65536 >> KOUT;
    const half2 w2 = __half2half2(w);
    const half2 inf2 = __half2half2(VT_H_INF);

    constexpr uint32_t mult = vt_mult<CB>();
    constexpr uint32_t step = mult << Kr;

    for (int out = 2 * threadIdx.x; out < edges_out; out += 2 * VT_THREADS)
    {
        half2 min_err2 = inf2;
        int min_k0 = 0, min_k1 = 0;
        uint32_t p0 = mult ? (uint32_t) out * mult : (uint32_t) out;
        uint32_t p1 = mult ? p0 + mult : p0 + 1;

        #pragma unroll
        for (int k = 0; k < max_q; ++k)
        {
            const int s0 = (k << Kr) | out;
            const int in_edge = s0 >> KIN;  // s0 and s0 + 1 share their in-node
            if (k)
            {
                p0 += mult ? step : (1u << Kr);
                p1 += mult ? step : (1u << Kr);
            }
            const half2 dh2 = __hsub2(vt_decode2<CB>(p0, p1, lut), w2);
            half2 err2;
            if (first)
            {
                err2 = __hmul2(dh2, dh2);
                if (pre_state >= 0 && in_edge != pre_state) err2 = inf2;
            }
            else
                err2 = __hfma2(dh2, dh2, __half2half2(costs_in[in_edge]));
            if (k == 0)
            {
                min_err2 = err2;
                continue;
            }
            if (__hlt(__low2half(err2), __low2half(min_err2)))
            {
                min_err2 = __halves2half2(__low2half(err2), __high2half(min_err2));
                min_k0 = k;
            }
            if (__hlt(__high2half(err2), __high2half(min_err2)))
            {
                min_err2 = __halves2half2(__low2half(min_err2), __high2half(err2));
                min_k1 = k;
            }
        }

        reinterpret_cast<half2*>(costs_out)[out >> 1] = min_err2;
        if (write_hist)
        {
            hist_row[out] = (uint8_t) min_k0;
            hist_row[out + 1] = (uint8_t) min_k1;
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

// Costs live in shared memory when both arrays fit, else in the caller's scratch
__host__ __device__ constexpr bool vt_shared_costs(int KA) { return KA >= 2; }

template <int KA, int CB>
__global__ __launch_bounds__(VT_THREADS, 2)
void viterbi_kernel
(
    const float* __restrict__ input_ptr,
    float* __restrict__ output_ptr,
    uint16_t* __restrict__ states_ptr,
    half* __restrict__ costs_ptr,
    uint8_t* __restrict__ hist_ptr,
    const half* __restrict__ lut,
    const int L,
    const uint32_t mask
)
{
    constexpr int edges_max = 65536 >> KA;
    constexpr int NW = VT_THREADS / 32;
    extern __shared__ uint8_t shbuf[];

    const int tile = blockIdx.x;
    const int thread = threadIdx.x;
    const float* input = input_ptr + (size_t) L * tile;
    float* output = output_ptr + (size_t) L * tile;
    uint16_t* states = states_ptr + (size_t) L * tile;
    uint8_t* hist = hist_ptr + (size_t) L * edges_max * tile;

    uint8_t* sh = shbuf;
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);
    half* sh_input = (half*) sh; sh += ((L * sizeof(half) + 15) / 16) * 16;
    half* costs = vt_shared_costs(KA) ? (half*) sh : costs_ptr + (size_t) 2 * edges_max * tile;
    half* costs_inc = costs + edges_max;

    for (int i = thread; i < L; i += VT_THREADS) sh_input[i] = __float2half_rn(input[i]);
    __syncthreads();

    auto ring = [&](int i, int roll) { int r = i + roll; return r >= L ? r - L : r; };
    auto width = [&](int i) { return KA + (int) ((mask >> (i & 15)) & 1); };

    auto forward = [&](int roll, int pre_state)
    {
        for (int i = 0; i < L; ++i)
        {
            const int ri = ring(i, roll);
            half* t = costs; costs = costs_inc; costs_inc = t;
            const bool first = i == 0;
            // The first pass only traces back to position 0: its history past L / 2 is never read
            const bool write_hist = pre_state >= 0 || ri < L / 2;
            uint8_t* hist_row = hist + (size_t) edges_max * ri;
            const half w = sh_input[ri];
            const int din = width(ri) - KA, dout = width(ri + 1) - KA;
            if constexpr (KA == 8)
                vt_step<8, 8, CB>(w, costs_inc, costs, hist_row, first, pre_state, write_hist, lut);
            else
            {
                switch (din * 2 + dout)
                {
                    case 0: vt_step<KA, KA, CB>(w, costs_inc, costs, hist_row, first, pre_state, write_hist, lut); break;
                    case 1: vt_step<KA, KA + 1, CB>(w, costs_inc, costs, hist_row, first, pre_state, write_hist, lut); break;
                    case 2: vt_step<KA + 1, KA, CB>(w, costs_inc, costs, hist_row, first, pre_state, write_hist, lut); break;
                    default: vt_step<KA + 1, KA + 1, CB>(w, costs_inc, costs, hist_row, first, pre_state, write_hist, lut); break;
                }
            }
            __syncthreads();
        }
    };

    // Final costs over the out nodes of the last step (width set by D(0), as L / 2 and L are
    // multiples of the 16-position period). The rank fixes the tie order to the original's
    // 512-thread reduction: bit-reversed warp, bit-reversed lane, then 1024-state stripe
    auto argmin_cost = [&]()
    {
        const int edges_last = 65536 >> width(0);
        uint32_t best = 0x7c00ffffu;
        for (int e = thread; e < edges_last; e += VT_THREADS)
        {
            const unsigned v = e & 1023;
            const unsigned rank = ((__brev(v >> 5) >> 27) << 10) | ((__brev(v & 31) >> 27) << 5) | (e >> 10);
            best = min(best, ((uint32_t) __half_as_ushort(costs[e]) << 16) | rank);
        }
        best = vt_warp_min(best);
        if ((thread & 31) == 0) ((uint32_t*) sh_idx)[thread >> 5] = best;
        __syncthreads();
        if (thread < 32)
        {
            best = thread < NW ? ((uint32_t*) sh_idx)[thread] : 0x7c00ffffu;
            best = vt_warp_min(best);
        }
        __syncthreads();
        const unsigned rank = best & 65535;
        const unsigned v = ((__brev(rank >> 10) >> 27) << 5) | (__brev((rank >> 5) & 31) >> 27);
        // All costs inf/NaN (degenerate input): any valid node keeps the traceback in bounds
        return best >= 0x7c000000u ? 0 : (int) (((rank & 31) << 10) | v);
    };

    auto backward = [&](int roll, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                const int kout = width(ri + 1);
                const int k = hist[(size_t) edges_max * ri + edge];
                const int s = (k << (16 - kout)) | edge;
                edge = s >> width(ri);
                if (write)
                {
                    states[ri] = (uint16_t) s;
                    output[ri] = __half2float(vt_decode<CB>((uint32_t) s, lut));
                }
                else if (ri == 0) break;
            }
            sh_idx[0] = edge;
        }
        __syncthreads();
        const int e = sh_idx[0];
        __syncthreads();
        return e;
    };

    forward(L / 2, -1);
    const int start = backward(L / 2, false, argmin_cost());
    forward(0, start);
    backward(0, true, start);
}

template <int CB>
__global__ void decode_kernel(const uint16_t* __restrict__ states, float* __restrict__ out, int64_t n,
                              const half* __restrict__ lut)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(vt_decode<CB>((uint32_t) states[i], lut));
}

// Window i of a packed tile: the 16 stream bits ending at S(i) = sum of D(0..i), cyclically over the
// R stream bits; words are uint16, most significant bit first
__global__ void unpack_kernel(const uint16_t* __restrict__ packed, uint16_t* __restrict__ states,
                              int64_t n_tiles, int L, int words, int R, int ka, uint32_t mask)
{
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tiles * L) return;
    const int64_t tile = idx / L;
    const int i = (int) (idx - tile * L);
    const int period = 16 * ka + __popc(mask & 0xffffu);
    int end = (i >> 4) * period;
    for (int j = 0; j <= (i & 15); ++j) end += ka + ((mask >> j) & 1);
    const uint16_t* w = packed + tile * words;
    int start = end - 16;
    if (start < 0) start += R;
    uint32_t v;
    if (start + 16 <= R)
    {
        const int wi = start >> 4, o = start & 15;
        const uint32_t hi = w[wi], lo = (o && wi + 1 < words) ? w[wi + 1] : 0u;
        v = (((hi << 16) | lo) >> (16 - o)) & 0xffffu;
    }
    else
    {
        v = 0;
        for (int j = 0; j < 16; ++j)
        {
            int p = start + j;
            if (p >= R) p -= R;
            v = (v << 1) | ((w[p >> 4] >> (15 - (p & 15))) & 1u);
        }
    }
    states[idx] = (uint16_t) v;
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

using vt_kernel_t = void (*)(const float*, float*, uint16_t*, half*, uint8_t*, const half*, int, uint32_t);

template <int CB>
static vt_kernel_t vt_select(int ka)
{
    switch (ka)
    {
        case 1: return viterbi_kernel<1, CB>;
        case 2: return viterbi_kernel<2, CB>;
        case 3: return viterbi_kernel<3, CB>;
        case 4: return viterbi_kernel<4, CB>;
        case 5: return viterbi_kernel<5, CB>;
        case 6: return viterbi_kernel<6, CB>;
        case 7: return viterbi_kernel<7, CB>;
        default: return viterbi_kernel<8, CB>;
    }
}

static int64_t vt_shmem(int ka, int L)
{
    const int64_t base = 32 * sizeof(int) + ((L * sizeof(half) + 15) / 16) * 16;
    return base + (vt_shared_costs(ka) ? 2 * (int64_t) (65536 >> ka) * sizeof(half) : 0);
}

static const half* vt_lut(int cb, const c10::optional<torch::Tensor>& lut, const torch::Tensor& like)
{
    if (cb != 3) return nullptr;
    TORCH_CHECK(lut.has_value(), "trellis: the LUT codebook needs a table");
    const auto& t = lut.value();
    TORCH_CHECK(t.scalar_type() == at::kHalf && t.numel() == 65536 && t.is_contiguous(),
                "trellis: the LUT must be 65536 contiguous fp16 values");
    TORCH_CHECK(t.device() == like.device(), "trellis: the LUT must be on the input's device");
    return (const half*) t.data_ptr();
}

// Tiles per launch that the scratch holds and the device keeps resident
int64_t viterbi_max_batch(int64_t device, int64_t ka, int64_t L)
{
    const c10::cuda::CUDAGuard guard((c10::DeviceIndex) device);
    TORCH_CHECK(ka >= 1 && ka <= 8, "trellis: KA must be 1..8");
    auto kernel = vt_select<2>((int) ka);
    const int64_t shmem = vt_shmem((int) ka, (int) L);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem));
    int blocks = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, VT_THREADS, shmem));
    const auto* props = at::cuda::getDeviceProperties((int) device);
    return std::max<int64_t>(1, (int64_t) blocks * props->multiProcessorCount);
}

void viterbi
(
    torch::Tensor input,      // [n, L] float32
    torch::Tensor output,     // [n, L] float32
    torch::Tensor states,     // [n, L] int16
    torch::Tensor costs,      // [b, 2, 65536 >> KA] float16 (read only when KA == 1)
    torch::Tensor hist,       // [b, L, 65536 >> KA] uint8
    int64_t ka,
    int64_t mask,
    int64_t cb,
    c10::optional<torch::Tensor> lut
)
{
    const at::cuda::OptionalCUDAGuard guard(input.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(ka >= 1 && ka <= 8, "trellis: KA must be 1..8");
    TORCH_CHECK(mask >= 0 && mask <= 0xffff, "trellis: MASK is a 16-bit pattern");
    TORCH_CHECK(mask == 0 || ka <= 7, "trellis: KA + 1 must not exceed 8");
    TORCH_CHECK(cb >= 0 && cb <= 3, "trellis: codebook must be 0..3");
    TORCH_CHECK(input.dim() == 2, "trellis: input must be [n, L]");
    const int64_t n = input.size(0);
    const int L = (int) input.size(1);
    TORCH_CHECK(L >= 32 && L % 32 == 0 && L <= 4096, "trellis: L must be a multiple of 32 in 32..4096");
    const int edges_max = 65536 >> ka;
    TORCH_CHECK(input.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat, "trellis: float32 tiles");
    TORCH_CHECK(states.scalar_type() == at::kShort, "trellis: states are int16");
    TORCH_CHECK(costs.scalar_type() == at::kHalf && hist.scalar_type() == at::kByte, "trellis: scratch dtypes");
    TORCH_CHECK(output.sizes() == input.sizes() && states.sizes() == input.sizes(), "trellis: shapes");
    TORCH_CHECK(hist.dim() == 3 && hist.size(1) == L && hist.size(2) == edges_max, "trellis: history shape");
    TORCH_CHECK(costs.dim() == 3 && costs.size(1) == 2 && costs.size(2) == edges_max, "trellis: cost scratch shape");
    for (const auto& t : {input, output, states, costs, hist})
    {
        TORCH_CHECK(t.is_contiguous(), "trellis: tensors must be contiguous");
        TORCH_CHECK(t.device() == input.device(), "trellis: tensors must share a device");
    }
    if (n == 0) return;
    const half* lut_ptr = vt_lut((int) cb, lut, input);

    vt_kernel_t kernel;
    switch (cb)
    {
        case 0: kernel = vt_select<0>((int) ka); break;
        case 1: kernel = vt_select<1>((int) ka); break;
        case 2: kernel = vt_select<2>((int) ka); break;
        default: kernel = vt_select<3>((int) ka); break;
    }
    const int64_t shmem = vt_shmem((int) ka, L);
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem));
    int64_t batch = hist.size(0);
    if (!vt_shared_costs((int) ka)) batch = std::min(batch, costs.size(0));
    TORCH_CHECK(batch > 0, "trellis: scratch must hold at least one tile");

    for (int64_t b0 = 0; b0 < n; b0 += batch)
    {
        const int64_t bsz = std::min(batch, n - b0);
        kernel<<<(unsigned) bsz, VT_THREADS, shmem, stream>>>(
            input.data_ptr<float>() + (size_t) L * b0,
            output.data_ptr<float>() + (size_t) L * b0,
            (uint16_t*) states.data_ptr() + (size_t) L * b0,
            (half*) costs.data_ptr(),
            hist.data_ptr<uint8_t>(),
            lut_ptr, L, (uint32_t) mask);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void decode(torch::Tensor states, torch::Tensor output, int64_t cb, c10::optional<torch::Tensor> lut)
{
    const at::cuda::OptionalCUDAGuard guard(states.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(states.scalar_type() == at::kShort && output.scalar_type() == at::kFloat, "trellis decode: dtypes");
    TORCH_CHECK(states.numel() == output.numel() && states.is_contiguous() && output.is_contiguous(),
                "trellis decode: shapes");
    TORCH_CHECK(cb >= 0 && cb <= 3, "trellis decode: codebook must be 0..3");
    const int64_t n = states.numel();
    if (n == 0) return;
    const half* lut_ptr = vt_lut((int) cb, lut, states);
    const auto* s = (const uint16_t*) states.data_ptr();
    float* o = output.data_ptr<float>();
    const unsigned grid = (unsigned) ((n + 255) / 256);
    switch (cb)
    {
        case 0: decode_kernel<0><<<grid, 256, 0, stream>>>(s, o, n, lut_ptr); break;
        case 1: decode_kernel<1><<<grid, 256, 0, stream>>>(s, o, n, lut_ptr); break;
        case 2: decode_kernel<2><<<grid, 256, 0, stream>>>(s, o, n, lut_ptr); break;
        default: decode_kernel<3><<<grid, 256, 0, stream>>>(s, o, n, lut_ptr); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unpack(torch::Tensor packed, torch::Tensor states, int64_t ka, int64_t mask)
{
    const at::cuda::OptionalCUDAGuard guard(packed.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(packed.scalar_type() == at::kShort && states.scalar_type() == at::kShort, "trellis unpack: int16");
    TORCH_CHECK(packed.dim() == 2 && states.dim() == 2 && packed.size(0) == states.size(0), "trellis unpack: shapes");
    TORCH_CHECK(packed.is_contiguous() && states.is_contiguous(), "trellis unpack: contiguous");
    const int L = (int) states.size(1);
    TORCH_CHECK(L % 16 == 0, "trellis unpack: L must be a multiple of 16");
    int extra = 0;
    for (int64_t m = mask & 0xffff; m; m &= m - 1) extra++;
    const int R = (L / 16) * (16 * (int) ka + extra);
    TORCH_CHECK(packed.size(1) == (R + 15) / 16, "trellis unpack: word count");
    const int64_t total = states.numel();
    if (total == 0) return;
    unpack_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, stream>>>(
        (const uint16_t*) packed.data_ptr(), (uint16_t*) states.data_ptr(), packed.size(0), L,
        (int) packed.size(1), R, (int) ka, (uint32_t) mask);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("unpack", &unpack, "Unpack trellis tiles to their 16-bit window states");
    m.def("viterbi", &viterbi, "Tail-biting Viterbi encoding of [n, L] tiles to a bitshift trellis code");
    m.def("viterbi_max_batch", &viterbi_max_batch, "Resident tiles per launch for (device, KA, L)");
    m.def("decode", &decode, "Decode 16-bit trellis states through a codebook");
}
