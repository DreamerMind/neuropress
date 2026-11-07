/*
 * vq_dequant.cu — one-pass VQ weight decode: packed indices -> dense weight.
 *
 *   W.flat[v * VL + j] = centroids[idx(v), j]     v in [0, n_vectors), j in [0, VL)
 *
 * idx(v) is read from the LSB-first bit-stream used by IndexPacking
 * (vq_packing.py), 4 <= bits <= 16.  Large-batch NPVQLinear forwards decode
 * the weight with this kernel then hand it to cuBLAS.
 *
 * Why a dedicated kernel: the Python path (unpack to int32, cast to int64,
 * gather fp16 rows, cast to the activation dtype) writes ~3.5x the final
 * weight in intermediates and is bandwidth-bound on them.  Here each thread
 * reads one index and writes VL output values once: traffic is the packed
 * buffer plus the output, and the 32-128 KiB codebook stays in L2.
 *
 * No padding is required on the packed buffer: the 3-byte window read is
 * bounds-checked against n_bytes.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

__device__ __forceinline__ uint32_t read_index(
    const uint8_t* __restrict__ packed, int64_t n_bytes, int64_t v, int bits) {
  const int64_t bit_start = v * (int64_t)bits;
  const int64_t b0 = bit_start >> 3;
  const int bit_off = (int)(bit_start & 7);
  uint32_t window = (uint32_t)__ldg(&packed[b0]);
  if (b0 + 1 < n_bytes) window |= (uint32_t)__ldg(&packed[b0 + 1]) << 8;
  if (b0 + 2 < n_bytes) window |= (uint32_t)__ldg(&packed[b0 + 2]) << 16;
  return (window >> bit_off) & ((1u << bits) - 1u);
}

template <typename T>
__device__ __forceinline__ T from_half(__half h);
template <>
__device__ __forceinline__ __half from_half<__half>(__half h) { return h; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_half<__nv_bfloat16>(__half h) {
  return __float2bfloat16(__half2float(h));
}
template <>
__device__ __forceinline__ float from_half<float>(__half h) { return __half2float(h); }

// VL=4 fast path: one 8-byte codebook load per vector.
template <typename T>
__global__ void _vq_dequant_vl4(
    const uint8_t* __restrict__ packed, int64_t n_bytes,
    const __half* __restrict__ centroids, T* __restrict__ out,
    int64_t n_vectors, int bits) {
  const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n_vectors) return;
  const uint32_t idx = read_index(packed, n_bytes, v, bits);
  const uint2 raw = __ldg(reinterpret_cast<const uint2*>(centroids) + idx);
  const __half* c = reinterpret_cast<const __half*>(&raw);
  // Assemble the 4 outputs in registers and write them with one vector store
  // (8 bytes for 16-bit T, 16 bytes for float): out is allocated by us, so
  // row v*4 is always 4*sizeof(T)-aligned.
  struct alignas(4 * sizeof(T)) Vec4 { T x[4]; };
  Vec4 o;
  o.x[0] = from_half<T>(c[0]);
  o.x[1] = from_half<T>(c[1]);
  o.x[2] = from_half<T>(c[2]);
  o.x[3] = from_half<T>(c[3]);
  reinterpret_cast<Vec4*>(out)[v] = o;
}

template <typename T>
__global__ void _vq_dequant_generic(
    const uint8_t* __restrict__ packed, int64_t n_bytes,
    const __half* __restrict__ centroids, T* __restrict__ out,
    int64_t n_vectors, int bits, int vl) {
  const int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n_vectors) return;
  const uint32_t idx = read_index(packed, n_bytes, v, bits);
  const __half* c = centroids + (int64_t)idx * vl;
  T* dst = out + v * vl;
  for (int j = 0; j < vl; ++j) dst[j] = from_half<T>(__ldg(&c[j]));
}

template <typename T>
static void launch(const torch::Tensor& packed, const torch::Tensor& centroids,
                   torch::Tensor& out, int64_t n_vectors, int bits, int vl) {
  const int threads = 256;
  const int64_t blocks = (n_vectors + threads - 1) / threads;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* p = packed.data_ptr<uint8_t>();
  const auto* c = reinterpret_cast<const __half*>(centroids.data_ptr<at::Half>());
  T* o = reinterpret_cast<T*>(out.data_ptr());
  // The VL=4 path loads each codebook row as one 8-byte word.
  if (vl == 4 && reinterpret_cast<uintptr_t>(c) % 8 == 0) {
    _vq_dequant_vl4<T><<<blocks, threads, 0, stream>>>(
        p, packed.numel(), c, o, n_vectors, bits);
  } else {
    _vq_dequant_generic<T><<<blocks, threads, 0, stream>>>(
        p, packed.numel(), c, o, n_vectors, bits, vl);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor vq_dequant(
    torch::Tensor packed,     // [n_bytes] uint8 CUDA bit-stream
    torch::Tensor centroids,  // [NC, VL] fp16 CUDA
    int64_t bits,
    int64_t out_features,
    int64_t in_features,
    at::ScalarType dtype) {
  TORCH_CHECK(packed.is_cuda() && packed.is_contiguous() &&
                  packed.scalar_type() == torch::kUInt8,
              "vq_dequant: packed must be a contiguous uint8 CUDA tensor.");
  TORCH_CHECK(centroids.is_cuda() && centroids.is_contiguous() &&
                  centroids.scalar_type() == torch::kHalf && centroids.dim() == 2,
              "vq_dequant: centroids must be a contiguous [NC, VL] fp16 CUDA tensor.");
  TORCH_CHECK(packed.device() == centroids.device(),
              "vq_dequant: packed and centroids must be on the same device.");
  TORCH_CHECK(bits >= 4 && bits <= 16, "vq_dequant: bits must be in [4, 16], got ", bits);
  const int vl = (int)centroids.size(1);
  const int64_t numel = out_features * in_features;
  TORCH_CHECK(numel % vl == 0, "vq_dequant: weight size not divisible by VL=", vl);
  const int64_t n_vectors = numel / vl;
  TORCH_CHECK((n_vectors * bits + 7) / 8 <= packed.numel(),
              "vq_dequant: packed buffer too short for ", n_vectors, " indices of ",
              bits, " bits.");

  auto out = torch::empty({out_features, in_features},
                          packed.options().dtype(dtype));
  if (n_vectors == 0) return out;
  switch (dtype) {
    case torch::kBFloat16:
      launch<__nv_bfloat16>(packed, centroids, out, n_vectors, (int)bits, vl);
      break;
    case torch::kHalf:
      launch<__half>(packed, centroids, out, n_vectors, (int)bits, vl);
      break;
    case torch::kFloat:
      launch<float>(packed, centroids, out, n_vectors, (int)bits, vl);
      break;
    default:
      TORCH_CHECK(false, "vq_dequant: dtype must be bfloat16, float16 or float32.");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vq_dequant", &vq_dequant,
        "Decode a packed VQ weight to a dense [out, in] tensor in one pass.",
        py::arg("packed"), py::arg("centroids"), py::arg("bits"),
        py::arg("out_features"), py::arg("in_features"), py::arg("dtype"));
}
