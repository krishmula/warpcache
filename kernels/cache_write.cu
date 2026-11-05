// Write new tokens' K and V into a paged FP16 or INT8 KV cache. One warp per
// (token, KV head) row of 128 values. Each lane owns 4 of the 128 columns, as
// in decode.cu. It must match PagedKVCache.write in reference/paged.py exactly.

#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <type_traits>

constexpr int HKV = 8;   // KV heads
constexpr int D = 128;   // head dim
constexpr int PAGE = 16; // slots per page

// k, v:                [N, HKV, D] fp16, the new tokens
// slots:               [N] int64, slot number = page * PAGE + page slot
// k_cache, v_cache:    [P, PAGE, HKV, D] fp16 or int8
// k_scale, v_scale:    [P, PAGE, HKV] fp32 for int8, nullptr for fp16
// Grid (N, HKV): blockIdx.x is the token, blockIdx.y is the KV head.

template <typename T>
__global__ void cache_write_kernel(const __half *k, const __half *v,
                                   const int64_t *slots, T *k_cache,
                                   T *v_cache, float *k_scale, float *v_scale) {
}

void cache_write(torch::Tensor k, torch::Tensor v, torch::Tensor slots,
                 torch::Tensor k_cache, torch::Tensor v_cache,
                 std::optional<torch::Tensor> k_scale,
                 std::optional<torch::Tensor> v_scale) {
  for (auto &t : {k, v, slots, k_cache, v_cache}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(),
                "all inputs must be contiguous on the GPU");
  }
  int n = k.size(0);
  for (auto &x : {k, v}) {
    TORCH_CHECK(x.scalar_type() == torch::kHalf && x.dim() == 3 &&
                    x.size(0) == n && x.size(1) == HKV && x.size(2) == D,
                "k and v must be fp16 [N, 8, 128] with the same N");
  }
  TORCH_CHECK(slots.scalar_type() == torch::kInt64 && slots.dim() == 1 &&
                  slots.size(0) == n,
              "slots must be int64 [N]");
  auto cache_type = k_cache.scalar_type();
  TORCH_CHECK(cache_type == torch::kHalf || cache_type == torch::kInt8,
              "k_cache must be fp16 or int8");
  for (auto &c : {k_cache, v_cache}) {
    TORCH_CHECK(c.scalar_type() == cache_type && c.dim() == 4 &&
                    c.size(0) == k_cache.size(0) && c.size(1) == PAGE &&
                    c.size(2) == HKV && c.size(3) == D,
                "k_cache and v_cache must be [P, 16, 8, 128] of one dtype");
  }
  bool int8 = cache_type == torch::kInt8;
  TORCH_CHECK(
      k_scale.has_value() == int8 && v_scale.has_value() == int8,
      "k_scale and v_scale are required for int8 and must be None for fp16");
  if (int8) {
    for (auto &s : {*k_scale, *v_scale}) {
      TORCH_CHECK(
          s.is_cuda() && s.is_contiguous() &&
              s.scalar_type() == torch::kFloat32 && s.dim() == 3 &&
              s.size(0) == k_cache.size(0) && s.size(1) == PAGE &&
              s.size(2) == HKV,
          "k_scale and v_scale must be contiguous fp32 [P, 16, 8] on the GPU");
    }
  }
  if (n == 0) {
    return;
  }

  dim3 grid(n, HKV);
  auto k_ptr = reinterpret_cast<const __half *>(k.data_ptr<at::Half>());
  auto v_ptr = reinterpret_cast<const __half *>(v.data_ptr<at::Half>());
  auto slots_ptr = slots.data_ptr<int64_t>();
  if (int8) {
    cache_write_kernel<int8_t><<<grid, 32>>>(
        k_ptr, v_ptr, slots_ptr, k_cache.data_ptr<int8_t>(),
        v_cache.data_ptr<int8_t>(), k_scale->data_ptr<float>(),
        v_scale->data_ptr<float>());
  } else {
    cache_write_kernel<__half><<<grid, 32>>>(
        k_ptr, v_ptr, slots_ptr,
        reinterpret_cast<__half *>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(v_cache.data_ptr<at::Half>()), nullptr,
        nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
