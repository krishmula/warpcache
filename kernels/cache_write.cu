// Write new tokens' K and V into a paged FP16 or INT8 KV cache. One warp per
// (token, KV head) row of 128 values. Each lane owns 4 of the 128 columns, as
// in decode.cu. It must match PagedKVCache.write in reference/paged.py exactly.

#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cstdlib>
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
                                   const int64_t *slots, T *k_cache, T *v_cache,
                                   float *k_scale, float *v_scale) {
  int b = blockIdx.x;
  int kv_head = blockIdx.y;
  int lane = threadIdx.x;
  int64_t slot = slots[b];

  // we have to basically take these, and figure out where the 4 values that
  // this thread is supposed to operate on start from.
  T *k_mem = k_cache + (slot * 1024) + (kv_head * 128) + (lane * 4);
  T *v_mem = v_cache + (slot * 1024) + (kv_head * 128) + (lane * 4);

  // we also have to know which of the 1024 values of the K/V we need to pick
  const __half *k_put = k + (b * 1024) + (kv_head * 128) + (lane * 4);
  const __half *v_put = v + (b * 1024) + (kv_head * 128) + (lane * 4);

  float *k_scale_ptr = k_scale + (slot * 8) + kv_head;
  float *v_scale_ptr = v_scale + (slot * 8) + kv_head;

  float maxValK = 0.0f;
  float maxValV = 0.0f;
  float k_scale_val = 0.0f;
  float v_scale_val = 0.0f;

  // so, each thread handles 4 of the 1024 values of a K/V. This is gotten by
  // k_mem, and the 3 after it, and the same for v_mem, and the 3 after it
  for (int i = 0; i < 4; i++) {
    T *k_add = k_mem + i;
    T *v_add = v_mem + i;

    // now, we have to figure out which of the 1024 values of k/v we're going to
    // put in these 4 places.
    const __half *k_val = k_put + i;
    const __half *v_val = v_put + i;

    if constexpr (std::is_same_v<T, __half>) {
      *k_add = *k_val;
      *v_add = *v_val;
    }
  }

  for (int i = 0; i < 4; i++) {
    const __half *k_val = k_put + i;
    const __half *v_val = v_put + i;
    if constexpr (std::is_same_v<T, int8_t>) {
      maxValK = fmaxf(maxValK, fabsf(__half2float(*k_val)));
      maxValV = fmaxf(maxValV, fabsf(__half2float(*v_val)));
    }
  }

  if constexpr (std::is_same_v<T, int8_t>) {
    int shuffles[] = {16, 8, 4, 2, 1};
    for (int i = 0; i < 5; i++) {
      int shuffleDist = shuffles[i];
      float shuffleMaxK = __shfl_xor_sync(0xffffffff, maxValK, shuffleDist);
      float shuffleMaxV = __shfl_xor_sync(0xffffffff, maxValV, shuffleDist);

      maxValK = fmaxf(maxValK, shuffleMaxK);
      maxValV = fmaxf(maxValV, shuffleMaxV);
    }

    if (maxValK == 0)
      k_scale_val = 1.0f;
    else
      k_scale_val = maxValK / 127;

    if (maxValV == 0)
      v_scale_val = 1.0f;
    else
      v_scale_val = maxValV / 127;

    for (int i = 0; i < 4; i++) {
      const __half *k_val = k_put + i;
      const __half *v_val = v_put + i;
      T *k_add = k_mem + i;
      T *v_add = v_mem + i;

      int k_q = __float2int_rn(__half2float(*k_val) / k_scale_val);
      int v_q = __float2int_rn(__half2float(*v_val) / v_scale_val);

      k_q = min(max(k_q, -127), 127);
      v_q = min(max(v_q, -127), 127);

      *k_add = k_q;
      *v_add = v_q;
    }

    if (lane == 0) {
      *k_scale_ptr = k_scale_val;
      *v_scale_ptr = v_scale_val;
    }
  }
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
    cache_write_kernel<int8_t>
        <<<grid, 32>>>(k_ptr, v_ptr, slots_ptr, k_cache.data_ptr<int8_t>(),
                       v_cache.data_ptr<int8_t>(), k_scale->data_ptr<float>(),
                       v_scale->data_ptr<float>());
  } else {
    cache_write_kernel<__half>
        <<<grid, 32>>>(k_ptr, v_ptr, slots_ptr,
                       reinterpret_cast<__half *>(k_cache.data_ptr<at::Half>()),
                       reinterpret_cast<__half *>(v_cache.data_ptr<at::Half>()),
                       nullptr, nullptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
