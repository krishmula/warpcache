// Decode attention over a paged FP16 KV cache. Path A: one chunk per head.
// Version 1 runs one warp per (request, query head). Each lane owns 4 of the
// 128 columns, as in warp_reduce.cu.

#include <cuda_fp16.h>
#include <torch/extension.h>

constexpr int HQ = 16;  // query heads
constexpr int HKV = 8;  // KV heads. Query head h reads KV head h / (HQ / HKV).
constexpr int D = 128;  // head dim
constexpr int PAGE = 16; // tokens per page

// q, out:              [B, HQ, D] fp16
// k_cache, v_cache:    [P, PAGE, HKV, D] fp16
// block_table:         [B, max_pages] int32, logical page -> physical page
// seq_lens:            [B] int32, at least 1
// Grid (HQ, B): blockIdx.x is the query head, blockIdx.y is the request.
__global__ void decode_kernel(const __half *q, const __half *k_cache,
                              const __half *v_cache, const int *block_table,
                              const int *seq_lens, int max_pages,
                              __half *out) {
  // Step 0. Who am I: request b, query head h, lane, KV head, history length.

  // Step 1. Load your 4 numbers of q as floats. __half2float(x) converts.

  // Step 2. Start the running state: m = -INFINITY, l = 0, u[4] = 0.

  // Step 3. For each token t in [0, seq_len):
  //   a. Find token t's K row through the block table, then your 4 numbers.
  //   b. Partial dot product of your 4 q and 4 K numbers, then five
  //      __shfl_xor_sync rounds give every lane the full score. Divide by
  //      sqrt(128).
  //   c. Online softmax update of m, l, and your 4 u, using token t's V row.

  // Step 4. Write your 4 outputs, u / l, as fp16. __float2half(x) converts.
}

torch::Tensor decode_fp16(torch::Tensor q, torch::Tensor k_cache,
                          torch::Tensor v_cache, torch::Tensor block_table,
                          torch::Tensor seq_lens) {
  for (auto &t : {q, k_cache, v_cache, block_table, seq_lens}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(),
                "all inputs must be contiguous on the GPU");
  }
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == HQ && q.size(2) == D,
              "q must be fp16 [B, 16, 128]");
  for (auto &c : {k_cache, v_cache}) {
    TORCH_CHECK(c.scalar_type() == torch::kHalf && c.dim() == 4 &&
                    c.size(1) == PAGE && c.size(2) == HKV && c.size(3) == D,
                "k_cache and v_cache must be fp16 [P, 16, 8, 128]");
  }
  int batch = q.size(0);
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  block_table.dim() == 2 && block_table.size(0) == batch,
              "block_table must be int32 [B, max_pages]");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 &&
                  seq_lens.dim() == 1 && seq_lens.size(0) == batch,
              "seq_lens must be int32 [B]");

  auto out = torch::empty_like(q);
  if (batch > 0) {
    decode_kernel<<<dim3(HQ, batch), 32>>>(
        reinterpret_cast<const __half *>(q.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_cache.data_ptr<at::Half>()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        block_table.size(1),
        reinterpret_cast<__half *>(out.data_ptr<at::Half>()));
  }
  return out;
}
