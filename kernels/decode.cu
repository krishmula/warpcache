// Decode attention over a paged FP16 or INT8 KV cache. Version 1 runs one warp
// per (request, query head, chunk). Each lane owns 4 of the 128 columns, as in
// warp_reduce.cu. Path A (1 chunk) writes the output. Path B (more than 1
// chunk) writes each chunk's partial state, and PyTorch merges them.

#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <type_traits>

constexpr int HQ = 16;   // query heads
constexpr int HKV = 8;   // KV heads. Query head h reads KV head h / (HQ / HKV).
constexpr int D = 128;   // head dim
constexpr int PAGE = 16; // tokens per page

// q, out:              [B, HQ, D] fp16
// k_cache, v_cache:    [P, PAGE, HKV, D] fp16 or int8
// k_scale, v_scale:    [P, PAGE, HKV] fp32 for int8, nullptr for fp16
// block_table:         [B, max_pages] int32, logical page -> physical page
// seq_lens:            [B] int32, at least 1
// m_ptr, l_ptr:        [B, HQ, C] fp32 for Path B, nullptr for Path A
// u_ptr:               [B, HQ, C, D] fp32 for Path B, nullptr for Path A
// Grid (HQ, B, C): blockIdx.x is the query head, blockIdx.y is the request,
// blockIdx.z is the chunk.

__device__ float flt_convert(__half x) { return __half2float(x); }

__device__ float flt_convert(int8_t x) { return float(x); }

template <typename T>
__global__ void decode_kernel(const __half *q, const T *k_cache,
                              const T *v_cache, const float *k_scale,
                              const float *v_scale, const int *block_table,
                              const int *seq_lens, int max_pages, __half *out,
                              float *m_ptr, float *l_ptr, float *u_ptr) {
  // Step 0. Who am I: request b, query head h, lane, KV head, history length.
  int b = blockIdx.y;
  int q_head = blockIdx.x;
  int lane = threadIdx.x;
  int hist_len = seq_lens[b];
  int kv_head = q_head / (HQ / HKV);
  int chunk_size = (hist_len + gridDim.z - 1) / gridDim.z;

  // This block's chunk of the history. The last chunks can be short or empty.
  int start = chunk_size * blockIdx.z;
  int end = start + chunk_size;

  // Step 1. Load your 4 numbers of q as floats. __half2float(x) converts.
  const __half *mine = q + 128 * (b * HQ + q_head) + 4 * lane;
  __half *mine_out = out + 128 * (b * HQ + q_head) + 4 * lane;

  float q_f[4];
  for (int i = 0; i < 4; i++) {
    q_f[i] = __half2float(mine[i]);
  }

  // Step 2. Start the running state: m = -INFINITY, l = 0, u[4] = 0.
  float m = -INFINITY;
  float l = 0.0f;
  float u[4] = {0};

  int ws_slot = b * (HQ * gridDim.z) + q_head * gridDim.z + blockIdx.z;
  float *mine_m = m_ptr + ws_slot;
  float *mine_l = l_ptr + ws_slot;
  float *mine_u = u_ptr + ws_slot * D + 4 * lane;

  // Step 3. For each token t in [start, end):
  //   a. Find token t's K row through the block table, then your 4 numbers.
  //   b. Partial dot product of your 4 q and 4 K numbers, then five
  //      __shfl_xor_sync rounds give every lane the full score. Divide by
  //      sqrt(128).
  //   c. Online softmax update of m, l, and your 4 u, using token t's V row.

  for (int t = start; t < end; t++) {
    if (t >= hist_len)
      break;
    int physical_page = block_table[b * max_pages + t / PAGE];
    int slot = t % PAGE;
    const T *mine_k = k_cache + physical_page * (PAGE * HKV * D) +
                      slot * (HKV * D) + kv_head * D + 4 * lane;
    const T *mine_v = v_cache + physical_page * (PAGE * HKV * D) +
                      slot * (HKV * D) + kv_head * D + 4 * lane;

    // K/V scales are stored as [P, 16, 8], so, that means we have to know what
    // page we want to access, and then we have to know which of the 16 slots
    // the token is in, and then, we have to know what head we want to access.

    float ks = 1.0f;
    float vs = 1.0f;
    if constexpr (std::is_same_v<T, int8_t>) {
      const float *k_s = k_scale + physical_page * (PAGE * HKV) + slot * HKV;
      const float *v_s = v_scale + physical_page * (PAGE * HKV) + slot * HKV;
      ks = k_s[kv_head];
      vs = v_s[kv_head];
    }

    float k_f[4];
    float v_f[4];
    for (int i = 0; i < 4; i++) {
      k_f[i] = flt_convert(mine_k[i]) * ks;
      v_f[i] = flt_convert(mine_v[i]) * vs;
    }

    float partial = 0.0f;
    for (int i = 0; i < 4; i++) {
      partial += q_f[i] * k_f[i];
    }

    int shuffles[] = {16, 8, 4, 2, 1};
    for (int i = 0; i < 5; i++) {
      int shuffleDist = shuffles[i];
      float shufflePartial = __shfl_xor_sync(0xffffffff, partial, shuffleDist);
      partial += shufflePartial;
    }

    float z = partial / sqrtf(float(D));
    float new_m = fmaxf(m, z);
    float shrink = expf(m - new_m);
    float w = expf(z - new_m);
    float new_l = l * shrink + w;
    l = new_l;

    for (int i = 0; i < 4; i++) {
      u[i] = u[i] * shrink + w * v_f[i];
    }

    m = new_m;
  }

  // Step 4. Path A writes u / l as fp16. Path B writes the partial state
  // without the divide, so an empty chunk (l = 0) never divides by zero.
  if (gridDim.z == 1) {
    for (int i = 0; i < 4; i++) {
      mine_out[i] = __float2half(u[i] / l);
    }
  } else {
    if (lane == 0) {
      mine_m[0] = m;
      mine_l[0] = l;
    }
    for (int i = 0; i < 4; i++) {
      mine_u[i] = u[i];
    }
  }
}

std::vector<torch::Tensor> decode(torch::Tensor q, torch::Tensor k_cache,
                                  torch::Tensor v_cache,
                                  std::optional<torch::Tensor> k_scale,
                                  std::optional<torch::Tensor> v_scale,
                                  torch::Tensor block_table,
                                  torch::Tensor seq_lens, int num_chunks) {
  for (auto &t : {q, k_cache, v_cache, block_table, seq_lens}) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous(),
                "all inputs must be contiguous on the GPU");
  }
  TORCH_CHECK(q.scalar_type() == torch::kHalf && q.dim() == 3 &&
                  q.size(1) == HQ && q.size(2) == D,
              "q must be fp16 [B, 16, 128]");
  auto cache_type = k_cache.scalar_type();
  TORCH_CHECK(cache_type == torch::kHalf || cache_type == torch::kInt8,
              "k_cache must be fp16 or int8");
  for (auto &c : {k_cache, v_cache}) {
    TORCH_CHECK(c.scalar_type() == cache_type && c.dim() == 4 &&
                    c.size(1) == PAGE && c.size(2) == HKV && c.size(3) == D,
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
  int batch = q.size(0);
  TORCH_CHECK(block_table.scalar_type() == torch::kInt32 &&
                  block_table.dim() == 2 && block_table.size(0) == batch,
              "block_table must be int32 [B, max_pages]");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32 && seq_lens.dim() == 1 &&
                  seq_lens.size(0) == batch,
              "seq_lens must be int32 [B]");
  TORCH_CHECK(num_chunks > 0, "num_chunks should be greater than 0");

  auto out = torch::empty_like(q);
  if (batch == 0) {
    return {out};
  }
  torch::Tensor m_ws, l_ws, u_ws;
  if (num_chunks > 1) {
    auto f32 = q.options().dtype(torch::kFloat32);
    m_ws = torch::empty({batch, HQ, num_chunks}, f32);
    l_ws = torch::empty({batch, HQ, num_chunks}, f32);
    u_ws = torch::empty({batch, HQ, num_chunks, D}, f32);
  }
  float *m_ptr = m_ws.defined() ? m_ws.data_ptr<float>() : nullptr;
  float *l_ptr = l_ws.defined() ? l_ws.data_ptr<float>() : nullptr;
  float *u_ptr = u_ws.defined() ? u_ws.data_ptr<float>() : nullptr;

  dim3 grid(HQ, batch, num_chunks);
  auto q_ptr = reinterpret_cast<const __half *>(q.data_ptr<at::Half>());
  auto out_ptr = reinterpret_cast<__half *>(out.data_ptr<at::Half>());
  if (int8) {
    decode_kernel<int8_t><<<grid, 32>>>(
        q_ptr, k_cache.data_ptr<int8_t>(), v_cache.data_ptr<int8_t>(),
        k_scale->data_ptr<float>(), v_scale->data_ptr<float>(),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        block_table.size(1), out_ptr, m_ptr, l_ptr, u_ptr);
  } else {
    decode_kernel<__half><<<grid, 32>>>(
        q_ptr, reinterpret_cast<const __half *>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const __half *>(v_cache.data_ptr<at::Half>()), nullptr,
        nullptr, block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        block_table.size(1), out_ptr, m_ptr, l_ptr, u_ptr);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (num_chunks == 1)
    return {out};
  else
    return {m_ws, l_ws, u_ws};
}
