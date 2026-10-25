// Warm-up: one warp sums and maxes one row of 128 floats with shuffles.
// This is phase 2 of the decode kernel on its own.

#include <cmath>
#include <torch/extension.h>

// One block of 32 threads (one warp) per row. Lane l owns columns 4l .. 4l+3.
__global__ void warp_sum_max_kernel(const float *x, float *row_sum,
                                    float *row_max) {
  int row = blockIdx.x;
  int lane = threadIdx.x; // 0 .. 31
  const float *mine = x + row * 128 + 4 * lane;

  // Step 1. Reduce your own 4 numbers inside this lane.
  float s = mine[0] + mine[1] + mine[2] + mine[3];
  float m = fmaxf(mine[0], fmaxf(mine[1], fmaxf(mine[2], mine[3])));

  // Step 2. Five shuffle rounds with xor 16, 8, 4, 2, 1. After them, every
  // lane holds the whole row's sum and max.
  int shuffleArr[] = {16, 8, 4, 2, 1};
  for (int i = 0; i < 5; i++) {
    int dist = shuffleArr[i];
    float distSum = __shfl_xor_sync(0xffffffff, s, dist);
    float distMax = __shfl_xor_sync(0xffffffff, m, dist);
    s = s + distSum;
    m = fmaxf(m, distMax);
  }

  // Step 3. Every lane has the answer now, so one lane writes it.
  if (lane == 0) {
    row_sum[row] = s;
    row_max[row] = m;
  }
}

std::tuple<torch::Tensor, torch::Tensor> warp_sum_max(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32 &&
                  x.is_contiguous(),
              "x must be contiguous fp32 on the GPU");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 128, "x must be [N, 128]");
  int n = x.size(0);
  auto row_sum = torch::empty({n}, x.options());
  auto row_max = torch::empty({n}, x.options());
  if (n > 0) {
    warp_sum_max_kernel<<<n, 32>>>(x.data_ptr<float>(),
                                   row_sum.data_ptr<float>(),
                                   row_max.data_ptr<float>());
  }
  return {row_sum, row_max};
}
