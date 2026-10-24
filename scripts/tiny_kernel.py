"""Compile and run a trivial custom CUDA kernel through a PyTorch extension."""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
__global__ void scale_kernel(const float* x, float* y, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i];
}

torch::Tensor scale(torch::Tensor x, double a) {
  auto y = torch::empty_like(x);
  int n = x.numel();
  scale_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), (float)a, n);
  return y;
}
"""

ext = load_inline(
    name="tiny_scale",
    cpp_sources="torch::Tensor scale(torch::Tensor x, double a);",
    cuda_sources=CUDA_SRC,
    functions=["scale"],
)
x = torch.randn(1 << 22, device="cuda")
torch.testing.assert_close(ext.scale(x, 2.0), 2 * x)
torch.cuda.synchronize()
print("custom CUDA extension OK")
