import torch

torch.ops.load_library("../csrc/jit/build/libwarp_perspective.so")
print(torch.ops.my_ops.warp_perspective)
print(torch.ops.my_ops.warp_perspective(torch.randn(32, 32), torch.rand(3, 3)))

def compute(x, y, z):
    x = torch.ops.my_ops.warp_perspective(x, torch.eye(3))
    return x.matmul(y) + torch.relu(z)

inputs = [torch.randn(4, 8), torch.randn(8, 5), torch.randn(8, 5)]
trace = torch.jit.trace(compute, inputs)
print(trace.graph)
