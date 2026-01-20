import copy

import torch
from torchao.quantization import Int8WeightOnlyConfig, quantize_
from torchao.utils import (
    benchmark_model,
)

tensor = torch.randn(1, 3)
quantized_tensor = torch.quantize_per_tensor(tensor, scale=0.1, zero_point=0, dtype=torch.qint8)
dequantized_tensor = quantized_tensor.dequantize()
print("Original tensor:", tensor)
print("Quantized tensor:", quantized_tensor)
print("Dequantized tensor:", dequantized_tensor)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


model = ToyLinearModel(1024, 1024, 1024).eval().to(torch.float32)

# Optional: compile model for faster inference and generation
model = torch.compile(model, mode="max-autotune", fullgraph=True)
model_f32 = copy.deepcopy(model)


quantize_(
    model,
    Int8WeightOnlyConfig(),
)


num_runs = 100
torch._dynamo.reset()
example_inputs = (torch.randn(1, 1024, dtype=torch.float32),)
bf32_time = benchmark_model(model_f32, num_runs, example_inputs)
int8_time = benchmark_model(model, num_runs, example_inputs)

print("bf32 mean time: %0.3f ms" % bf32_time)
print("int8 mean time: %0.3f ms" % int8_time)
print("speedup: %0.1fx" % (bf32_time / int8_time))
