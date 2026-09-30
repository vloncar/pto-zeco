"""Cost of the host-side transpose megagdn does between cumsum and kkt."""
import torch, torch_npu, time
T, H = 32768, 16
g = torch.randn(1, T, H, dtype=torch.float32).npu()
for _ in range(20):
    out = g.squeeze(0).t().contiguous()
torch.npu.synchronize()
ts = []
for _ in range(50):
    torch.npu.synchronize(); t0 = time.perf_counter()
    out = g.squeeze(0).t().contiguous()
    torch.npu.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
ts.sort()
print(f"[mega] torch transpose [T,H]->[H,T] median={ts[len(ts)//2]:.1f} us  min={ts[0]:.1f}")
