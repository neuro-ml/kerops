# Kerops
Fast algorithms for GPU

# Install
```shell
pip install kerops
```

# What is implemented
Activations are fp16 `(N, C, H, W, D)` in `channels_last_3d` with C a power of 2; stats are the per-channel mean and mean
of squares over N, H, W, D in fp32. Ops are in `kerops.ops`.

| kerops op | torch equivalent, weights in torch layout | backward op: gradients it returns |
|:----------|:-----------------|:----------------------------------|
| `conv.Conv3d(x, w)`, `w` of `[3, 3, 3, C_in, C_out]` | `F.conv3d(x, w, padding=1)` | `conv.Conv3dWgrad(grad, x)`: `w` |
| `conv.DWConv(x, w)`, `w` of `[3, 3, 3, C]` | `F.conv3d(x, w, padding=1, groups=C)` | `conv.DWConvWGRAD(x, grad)`: `w` |
| `linear.ReLULinearAdd(x, w, y)`, `w` of `[C, C / 2]` | `F.relu(x) @ w + y` over channels | `linear.ReLULinearBackward(x, grad, w)`: `x`, `w` |
| `linear.LinBReLULinAdd(x, w_up, w_down, b, y)`, `w_up` of `[C, 2C]` | `F.relu(x @ w_up + b) @ w_down + y` over channels | `linear.LinBReLULinBackward(x, grad, w_up, w_down, b)`: `x`, `w_up`, `w_down`, `b` |
| `addition.AddStats(x, y)` | `x + y` and its stats | `addition.AddStatsBackward(grad, mean_grad, sqmean_grad, x + y)`: `x` |
| `avgpool.AvgPoolCeilStats(x)` | `F.avg_pool3d(x, 2, ceil_mode=True)` and its stats | `avgpool.AvgPoolCeilStatsBackward(grad, mean_grad, sqmean_grad, out, x.shape)`: `x` |
| `bnrelu.ApplyBNReLU(x, w, b)` | `F.relu(x * w + b)` per channel: batch norm with stats folded into `w`, `b` | `bnrelu.ApplyBNReLUBackward(x, w, b, grad)`: `x`, `w`, `b` |
| `stats.Stats(x)` | stats of `x` | `stats.StatsBackward(x, mean_grad, sqmean_grad)`: `x` |
| `quantization.QuantUint8Window(x, window)` | `((x.clamp(-window, window) + window) / (2 * window) * 255).to(torch.uint8)` | `quantization.DequantUint8Window(q, dtype, window)`: `x` back |

# How fast is it?
Launch parameters are tuned for the NVIDIA RTX 4090; [BENCHMARK.md](BENCHMARK.md) has the speedup of every op over the
torch code it replaces, by channels and input size. `python scripts/benchmark.py > BENCHMARK.md` regenerates it on the
local GPU.

# How ops are configured
Every op is a `ConfiguredFunction`: tensors are positional-only arguments, kernel launch parameters are keyword-only.

```text
scripts/autotune.py ── best params of this GPU ──▶ kerops/ops/assets/<Op>.toml
                                                         │ read at import
                                                         ▼
op(*tensors, **overrides) ──▶ ConfiguredFunction ◀── TableKernelConfig: params for
                                      │                (GPU of the tensors, args_to_problem_sizes(*tensors))
                                      ▼
                       kernel launch with the params, overrides win
```

- An op without a table takes its launch parameters from their defaults.
- `op.can_be_configured(*tensors)` tells whether the table has params for this GPU and these sizes.
- At import, the TOML params must equal the op's keyword-only arguments, and the keys returned by
  `args_to_problem_sizes` must equal the TOML's `problem_size_names`.
- A GPU missing from a table raises an error; `python scripts/autotune.py <target>` (e.g. `conv`) on that GPU
  adds it and keeps the others.
