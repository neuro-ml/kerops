# Kerops
Fast algorithms for GPU

# Install
```shell
pip install kerops
```

# How fast is it?
Launch parameters are tuned for the NVIDIA RTX 4090. `python scripts/benchmark.py [ops...]` times every op against the
torch code it replaces, over channel counts and input sizes, and prints a table per op headed by the GPU and the kerops
version.

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
