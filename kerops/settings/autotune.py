import itertools
import traceback as tb
import torch
from pathlib import Path
from typing import Callable
from tqdm.notebook import tqdm
from time import perf_counter, sleep
from warnings import warn

import numpy as np
from joblib import Parallel, delayed

from .utils import get_device_name_from_args


def mean_std_percentile(x, lo=20, hi=80):
    x = np.asarray(x, dtype=np.float32)
    p_lo, p_hi = np.percentile(x, [lo, hi])
    mask = (x >= p_lo) & (x <= p_hi)
    x_mid = x[mask]
    return x_mid.mean(), x_mid.std()


def bench_single(
    func,
    args,
    keys,
    configs,
    warmup,
    sleep_ms,
    n_iters,
    quantiles,
):
    results = []
    device = get_device_name_from_args(*args)

    for config in tqdm(configs, desc="Benchmark configs", leave=False):
        kwargs = dict(zip(keys, config))

        if sleep_ms is not None:
            sleep(sleep_ms / 1000)

        try:
            func(*args, **kwargs)
            torch.cuda.synchronize()
        except Exception:
            results.append({"spec": kwargs, "mean_ms": float("inf"), "std_ms": 0.0})
            continue

        for _ in range(warmup):
            func(*args, **kwargs)
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(n_iters):
            start = perf_counter()
            func(*args, **kwargs)
            torch.cuda.synchronize()
            end = perf_counter()
            times_ms.append((end - start) * 1e3)

        mean, std = mean_std_percentile(times_ms, *quantiles)
        results.append({"spec": kwargs, "mean_ms": float(mean), "std_ms": float(std), "device": device})

    return results


def compare(best_entry, comparator, args, n_iters, quantiles):
    best_ms = best_entry['mean_ms']

    start = perf_counter()
    end = perf_counter()

    times_ms = []
    for _ in range(n_iters):
        start = perf_counter()
        comparator(*args)
        torch.cuda.synchronize()
        end = perf_counter()
        times_ms.append((end - start) * 1e3)

    mean_comparator, _ = mean_std_percentile(times_ms, *quantiles)

    ratio = mean_comparator / best_ms

    return ratio


def _build_toml(
    device: str,
    problem_size_names: list[str],
    entries: list[dict],  # [{"problem_sizes": [...], **kernel_params}]
) -> str:
    lines = []

    lines.append("[meta]")
    lines.append(f'problem_size_names = {problem_size_names}')
    lines.append("")

    lines.append(f'["{device}"]')
    for entry in entries:
        lines.append(f'  [["{device}".configs]]')
        ps = list(entry["problem_sizes"])
        lines.append(f"  problem_sizes = {ps}")
        for k, v in entry["kernel_params"].items():
            val = str(v).lower() if isinstance(v, bool) else v
            lines.append(f"  {k} = {val}")
        lines.append("")

    return "\n".join(lines)


def autotune(
    func,
    generate_inputs: Callable[[dict], tuple],
    problem_sizes: list[dict],
    pruning_rule: Callable[[dict, dict], bool] | None = None,
    toml_path: str | Path | None = None,
    n_jobs_precompile: int = 4,
    warmup: int = 25,
    sleep_ms: int = 100,
    n_iters: int = 100,
    quantiles: tuple = (20, 80),
    comparator: Callable | None = None,
    **specset,
):
    keys = list(specset.keys())
    values = list(specset.values())
    configs = list(itertools.product(*values))

    problem_size_names = list(next(iter(problem_sizes)).keys())
    assert all(set(problem_size_names) == set(problem_size.keys()) for problem_size in problem_sizes)

    def precompile_call(*args, config):
        try:
            kwargs = dict(zip(keys, config))
            func(*args, **kwargs)
        except Exception as e:
            return ''.join(tb.format_exception(e))
        
        return None

    n_jobs = min(n_jobs_precompile, len(configs))
    toml_entries = []
    devices = []

    for problem_size in tqdm(problem_sizes, desc="Problem sizes", leave=False):
        args = generate_inputs(problem_size)
        ps_values = list(problem_size.values())
        pruned_configs = [config for config in configs if pruning_rule is None or pruning_rule(problem_size, dict(zip(keys, config)))]

        precompile_statuses = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(precompile_call)(*args, config=config)
            for config in tqdm(pruned_configs, desc="Precompiling", leave=False)
        )

        if not any(status is None for status in precompile_statuses):
            example_tb = next(status for status in precompile_statuses if status is not None)

            raise RuntimeError(f'Precompilation failed - all configs cause exception.\nExample:\n{example_tb}')

        results = bench_single(func, args, keys, pruned_configs, warmup, sleep_ms, n_iters, quantiles)

        valid = [r for r in results if r["mean_ms"] != float("inf")]
        if not valid:
            warn(f"No valid configs for problem_size={problem_size}, skipping", stacklevel=2)
            continue

        best = min(valid, key=lambda r: r["mean_ms"])

        if comparator is not None:
            ratio = compare(best, comparator, args, n_iters, quantiles)
            print(f'{problem_size=} best ratio - {ratio:.3f} (bigger is better)')

        toml_entries.append({
            "problem_sizes": ps_values,
            "kernel_params": best["spec"],
        })

        devices.append(best["device"])

    devices_found = set(device for device in devices)
    assert len(devices_found) == 1, f"Expected single-device autotune, got devices {devices_found}"

    if toml_path is not None:
        toml_str = _build_toml(devices_found.pop(), problem_size_names, toml_entries)
        Path(toml_path).write_text(toml_str, encoding="utf-8")
