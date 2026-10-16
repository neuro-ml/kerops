import itertools
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import Callable
from warnings import warn

from tqdm.auto import tqdm
from triton.runtime.jit import KernelInterface
from triton.testing import do_bench

from .utils import get_device_name_from_args


def _compile_instead_of_launch(kernel, grid):
    return lambda *args, **kwargs: kernel.run(*args, grid=grid, warmup=True, **kwargs)


def compile_configs(func_path: tuple[str, str], generate_inputs: Callable, problem_size: dict, configs: list[dict], n_threads: int) -> list[str]:
    """Worker process: compiles the kernels `func` launches for each config into triton's disk cache; returns tracebacks, '' where compiled."""
    KernelInterface.__getitem__ = _compile_instead_of_launch
    module, name = func_path
    func = getattr(import_module(module), name)
    func = getattr(func, 'function', func)
    args = generate_inputs(problem_size)

    def compile_config(config):
        try:
            func(*args, **config)
        except Exception as e:
            return ''.join(traceback.format_exception(e))
        return ''

    with ThreadPoolExecutor(n_threads) as threads:
        return list(threads.map(compile_config, configs))


def compile_in_processes(func, generate_inputs, problem_sizes, configs_per_size, n_processes, n_threads) -> list[list[str]]:
    """Tracebacks per problem size and config, '' where compiled."""
    func_path = (func.__module__, func.__qualname__)
    with ProcessPoolExecutor(n_processes, mp_context=get_context('spawn')) as pool:
        futures_per_size = [
            [
                pool.submit(compile_configs, func_path, generate_inputs, problem_size, configs[i:i + n_threads], n_threads)
                for i in range(0, len(configs), n_threads)
            ]
            for problem_size, configs in zip(problem_sizes, configs_per_size)
        ]
        return [
            [error for future in futures for error in future.result()]
            for futures in tqdm(futures_per_size, desc='Compiling')
        ]


def bench_ms(func, args, config, warmup_ms, rep_ms) -> float:
    try:
        return do_bench(lambda: func(*args, **config), warmup=warmup_ms, rep=rep_ms, return_mode='median')
    except Exception:
        return float('inf')


def write_toml(path: Path, device: str, problem_size_names: list[str], entries: list[dict]) -> None:
    """Writes `entries` as the configs of `device`, keeping the configs of other devices already in the file."""
    configs_per_device = {}
    if path.exists():
        configs_per_device = {name: data['configs'] for name, data in tomllib.loads(path.read_text()).items() if name != 'meta'}
    configs_per_device[device] = entries

    lines = ['[meta]', f'problem_size_names = {problem_size_names}', '']
    for name, configs in configs_per_device.items():
        lines.append(f'["{name}"]')
        for config in configs:
            lines.append(f'  [["{name}".configs]]')
            for key, value in config.items():
                lines.append(f'  {key} = {str(value).lower() if isinstance(value, bool) else value}')
            lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def autotune(
    func,
    generate_inputs: Callable[[dict], tuple],
    problem_sizes: list[dict],
    pruning_rule: Callable[[dict, dict], bool] | None = None,
    toml_path: str | Path | None = None,
    n_processes: int = 8,
    n_threads: int = 4,
    warmup_ms: int = 25,
    rep_ms: int = 100,
    comparator: Callable | None = None,
    **specset,
):
    """Compiles in spawned processes, so a calling script must guard it with `if __name__ == '__main__'`."""
    keys = list(specset)
    configs = [dict(zip(keys, values)) for values in itertools.product(*specset.values())]
    configs_per_size = [
        [config for config in configs if pruning_rule is None or pruning_rule(problem_size, config)]
        for problem_size in problem_sizes
    ]
    errors_per_size = compile_in_processes(func, generate_inputs, problem_sizes, configs_per_size, n_processes, n_threads)

    entries = []
    devices = set()
    for problem_size, configs, errors in tqdm(list(zip(problem_sizes, configs_per_size, errors_per_size)), desc='Benchmarking'):
        if all(errors):
            raise RuntimeError(f'Compilation failed for {problem_size} - all configs cause exception.\nExample:\n{errors[0]}')

        args = generate_inputs(problem_size)
        devices.add(get_device_name_from_args(*args))
        timings = [float('inf') if error else bench_ms(func, args, config, warmup_ms, rep_ms) for config, error in zip(configs, errors)]
        best_ms, best = min(zip(timings, configs), key=lambda timing_config: timing_config[0])
        if best_ms == float('inf'):
            warn(f'No valid configs for {problem_size=}, skipping', stacklevel=2)
            continue

        if comparator is not None:
            ratio = do_bench(lambda: comparator(*args), warmup=warmup_ms, rep=rep_ms, return_mode='median') / best_ms
            print(f'{problem_size=} best ratio - {ratio:.3f} (bigger is better)')

        entries.append({'problem_sizes': list(problem_size.values()), **best})

    assert len(devices) == 1, f'Expected single-device autotune, got devices {devices}'
    if toml_path is not None:
        write_toml(Path(toml_path), devices.pop(), list(problem_sizes[0]), entries)
