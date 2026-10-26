import itertools
import json
import tomllib
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path
from typing import Callable
from warnings import warn

import triton
from tqdm.auto import tqdm
from triton import knobs, language as tl
from triton.compiler.compiler import ASTSource, make_backend
from triton.experimental.gluon._runtime import GluonASTSource
from triton.runtime import driver
from triton.runtime.jit import convert_to_tuple_if_list
from triton.testing import do_bench

from .utils import get_device_name_from_args

# (kernel module, kernel name, triton's serialized specialization): one kernel binary to compile
Compilation = tuple[str, str, str]

# The fastest configs of the first pass are timed again, longer and in interleaved rounds: one pass times configs one
# after another while the GPU warms up, which reorders configs that are within a few percent of each other.
FINALISTS = 16
FINAL_REP_MS = 300
FINAL_ROUNDS = 3


def record_compilations(func, args, config) -> tuple[list[Compilation], str]:
    """Kernels `func` needs for `config`, recorded instead of compiled or launched; the traceback if it raised, else ''."""
    compilations = []

    def skip_compilation(fn, compile, **_):
        compilations.append((fn.module, fn.name, compile['specialization_data']))
        return True

    knobs.runtime.jit_cache_hook = skip_compilation
    try:
        func(*args, **config)
    except Exception as e:
        return compilations, ''.join(traceback.format_exception(e))
    finally:
        knobs.runtime.jit_cache_hook = None
    return compilations, ''


def compile_specialization(compilation: Compilation, target) -> str:
    """Compiles into triton's disk cache like `JITFunction.preload`, but without touching the GPU; the traceback or ''."""
    module, name, specialization_data = compilation
    try:
        kernel = getattr(import_module(module), name)
        data = json.loads(specialization_data)
        constexprs = {
            tuple(key): tl.dtype(value) if tl.dtype.is_dtype(value) else
            tl.constexpr(value['constexpr']) if isinstance(value, dict) and 'constexpr' in value else value
            for key, value in zip(data['constant_keys'], data['constant_vals'])
        }
        attrs = {tuple(key): value for key, value in zip(data['attrs_keys'], data['attrs_vals'])}
        signature = {key: convert_to_tuple_if_list(value) for key, value in data['signature'].items()}
        options = {key: tuple(value) if isinstance(value, list) else value for key, value in data['options'].items()}
        options = make_backend(target).parse_options(options)
        source = (GluonASTSource if kernel.is_gluon() else ASTSource)(kernel, signature, constexprs, attrs)
        triton.compile(source, target=target, options=options.__dict__)
    except Exception as e:
        return ''.join(traceback.format_exception(e))
    return ''


def compile_specializations(compilations: list[Compilation], target, n_threads: int) -> list[str]:
    """Worker process: compiles in `n_threads` threads; tracebacks, '' where compiled."""
    with ThreadPoolExecutor(n_threads) as threads:
        return list(threads.map(compile_specialization, compilations, itertools.repeat(target)))


def compile_in_processes(func, generate_inputs, problem_sizes, configs_per_size, n_processes, n_threads) -> list[list[str]]:
    """Compiles every config into triton's disk cache in GPU-free processes; tracebacks per problem size and config, '' where compiled."""
    recorded_per_size = []
    for problem_size, configs in zip(problem_sizes, configs_per_size):
        args = generate_inputs(problem_size)
        recorded_per_size.append([record_compilations(func, args, config) for config in configs])
        del args

    unique = sorted({compilation for recorded in recorded_per_size for compilations, _ in recorded for compilation in compilations})
    target = driver.active.get_current_target()
    with ProcessPoolExecutor(n_processes, mp_context=get_context('spawn')) as pool:
        futures = [pool.submit(compile_specializations, unique[i:i + n_threads], target, n_threads) for i in range(0, len(unique), n_threads)]
        compile_errors = dict(zip(unique, (error for future in tqdm(futures, desc='Compiling') for error in future.result())))

    return [
        [error or next((compile_errors[c] for c in compilations if compile_errors[c]), '') for compilations, error in recorded]
        for recorded in recorded_per_size
    ]


def bench_ms(func, args, config, warmup_ms, rep_ms) -> float:
    try:
        return do_bench(lambda: func(*args, **config), warmup=warmup_ms, rep=rep_ms, return_mode='median')
    except Exception:
        return float('inf')


def best_of_finalists(func, args, finalists: list[dict], warmup_ms) -> tuple[float, dict]:
    """(median ms, config) of the fastest finalist over FINAL_ROUNDS interleaved rounds."""
    timings = [[] for _ in finalists]
    for _ in range(FINAL_ROUNDS):
        for config_timings, config in zip(timings, finalists):
            config_timings.append(bench_ms(func, args, config, warmup_ms, FINAL_REP_MS))
    medians = [sorted(config_timings)[len(config_timings) // 2] for config_timings in timings]
    best = min(range(len(finalists)), key=medians.__getitem__)
    return medians[best], finalists[best]


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
        if not configs:
            continue
        if all(errors):
            raise RuntimeError(f'Compilation failed for {problem_size} - all configs cause exception.\nExample:\n{errors[0]}')

        args = generate_inputs(problem_size)
        devices.add(get_device_name_from_args(*args))
        timings = [float('inf') if error else bench_ms(func, args, config, warmup_ms, rep_ms) for config, error in zip(configs, errors)]
        ranked = sorted((index for index, timing in enumerate(timings) if timing != float('inf')), key=timings.__getitem__)
        if not ranked:
            warn(f'No valid configs for {problem_size=}, skipping', stacklevel=2)
            continue
        best_ms, best = best_of_finalists(func, args, [configs[index] for index in ranked[:FINALISTS]], warmup_ms)

        if comparator is not None:
            ratio = do_bench(lambda: comparator(*args), warmup=warmup_ms, rep=FINAL_REP_MS, return_mode='median') / best_ms
            print(f'{problem_size=} best ratio - {ratio:.3f} (bigger is better)')

        entries.append({'problem_sizes': list(problem_size.values()), **best})

    assert len(devices) == 1, f'Expected single-device autotune, got devices {devices}'
    if toml_path is not None:
        write_toml(Path(toml_path), devices.pop(), list(problem_sizes[0]), entries)
