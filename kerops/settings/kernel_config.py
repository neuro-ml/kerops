import tomllib
from inspect import signature as get_signature
from pathlib import Path
from typing import Any, Callable

from .utils import get_device_name_from_args


class TableKernelConfig:
    def __init__(
        self,
        toml_path: str | Path,
        args_to_problem_sizes: Callable[..., dict[str, int]],
    ):
        sig = get_signature(args_to_problem_sizes)

        self.arg_names = [param.name for param in sig.parameters.values()]
        self.args_to_problem_sizes = args_to_problem_sizes

        # { gpu_name: { problem_sizes: configured_args_dict } }
        self._configs: dict[str, dict[tuple, dict[str, Any]]] = {}
        self.problem_size_names: list[str] = []
        self.confarg_names: set[str] = set()

        self.load_toml(toml_path)

    def load_toml(self, path: str | Path):
        path = Path(path)
        with open(path, 'rb') as f:
            raw = tomllib.load(f)

        meta = raw.get('meta')
        if meta is None:
            raise ValueError('TOML must contain a [meta] section')

        toml_problem_size_names = meta.get('problem_size_names')
        if toml_problem_size_names is None:
            raise ValueError('[meta] must contain problem_size_names')
        self.problem_size_names = list(toml_problem_size_names)

        for gpu_name, gpu_data in raw.items():
            if gpu_name == 'meta':
                continue

            configs_raw = gpu_data.get('configs')
            if not isinstance(configs_raw, list):
                raise ValueError(f'[{gpu_name}] must contain an array of configs')

            parsed: dict[tuple, dict[str, Any]] = {}

            for entry in configs_raw:
                if 'problem_sizes' not in entry:
                    raise ValueError(f'[{gpu_name}] config entry missing "problem_sizes" field: {entry}')

                problem_sizes = tuple(entry['problem_sizes'])
                if len(problem_sizes) != len(self.problem_size_names):
                    raise ValueError(
                        f'[{gpu_name}] problem_sizes tuple length {len(problem_sizes)} '
                        f'!= problem_size_names length {len(self.problem_size_names)}'
                    )

                parsed[problem_sizes] = {k: v for k, v in entry.items() if k != 'problem_sizes'}

            self._configs[gpu_name] = parsed

        names = {frozenset(configured) for configs in self._configs.values() for configured in configs.values()}
        if len(names) != 1:
            raise ValueError(
                f'{path}: config entries must all configure the same args, got {[sorted(n) for n in names]}'
            )
        self.confarg_names = set(names.pop())

    def problem_sizes(self, *input_args) -> tuple:
        """Problem sizes of `input_args` in the order of the TOML's problem_size_names."""
        problem_sizes = self.args_to_problem_sizes(*input_args)
        if problem_sizes.keys() != set(self.problem_size_names):
            raise ValueError(
                f'{self}: args_to_problem_sizes gives {list(problem_sizes)}, the table is indexed by '
                f'{self.problem_size_names}'
            )
        return tuple(problem_sizes[name] for name in self.problem_size_names)

    def __call__(self, *input_args) -> dict[str, Any]:
        device = get_device_name_from_args(*input_args)

        problem_sizes = self.problem_sizes(*input_args)

        if device not in self._configs:
            raise RuntimeError(f'{self} is not configured for {device}')

        if problem_sizes not in self._configs[device]:
            raise KeyError(f'No config for problem_sizes {problem_sizes} on device {device}')

        return self._configs[device][problem_sizes]

    def can_be_configured(self, *input_args) -> bool:
        try:
            device = get_device_name_from_args(*input_args)
        except RuntimeError:
            return False

        return self.problem_sizes(*input_args) in self._configs.get(device, {})

    def __repr__(self):
        input_s = ', '.join(self.arg_names)
        inter_s = ', '.join(self.problem_size_names)
        conf_s = ', '.join(sorted(self.confarg_names))
        return f'TableKernelConfig({input_s})' f' -[{inter_s}]-> ({conf_s})'
