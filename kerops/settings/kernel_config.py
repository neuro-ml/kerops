import tomllib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Any
from inspect import signature as get_signature
from warnings import warn

from .utils import get_device_name_from_args


class KernelConfigBase(ABC):
    @abstractmethod
    def __call__(self, *input_args) -> dict[str, Any]:
        ...

    @abstractmethod
    def __repr__(self):
        ...

    def can_be_configured(self, *input_args):
        try:
            self(*input_args)
            return True
        except Exception:
            return False


class StaticKernelConfig(KernelConfigBase):
    def __init__(self, name: str | None = None, **params):
        self.params = params
        self.arg_names = []
        self.confarg_names = list(params.keys())
        self.name = name

    def can_be_configured(self):
        return True

    def __call__(self):
        return self.params
    
    def __repr__(self):
        prefix = f'{self.name}: ' if self.name else ''
        input_s = ', '.join(self.arg_names)
        conf_s  = ', '.join(self.confarg_names)
        return (
            f'{prefix}StaticKernelConfig({input_s})'
            f' -> ({conf_s})'
        )


class RuleKernelConfig(KernelConfigBase):
    def __init__(
        self,
        args_to_problem_sizes: Callable,
        problem_size_names: list[str],
        rule: Callable,
        confarg_names: list[str],
        name: str | None = None
    ):
        sig = get_signature(args_to_problem_sizes)

        self.arg_names = [param.name for param in sig.parameters.values()]
        self.problem_size_names = problem_size_names
        self.args_to_problem_sizes = args_to_problem_sizes

        sig = get_signature(rule)

        if any(param.name != problem_size_name for param, problem_size_name in zip(sig.parameters.values(), problem_size_names, strict=True)):
            raise ValueError(f'Rule required {list(sig.parameters.values())} as input, but provided {problem_size_names}')

        self.confarg_names = confarg_names
        self.rule = rule
        self.name = name

    def __call__(self, *input_args):
        problem_sizes = self.args_to_problem_sizes(*input_args)

        if not isinstance(problem_sizes, tuple):
            problem_sizes = (problem_sizes,)

        return self.rule(*problem_sizes)

    def __repr__(self):
        prefix = f'{self.name}: ' if self.name else ''
        input_s = ', '.join(self.arg_names)
        inter_s = ', '.join(self.problem_size_names)
        conf_s  = ', '.join(self.confarg_names)
        return (
            f'{prefix}RuleKernelConfig({input_s})'
            f' -[{inter_s}]-> ({conf_s})'
        )


class TableKernelConfig(KernelConfigBase):
    def __init__(
        self,
        problem_size_names: list[str],
        confarg_names: list[str],
        args_to_problem_sizes: Callable,
        fallback_device: str | None = None,
        toml_path: str | None = None,
        name: str | None = None
    ):
        sig = get_signature(args_to_problem_sizes)
        
        self.arg_names = [param.name for param in sig.parameters.values()]
        self.problem_size_names = problem_size_names
        self.confarg_names = confarg_names
        self.args_to_problem_sizes = args_to_problem_sizes

        # { gpu_name: { problem_sizes: configured_args_dict } }
        self._configs: dict[str, dict[tuple, dict[str, Any]]] = {}
        self.fallback_device = fallback_device
        self.name = name

        if toml_path is not None:
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
    
        if list(toml_problem_size_names) != self.problem_size_names:
            raise ValueError(
                f'problem_size_names mismatch: '
                f'toml={toml_problem_size_names}, config={self.problem_size_names}'
            )

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

                configured = {k: v for k, v in entry.items() if k != 'problem_sizes'}
                missing = set(self.confarg_names) - configured.keys()
                extra = configured.keys() - set(self.confarg_names)

                if missing:
                    raise ValueError(f'[{gpu_name}] config entry missing keys: {missing}')
                if extra:
                    raise ValueError(f'[{gpu_name}] config entry has unexpected keys: {extra}')

                parsed[problem_sizes] = configured

            self._configs[gpu_name] = parsed

        if self.fallback_device is not None and self.fallback_device not in self._configs:
            raise ValueError(
                f'fallback_device={self.fallback_device!r} not found in loaded configs: {list(self._configs)}'
            )

    def __call__(self, *input_args) -> dict[str, Any]:
        device = get_device_name_from_args(*input_args)

        problem_sizes = self.args_to_problem_sizes(*input_args)

        if not isinstance(problem_sizes, tuple):
            problem_sizes = (problem_sizes,)

        if device not in self._configs:
            if self.fallback_device is None:
                raise RuntimeError("Fallback device has not been set")

            warn(f'{self} is not configured for {device}, fallback to {self.fallback_device}', stacklevel=2)
            device_configs = self._configs[self.fallback_device]
            device = self.fallback_device
        else:
            device_configs = self._configs[device]

        if problem_sizes not in device_configs:
            raise KeyError(f'No config for problem_sizes {problem_sizes} on device {device}')

        return device_configs[problem_sizes]

    def __repr__(self):
        prefix = f'{self.name}: ' if self.name else ''
        input_s = ', '.join(self.arg_names)
        inter_s = ', '.join(self.problem_size_names)
        conf_s  = ', '.join(self.confarg_names)
        return (
            f'{prefix}TableKernelConfig({input_s})'
            f' -[{inter_s}]-> ({conf_s})'
        )
