from functools import wraps
from inspect import Parameter, signature
from typing import Callable

from .kernel_config import TableKernelConfig


class ConfiguredFunction:
    """Runs `function` with its keyword-only args from `kernel_config`, or from their defaults without one,
    unless passed explicitly."""

    def __init__(self, function: Callable, kernel_config: TableKernelConfig | None = None):
        params = list(signature(function).parameters.values())
        positional = [param.name for param in params if param.kind is Parameter.POSITIONAL_ONLY]
        keyword = [param for param in params if param.kind is Parameter.KEYWORD_ONLY]

        if len(positional) + len(keyword) != len(params):
            raise TypeError(f'{function.__name__}: args must be positional-only and launch parameters keyword-only')

        if kernel_config is None:
            if any(param.default is param.empty for param in keyword):
                raise TypeError(f'{function.__name__}: keyword-only args need defaults when there is no kernel_config')
        else:
            if any(param.default is not param.empty for param in keyword):
                raise TypeError(f'{function.__name__}: keyword-only args come from kernel_config and take no defaults')

            if {param.name for param in keyword} != kernel_config.confarg_names:
                raise ValueError(
                    f'{function.__name__}: keyword-only args {[param.name for param in keyword]} != configured args '
                    f'{sorted(kernel_config.confarg_names)}'
                )

            if kernel_config.arg_names != positional:
                raise ValueError(
                    f'{function.__name__}: args_to_problem_sizes takes {kernel_config.arg_names}, expected {positional}'
                )

        self.function = function
        self.kernel_config = kernel_config

    def __call__(self, *args, **overrides):
        if self.kernel_config is None:
            return self.function(*args, **overrides)
        return self.function(*args, **{**self.kernel_config(*args), **overrides})

    def can_be_configured(self, *args) -> bool:
        return self.kernel_config is None or self.kernel_config.can_be_configured(*args)

    @classmethod
    def configure(cls, kernel_config: TableKernelConfig | None = None):
        def wrapper(function: Callable):
            return wraps(function)(cls(function, kernel_config))

        return wrapper

    def __repr__(self):
        return f'{self.function.__name__}{signature(self.function)}\n{self.kernel_config}'
