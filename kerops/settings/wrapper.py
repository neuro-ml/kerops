from inspect import Parameter, signature as get_signature
from functools import wraps
from typing import Callable

from .kernel_config import KernelConfigBase


class ConfArg:
    pass


EmptyKwarg = object()


class ConfiguredFunction:
    def __init__(self, function: Callable, kernel_config: KernelConfigBase):
        self.function = function

        signature = get_signature(function)
        for param in signature.parameters.values():
            if param.kind is Parameter.VAR_POSITIONAL:
                raise TypeError(f'VAR_POSITIONAL (*args) is not supported - {param.name}')

            elif param.annotation is ConfArg:
                if param.kind is not Parameter.KEYWORD_ONLY:
                    raise TypeError(f'ConfArg must be keyword-only - {param.name}')
                
                if param.default is not param.empty:
                    raise TypeError(f'ConfArg must not have default value - {param.name}:{param.default}')

            elif param.annotation is not ConfArg and param.kind is Parameter.KEYWORD_ONLY:
                raise TypeError(f'non-ConfArg must not be keyword-only - {param.name}')

        self.signature = signature

        self.confargs = [param.name for param in self.signature.parameters.values() if param.annotation is ConfArg]
        self.usual_args = [param.name for param in self.signature.parameters.values() if param.annotation is not ConfArg]

        self.register_kernel_config(kernel_config)

    def register_kernel_config(self, kernel_config: KernelConfigBase):
        configured_arg_names = kernel_config.confarg_names
        input_arg_names = kernel_config.arg_names

        if set(self.confargs) != set(configured_arg_names):
            raise ValueError(
                f'Configuration mismatch, confargs={self.confargs}, configured_arg_names={configured_arg_names}'
            )

        for arg in input_arg_names:
            if arg not in self.usual_args:
                raise ValueError(f"{kernel_config.__class__.__name__} expects unknown arg: {arg}")

        self.kernel_config = kernel_config
        self.kernel_input_arg_indices = [self.usual_args.index(arg) for arg in input_arg_names]

    def kernel_config_call(self, usual_args):
        return self.kernel_config(*(usual_args[idx] for idx in self.kernel_input_arg_indices))

    def __call__(self, *args, **kwargs):
        # all confargs and, maybe, some usual args passed as keyword-argument
        full_kwargs = {arg: EmptyKwarg for arg in self.confargs}
        full_kwargs.update(kwargs)

        bind = self.signature.bind(*args, **full_kwargs)
        bind.apply_defaults()

        # after binding kwargs consists of ConfArgs ONLY, and kwargs.keys() == self.confargs
        # args does no contain ant ConfArg
        args, kwargs = bind.args, bind.kwargs

        # if all ConfArgs are overriden there is no reason to call kernel_config
        if any(v is EmptyKwarg for v in kwargs.values()):
            # configuration - ConfArg is configured in priority order for overriden values from kwargs
            configured_kwargs = self.kernel_config_call(bind.args)

            configured_kwargs = {
                k: configured_kwargs[k] if v is EmptyKwarg else v
                for k, v in kwargs.items()
            }
        else:
            configured_kwargs = kwargs

        return self.function(*args, **configured_kwargs)

    @classmethod
    def configure(cls, kernel_config: KernelConfigBase):
        def wrapper(function: Callable):
            return wraps(function)(cls(function, kernel_config))

        return wrapper

    def __repr__(self):
        return f'{self.function.__name__}{self.signature}\n{self.kernel_config}'
