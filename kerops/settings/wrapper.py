import inspect
from functools import wraps
from typing import Callable

from .utils import (
    check_function_signature,
    get_configurable_args_from_signature,
    get_usual_args_from_signature,
    is_configurators_fit,
)


class EmptyKwarg:
    pass


class ConfiguredFunction:
    def __init__(self, origin_function, signature, configurable_args, usual_args, **configurators):
        self.origin_function = origin_function
        self.signature = signature
        self.configurable_args = configurable_args
        self.usual_args = usual_args
        self.configurators = configurators

    @staticmethod
    def configurator_call(args, configurator, usual_args):
        if isinstance(configurator, Callable):
            conf_sign = inspect.signature(configurator)

            # take argnames from configurator, map args with respect to origin function's argnames
            conf_args = [args[usual_args.index(param.name)] for param in conf_sign.parameters.values()]

            return configurator(*conf_args)
        else:
            return configurator

    def __call__(self, *args, **kwargs):
        tmp_kwargs = {**{arg: EmptyKwarg for arg in self.configurable_args}, **kwargs}

        bind = self.signature.bind(*args, **tmp_kwargs)
        bind.apply_defaults()

        configured_kwargs = {
            k: (
                self.configurator_call(bind.args, self.configurators[k], self.usual_args)
                if input_v is EmptyKwarg
                else input_v
            )
            for k, input_v in bind.kwargs.items()
        }

        return self.origin_function(*bind.args, **configured_kwargs)

    def reconfigure(self, **new_configurators):
        is_configurators_fit(self.configurable_args, new_configurators.keys())
        self.configurators = new_configurators


def configure(**configurators):
    def wrapper(function):
        signature = inspect.signature(function)

        check_function_signature(signature)

        configurable_args = get_configurable_args_from_signature(signature)

        usual_args = get_usual_args_from_signature(signature)

        is_configurators_fit(configurable_args, configurators.keys())

        return wraps(function)(ConfiguredFunction(function, signature, configurable_args, usual_args, **configurators))

    return wrapper
