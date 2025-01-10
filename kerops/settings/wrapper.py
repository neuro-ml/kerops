import inspect
from functools import wraps
from typing import Callable

from .utils import CongiguratorError, configs_match, get_config_args, get_standard_args, validate_signature


class EmptyKwarg:
    pass


class ConfiguredFunction:
    def __init__(self, origin_function, signature, configurable_args, usual_args, **configurators):
        self.origin_function = origin_function
        self.signature = signature
        self.configurable_args = configurable_args
        self.usual_args = usual_args
        self.configurators = configurators

    def __repr__(self):
        def format_configurator(configurator):
            if isinstance(configurator, Callable):
                params = ', '.join(inspect.signature(configurator).parameters)
                return f'Configurator({params})'
            return str(configurator)

        configurators_repr = '\n'.join(
            f'{confarg}: {format_configurator(configurator)}' for confarg, configurator in self.configurators.items()
        )

        return f'{self.origin_function.__name__}{self.signature}\n{configurators_repr}'

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

    def can_be_configured(self, **kwargs):
        try:
            for configurator in self.configurators.values():
                if isinstance(configurator, Callable):
                    confargs = inspect.signature(configurator).parameters
                    _ = configurator(**{arg: kwargs[arg] for arg in confargs})

        except CongiguratorError:
            return False

        return True

    def reconfigure(self, **new_configurators):
        configs_match(self.configurable_args, new_configurators.keys())
        self.configurators = new_configurators


def configure(**configurators):
    def wrapper(function):
        signature = inspect.signature(function)

        validate_signature(signature)

        configurable_args = get_config_args(signature)

        usual_args = get_standard_args(signature)

        configs_match(configurable_args, configurators.keys())

        return wraps(function)(ConfiguredFunction(function, signature, configurable_args, usual_args, **configurators))

    return wrapper


def confexc(*exceptions):
    def wrapper(configurator):
        @wraps(configurator)
        def wrapped(*args, **kwargs):
            try:
                return configurator(*args, **kwargs)
            except exceptions as e:
                raise CongiguratorError(str(e))

        return wrapped

    return wrapper
