import inspect
from functools import wraps


L1_CACHE_BYTES = 65536


def get_l1_cache():
    global L1_CACHE_BYTES
    return L1_CACHE_BYTES


def set_l1_cache(new_cache):
    global L1_CACHE_BYTES
    L1_CACHE_BYTES = new_cache


class ConfigurableArg:
    pass


class EmptyKwarg:
    pass


def check_function_signature(signature):
    for param in signature.parameters.values():
        if param.annotation is ConfigurableArg and param.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise RuntimeError(f'ConfigurableArg must be keyword-only - {param.name}')
        elif param.annotation is not ConfigurableArg and param.kind is inspect.Parameter.KEYWORD_ONLY:
            raise RuntimeError(f'non-ConfigurableArg must not be keyword-only - {param.name}')


def get_configurable_args_from_signature(signature):
    return [param.name for param in signature.parameters.values() if param.annotation is ConfigurableArg]


def get_usual_args_from_signature(signature):    
    return [
        param.name for param in signature.parameters.values() 
        if param.kind is inspect.Parameter.POSITIONAL_ONLY or param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]


def is_configurators_fit(configurable_args, configurators_names):
    if set(configurable_args) != set(configurators_names):
        raise RuntimeError(f'Configuration mismatch, {configurable_args=}, {configurators_names=}')


def configurator_call(args, configurator, usual_args):
    conf_sign = inspect.signature(configurator)

    # take argnames from configurator, map args with respect to origin function's argnames
    conf_args = [args[usual_args.index(param.name)] for param in conf_sign.parameters.values()]

    return configurator(*conf_args)


class ConfiguredFunction:
    def __init__(self, origin_function, signature, configurable_args, usual_args, **configurators):
        self.origin_function = origin_function
        self.signature = signature
        self.configurable_args = configurable_args
        self.usual_args = usual_args
        self.configurators = configurators


    def __call__(self, *args, **kwargs):
        tmp_kwargs = {**{arg: EmptyKwarg for arg in self.configurable_args}, **kwargs}
        
        bind = self.signature.bind(*args, **tmp_kwargs)
        bind.apply_defaults()

        configured_kwargs = {
            k: configurator_call(bind.args, self.configurators[k], self.usual_args)
            if input_v is EmptyKwarg else input_v 
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
