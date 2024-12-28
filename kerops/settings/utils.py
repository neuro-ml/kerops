from inspect import Parameter


class ConfigurableArg:
    pass


def check_function_signature(signature):
    for param in signature.parameters.values():
        if param.annotation is ConfigurableArg and param.kind is not Parameter.KEYWORD_ONLY:
            raise RuntimeError(f'ConfigurableArg must be keyword-only - {param.name}')
        elif param.annotation is not ConfigurableArg and param.kind is Parameter.KEYWORD_ONLY:
            raise RuntimeError(f'non-ConfigurableArg must not be keyword-only - {param.name}')


def get_configurable_args_from_signature(signature):
    return [param.name for param in signature.parameters.values() if param.annotation is ConfigurableArg]


def get_usual_args_from_signature(signature):
    return [
        param.name
        for param in signature.parameters.values()
        if param.kind is Parameter.POSITIONAL_ONLY or param.kind is Parameter.POSITIONAL_OR_KEYWORD
    ]


def is_configurators_fit(configurable_args, configurators_names):
    if set(configurable_args) != set(configurators_names):
        raise RuntimeError(f'Configuration mismatch, {configurable_args=}, {configurators_names=}')
