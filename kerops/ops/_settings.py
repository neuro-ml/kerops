from functools import partial
from warnings import warn


# TODO: dirty, factorize responsibility in another way
FUNC_NUM_WARPS = {
    'AddStats': 8,
    'AddStatsBackward': 8,
    'ApplyBNReLU': 8,
    'ApplyBNReLUBackward': 8,
    'AvgPoolCeilStats': 2,
    'AvgPoolCeilStatsBackward': 4,
    'Stats': 8,
    'StatsBackward': 8,
    'QuantUint8Window': 4,
    'DequantUint8Window': 4,
}
L1_CACHE_BYTES = 65536


def get_num_warps(function):
    global FUNC_NUM_WARPS
    if function in FUNC_NUM_WARPS:
        return FUNC_NUM_WARPS[function]
    raise ValueError(f'function {function} is not registred')


def get_l1_cache():
    global L1_CACHE_BYTES
    return L1_CACHE_BYTES


def register_function(function, num_warps):
    global FUNC_NUM_WARPS
    if function in FUNC_NUM_WARPS:
        warn(f'function {function} is already registred, num_warps is set as {num_warps}')
    FUNC_NUM_WARPS[function] = num_warps


def set_l1_cache(new_cache):
    global L1_CACHE_BYTES
    L1_CACHE_BYTES = new_cache


def settings_wrapper(kernel_func):
    func_name = kernel_func.__name__
    _l1_cache_bytes = get_l1_cache()
    _num_warps = get_num_warps(func_name)
    wrapped_kernel = partial(kernel_func, _l1_cache_bytes=_l1_cache_bytes, _num_warps=_num_warps)

    return wrapped_kernel
