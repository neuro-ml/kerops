L1_CACHE_BYTES = 65536


def get_l1_cache():
    global L1_CACHE_BYTES
    return L1_CACHE_BYTES


def set_l1_cache(new_cache):
    global L1_CACHE_BYTES
    L1_CACHE_BYTES = new_cache
