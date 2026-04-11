from torch import Tensor
from torch.cuda import get_device_name


def get_device_name_from_args(*args):
    device_indices = {arg.device.index for arg in args if isinstance (arg, Tensor) and arg.device.type == 'cuda'}

    if len(device_indices) == 1:
        return get_device_name(device_indices.pop())
    elif len(device_indices) == 0:
        raise RuntimeError('Cannot configure due to non-cuda args')
    else:
        raise RuntimeError(f'Expected all tensors to be on the same GPU, got CUDA-devices:{device_indices}')
