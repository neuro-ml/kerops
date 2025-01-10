import torch
from torch.nn import functional as F


@torch.no_grad
def allclose_two_stage(
    input,
    other,
    rtol_strict=1e-4,
    atol_strict=1e-4,
    percentile_strict=0.99,
    rtol_narrow=1e-3,
    atol_narrow=1e-3,
    debug_info=None,
    precision=None,
):
    if isinstance(debug_info, str):
        if debug_info == 'print':
            debug_info = print
        else:
            raise ValueError(f'Unknown debug_info mode - {debug_info}')
    elif debug_info is None:
        debug_info = lambda x: None


    assert input.shape == other.shape
    assert input.dtype == other.dtype
    assert input.device == other.device

    if precision is None:
        precision = input.dtype

    input = input.reshape(-1)
    other = other.reshape(-1)

    strict_map = torch.abs(input.to(precision) - other.to(precision)) < atol_strict + rtol_strict * torch.abs(
        other.to(precision)
    )

    size = input.numel()
    num_true = strict_map.sum().item()

    ratio = num_true / size
    more_less = '>=' if ratio >= percentile_strict else '<'
    debug_info(f'strict stage - {size=}, {num_true=}, {ratio=} {more_less} percentile_strict')

    if ratio >= percentile_strict:
        debug_info('Strict threshold has been passed')
    else:
        debug_info('Strict threshold has not been passed')

    if size == num_true:
        debug_info('Strict threshold has been passed completely, propose decreasing rtol_strict and atol_strict')
        return True

    if num_true / size > percentile_strict:
        indices = torch.nonzero(~strict_map)

        input = input[indices].to(precision)
        other = other[indices].to(precision)

        narrow_map = torch.abs(input - other) < atol_narrow + rtol_narrow * torch.abs(other)

        if narrow_map.all():
            debug_info('Soft threshold has been passed')
            return True
        else:
            debug_info('Soft threshold has not been passed')

    return False


def weight_grad_similarity(
    input,
    other,
    rtol_cos=1e-4,
    atol_cos=1e-4,
    rtol_len=1e-3,
    atol_len=1e-3,
    debug_info=None
):
    if isinstance(debug_info, str):
        if debug_info == 'print':
            debug_info = print
        else:
            raise ValueError(f'Unknown debug_info mode - {debug_info}')
    elif debug_info is None:
        debug_info = lambda x: None

    assert input.shape == other.shape
    assert input.dtype == other.dtype
    assert input.device == other.device

    input_flat = input.reshape(-1)
    other_flat = other.reshape(-1)

    cos_sim = F.cosine_similarity(input_flat, other_flat, dim=0)

    if not torch.allclose(torch.ones_like(cos_sim), cos_sim, rtol=rtol_cos, atol=atol_cos):
        debug_info(f'Direction test failed - {cos_sim}')
        return False

    magnitudes_close = torch.allclose(
        torch.norm(input_flat),
        torch.norm(other_flat),
        rtol=rtol_len,
        atol=atol_len
    )

    if not magnitudes_close:
        debug_info(f'Magnitude test failed - {torch.norm(input_flat)=}, {torch.norm(other_flat)=}')
        return False

    debug_info('All stages passed')

    return True
