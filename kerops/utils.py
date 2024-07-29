import torch


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

    if debug_info:
        ratio = num_true / size
        more_less = '>=' if ratio >= percentile_strict else '<'
        debug_info(f'strict stage - {size=}, {num_true=}, {ratio=} {more_less} percentile_strict')

        if ratio >= percentile_strict:
            debug_info('Strict threshold has been passed')
        else:
            debug_info('Strict threshold has not been passed')

    if size == num_true:
        if debug_info:
            debug_info('Strict threshold has been passed completely, propose decreasing rtol_strict and atol_strict')
        return True

    if num_true / size > percentile_strict:
        indices = torch.nonzero(~strict_map)

        input = input[indices].to(precision)
        other = other[indices].to(precision)

        narrow_map = torch.abs(input - other) < atol_narrow + rtol_narrow * torch.abs(other)

        if narrow_map.all():
            if debug_info:
                debug_info('Soft threshold has been passed')
            return True
        elif debug_info:
            debug_info('Soft threshold has not been passed')

    return False
