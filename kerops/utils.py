import torch
from torch.nn import functional as F


def assert_same_shape(result, reference):
    assert result.shape == reference.shape, f'shape {list(result.shape)} != reference {list(reference.shape)}'


@torch.no_grad()
def assert_close(result, reference, max_error, mean_error):
    """Raises unless the error of `result` past one step of its dtype, in units of the reference's RMS, is at most
    max_error everywhere and mean_error on average."""
    assert_same_shape(result, reference)
    step = torch.finfo(result.dtype).eps
    result, reference = result.float(), reference.float()
    # A zero reference has no scale: against it any error is infinitely large.
    scale = reference.square().mean().sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    error = ((result - reference).abs() - step * reference.abs()).clamp_min(0) / scale
    worst = tuple(index.item() for index in torch.unravel_index(error.argmax(), error.shape))
    assert error.max() <= max_error, (
        f'max error past one {step:.0e} step {error.max():.1e} of the reference RMS > {max_error:.1e}, '
        f'at {worst}: {result[worst]:.6g} vs reference {reference[worst]:.6g}'
    )
    assert (
        error.mean() <= mean_error
    ), f'mean error past one {step:.0e} step {error.mean():.1e} of the reference RMS > {mean_error:.1e}'


@torch.no_grad()
def assert_wgrad_close(result, reference, cos_error, norm_error):
    """Raises unless 1 - cos(result, reference) <= cos_error and their norms differ by at most norm_error of the
    reference's; a zero reference, which has no direction, needs a zero result."""
    assert_same_shape(result, reference)
    if not reference.any():
        assert not result.any(), f'the reference is zero, the result has {result.count_nonzero()} nonzero values'
    else:
        result, reference = result.double().flatten(), reference.double().flatten()
        cos = 1 - F.cosine_similarity(result, reference, dim=0).item()
        norm = (result.norm() / reference.norm() - 1).abs().item()
        assert cos <= cos_error, f'1 - cos {cos:.1e} > {cos_error:.1e}'
        assert norm <= norm_error, f'norm differs by {norm:.1e} of the reference norm > {norm_error:.1e}'


def cdiv(a, b):
    assert type(a) is int
    assert type(b) is int

    return (a + b - 1) // b
