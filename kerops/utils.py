import torch
from torch.nn import functional as F


def assert_same_shape(result, reference):
    assert result.shape == reference.shape, f'shape {list(result.shape)} != reference {list(reference.shape)}'


def small_integers(*shape, magnitude=2, dtype=torch.float16):
    """Uniform integers from -magnitude to magnitude on CUDA: sums of their products are exact while they stay small."""
    return torch.randint(-magnitude, magnitude + 1, shape, device='cuda', dtype=dtype)


def sparse_signs(*shape, sum_length):
    """fp16 ±1 at so few positions that a sum of products of two such over `sum_length` positions spreads by about 250,
    within the integers fp16 holds, 0 elsewhere."""
    signs = torch.randint(0, 2, shape, device='cuda', dtype=torch.float16) * 2 - 1
    return signs * (torch.rand(shape, device='cuda') < 250 / sum_length**0.5)


@torch.no_grad()
def assert_exact(result, reference):
    """Raises unless `result` equals the float64 `reference`, which inputs of small integers keep exact in the dtype of
    `result`; a reference that dtype cannot hold means the test's inputs are too large."""
    assert_same_shape(result, reference)
    held = reference.to(result.dtype).double()
    assert torch.equal(held, reference), f'the reference is not exact in {result.dtype}: the test inputs are too large'
    torch.testing.assert_close(result.double(), reference, rtol=0, atol=0)


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
