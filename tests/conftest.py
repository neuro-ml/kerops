import pytest


@pytest.fixture(params=[2, 4, 8, 16, 32])
def channels(request):
    return request.param


@pytest.fixture(params=[16, 32])
def channels_out(request):
    return request.param


@pytest.fixture(params=[1, 2, 3])
def bsize(request):
    return request.param


@pytest.fixture(params=[2, 3, 13, 16, 32, 37, 53, 64])
def other_1(request):
    return request.param


@pytest.fixture(params=[2, 3, 13, 16, 32, 37, 53, 64])
def other_2(request):
    return request.param


@pytest.fixture(params=[2, 3, 13, 16, 32, 37, 53, 64])
def other_3(request):
    return request.param


@pytest.fixture(params=[16, 32, 64, 128])
def conv_in_channels(request):
    return request.param


@pytest.fixture(params=[16, 32, 64, 128])
def conv_out_channels(request):
    return request.param


# Small (bsize, H, W, D) with odd sides, which reach every border, for the exact tests.
@pytest.fixture(params=[(1, 13, 7, 37), (2, 16, 3, 32), (3, 5, 53, 2)], ids=lambda shape: '-'.join(map(str, shape)))
def exact_shape(request):
    return request.param


# (bsize, H, W, D) of a power-of-2 voxel count, which keeps means of small integers exact.
@pytest.fixture(params=[(2, 4, 8, 16), (1, 2, 2, 64), (4, 8, 4, 2)], ids=lambda shape: '-'.join(map(str, shape)))
def power_of_2_shape(request):
    return request.param
