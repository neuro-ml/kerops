import pytest


@pytest.fixture(params=[2, 4, 8, 16, 32])
def channels(request):
    return request.param


@pytest.fixture(params=[16, 32])
def channels_out(request):
    return request.param


@pytest.fixture(params=[1, 2, 3, 4])
def bsize(request):
    return request.param


@pytest.fixture(params=[2, 3, 7, 13, 16, 32, 37, 53, 111, 128])
def other_1(request):
    return request.param


@pytest.fixture(params=[2, 3, 7, 13, 16, 32, 37, 53, 111, 128])
def other_2(request):
    return request.param


@pytest.fixture(params=[2, 3, 7, 13, 16, 32, 37, 53, 111, 128])
def other_3(request):
    return request.param
