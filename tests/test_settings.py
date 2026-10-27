from importlib import import_module
from pathlib import Path

import pytest
import torch

import kerops.ops
from kerops.settings import ConfiguredFunction, TableKernelConfig, kernel_config, utils


def channels(x, factor):
    return {'channels': x.shape[0]}


def launch_params(x, factor, /, *, BLOCK, num_warps):
    return BLOCK, num_warps


TOML = """
[meta]
problem_size_names = ['channels']

["GPU A"]
  [["GPU A".configs]]
  problem_sizes = [16]
  BLOCK = 32
  num_warps = 4
"""


def make_table(tmp_path, toml=TOML, args_to_problem_sizes=channels):
    path = tmp_path / 'table.toml'
    path.write_text(toml)
    return TableKernelConfig(path, args_to_problem_sizes)


def on_device(monkeypatch, device):
    monkeypatch.setattr(kernel_config, 'get_device_name_from_args', lambda *args: device)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    on_device(monkeypatch, 'GPU A')
    return ConfiguredFunction(launch_params, make_table(tmp_path))


def test_config_comes_from_table(configured):
    assert configured(torch.zeros(16), 2) == (32, 4)


def test_override_wins_over_table(configured):
    assert configured(torch.zeros(16), 2, num_warps=8) == (32, 8)


def test_unknown_override_raises(configured):
    with pytest.raises(TypeError):
        configured(torch.zeros(16), 2, BLOK=64)


def test_can_be_configured_only_for_tuned_sizes(configured):
    assert configured.can_be_configured(torch.zeros(16), 2)
    assert not configured.can_be_configured(torch.zeros(8), 2)


def test_untuned_device(configured, monkeypatch):
    on_device(monkeypatch, 'GPU B')
    assert not configured.can_be_configured(torch.zeros(16), 2)
    with pytest.raises(RuntimeError):
        configured(torch.zeros(16), 2)


def test_cpu_args_cannot_be_configured(tmp_path):
    assert not ConfiguredFunction(launch_params, make_table(tmp_path)).can_be_configured(torch.zeros(16), 2)


IN_OUT_TOML = """
[meta]
problem_size_names = ['in_channels', 'out_channels']

["GPU A"]
  [["GPU A".configs]]
  problem_sizes = [16, 32]
  BLOCK = 1

  [["GPU A".configs]]
  problem_sizes = [32, 16]
  BLOCK = 2
"""


def block(x, weight, /, *, BLOCK):
    return BLOCK


def test_problem_sizes_follow_toml_order(tmp_path, monkeypatch):
    on_device(monkeypatch, 'GPU A')
    table = make_table(
        tmp_path, IN_OUT_TOML, lambda x, weight: {'out_channels': weight.shape[1], 'in_channels': weight.shape[0]}
    )
    assert ConfiguredFunction(block, table)(torch.zeros(16), torch.zeros(16, 32)) == 1


def test_problem_size_names_must_match_toml(tmp_path, monkeypatch):
    on_device(monkeypatch, 'GPU A')
    table = make_table(tmp_path, args_to_problem_sizes=lambda x, factor: {'chanels': x.shape[0]})
    with pytest.raises(ValueError):
        ConfiguredFunction(launch_params, table)(torch.zeros(16), 2)


def not_positional_only(x, factor, *, BLOCK, num_warps): ...
def configured_with_default(x, factor, /, *, BLOCK, num_warps=4): ...
def var_positional(x, factor, /, *args, BLOCK, num_warps): ...
def missing_configured(x, factor, /, *, BLOCK): ...
def extra_configured(x, factor, /, *, BLOCK, num_warps, ILP): ...
def swapped_args(factor, x, /, *, BLOCK, num_warps): ...


@pytest.mark.parametrize(
    'function, error',
    [
        (not_positional_only, TypeError),
        (configured_with_default, TypeError),
        (var_positional, TypeError),
        (missing_configured, ValueError),
        (extra_configured, ValueError),
        (swapped_args, ValueError),
    ],
)
def test_signature_must_match_table(tmp_path, function, error):
    with pytest.raises(error):
        ConfiguredFunction(function, make_table(tmp_path))


def test_table_entries_configure_same_args(tmp_path):
    toml = TOML + """
  [["GPU A".configs]]
  problem_sizes = [32]
  BLOCK = 64
"""
    with pytest.raises(ValueError):
        make_table(tmp_path, toml)


def warps(x, factor, /, *, num_warps=4):
    return num_warps


def test_function_without_table_runs_with_defaults():
    configured = ConfiguredFunction(warps)
    assert configured(torch.zeros(16), 2) == 4
    assert configured(torch.zeros(16), 2, num_warps=8) == 8
    assert configured.can_be_configured(torch.zeros(16), 2)


def keyword_without_default(x, factor, /, *, num_warps): ...
def static_not_positional_only(x, factor, *, num_warps=4): ...


@pytest.mark.parametrize('function', [keyword_without_default, static_not_positional_only])
def test_signature_without_table(function):
    with pytest.raises(TypeError):
        ConfiguredFunction(function)


def test_device_name_is_asked_once(monkeypatch):
    calls = []
    monkeypatch.setattr(utils, 'get_device_name', lambda index: calls.append(index) or 'GPU A')
    utils.device_name.cache_clear()
    assert utils.device_name(0) == utils.device_name(0) == 'GPU A'
    assert calls == [0]
    utils.device_name.cache_clear()


OPS_ROOT = Path(kerops.ops.__file__).parent


OP_MODULES = sorted(
    'kerops.ops.' + '.'.join(path.relative_to(OPS_ROOT).with_suffix('').parts)
    for path in OPS_ROOT.rglob('*.py')
    if path.name != '__init__.py'
)


@pytest.mark.parametrize('module', OP_MODULES)
def test_op_module_imports(module):
    import_module(module)
