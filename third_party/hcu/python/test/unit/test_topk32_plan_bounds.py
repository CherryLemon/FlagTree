"""Metadata-only range tests; large tensors are never allocated."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def constructor(monkeypatch):
    allocations = []
    torch = ModuleType("torch")
    torch.float16, torch.bfloat16, torch.uint32, torch.int64 = "fp16", "bf16", "u32", "i64"

    def empty(shape, **kwargs):
        allocations.append((shape, kwargs))
        return object()

    torch.empty = empty
    triton = ModuleType("triton")
    triton.__path__ = []
    triton.jit = lambda fn: fn
    triton.cdiv = lambda x, y: (x + y - 1) // y
    tl = ModuleType("triton.language")
    tl.constexpr = object()
    triton.language = tl
    for name, module in (("torch", torch), ("triton", triton), ("triton.language", tl)):
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[5]
    path = root / "python/triton/experimental/tle/ops/hcu/topk32.py"
    package_name = "_topk32_bounds_test"
    package = ModuleType(package_name)
    package.__path__ = [str(path.parent)]
    monkeypatch.setitem(sys.modules, package_name, package)
    spec = importlib.util.spec_from_file_location(package_name + ".topk32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module.TopK32Plan, allocations
    finally:
        for key in tuple(sys.modules):
            if key.startswith(package_name + "."):
                del sys.modules[key]


def tensor(rows, width):
    return SimpleNamespace(shape=(rows, width), ndim=2, dtype="bf16", device=SimpleNamespace(type="cuda"),
                           is_contiguous=lambda: True, element_size=lambda: 2)


@pytest.mark.parametrize("rows,width,buffer", [(32770, 65535, "input"), (16385, 65535, "input"),
                                               (8388608, 32, "indices")])
def test_oversize_plan_fails_before_allocation(constructor, rows, width, buffer):
    plan, allocations = constructor
    with pytest.raises(ValueError, match=f"{buffer} exceeds the validated buffer extent"):
        plan(tensor(rows, width))
    assert allocations == []


@pytest.mark.parametrize("rows,width", [(16384, 65535), (8388607, 32), (7, 20480), (2, 129)])
def test_valid_extent_boundaries_remain_accepted(constructor, rows, width):
    plan, allocations = constructor
    result = plan(tensor(rows, width))
    assert result.rows == rows
    assert result.width == width
    assert len(allocations) == 3


@pytest.mark.parametrize("options", [{"tile": 64}, {"waves": 1}])
def test_invalid_launch_options_still_fail_before_allocation(constructor, options):
    plan, allocations = constructor
    with pytest.raises(ValueError, match="unsupported selection"):
        plan(tensor(2, 129), **options)
    assert allocations == []
