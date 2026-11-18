"""Host routing tests; the launch spies do not execute or validate GPU kernels."""

from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def plans(monkeypatch):
    state = SimpleNamespace(device=0, streams={0: "other", 1: "first"}, events=[], fail=False)

    @contextmanager
    def device_guard(device):
        previous = state.device
        state.device = device.index
        try:
            yield
        finally:
            state.device = previous

    class Tensor:

        def __init__(self, shape, dtype="bf16", device=None):
            self.shape = tuple(shape) if not isinstance(shape, int) else (shape, )
            self.dtype = dtype
            self.device = device or SimpleNamespace(type="cuda", index=1)
            self.ndim = len(self.shape)
            self.is_cuda = True

        def stride(self, index=None):
            strides = tuple(math.prod(self.shape[i + 1:]) for i in range(self.ndim))
            return strides if index is None else strides[index]

        def is_contiguous(self):
            return True

        def numel(self):
            return math.prod(self.shape)

        def element_size(self):
            return 2

        def data_ptr(self):
            return 0x100000

        def copy_(self, other):
            state.events.append(("copy", state.device, state.streams[state.device]))
            return self

    def record(stage):
        state.events.append((stage, state.device, state.streams[state.device]))
        if state.fail and stage in ("jit", "cached"):
            raise RuntimeError("launch failed")

    class CompiledKernel:

        def __getitem__(self, grid):
            record("handles")

            def launch(*args):
                record("cached")

            return launch

    class JITKernel:

        def __init__(self, fn):
            self.fn = fn

        def warmup(self, *args, **kwargs):
            record("warmup")
            return CompiledKernel()

        def __getitem__(self, grid):

            def launch(*args, **kwargs):
                record("jit")
                return CompiledKernel()

            return launch

    torch = ModuleType("torch")
    for dtype in ("float16", "bfloat16", "float32", "float8_e4m3fn", "int8", "uint32", "int64"):
        setattr(torch, dtype, "bf16" if dtype == "bfloat16" else dtype)
    torch.empty = lambda shape, **kwargs: Tensor(shape, **kwargs)
    torch.empty_like = lambda x, **kwargs: Tensor(x.shape, kwargs.get("dtype", x.dtype), x.device)
    torch.mean = lambda *args, **kwargs: record("mean")
    torch.cuda = SimpleNamespace(device=device_guard)
    monkeypatch.setitem(sys.modules, "torch", torch)

    modules = {}
    for name in ("triton", "triton.language", "triton.experimental", "triton.experimental.tle",
                 "triton.experimental.tle.language"):
        modules[name] = ModuleType(name)
        modules[name].__path__ = []
        monkeypatch.setitem(sys.modules, name, modules[name])
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, modules[name])
    triton = modules["triton"]
    triton.jit = JITKernel
    triton.cdiv = lambda x, y: (x + y - 1) // y
    triton.next_power_of_2 = lambda x: 1 << (x - 1).bit_length()
    modules["triton.language"].constexpr = object()

    root = Path(__file__).resolve().parents[5]
    path = root / "python/triton/experimental/tle/ops/hcu/__init__.py"
    name = "_hcu_plan_device_test"
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    try:
        yield module, Tensor, state
    finally:
        for key in tuple(sys.modules):
            if key.startswith(name + "."):
                del sys.modules[key]


def make_plan(module, tensor, name):
    if name == "small_m":
        return module.SmallMBF16GemmPlan(tensor((2, 64)), tensor((3, 64)), split_k=2)
    if name == "quant":
        return module.BlockFP8QuantPlan(tensor((2, 129)))
    if name == "topk":
        return module.TopK32Plan(tensor((2, 513)))
    if name == "single_row":
        return module.SingleRowGemmPlan(tensor((1, 64)), tensor((3, 64)))
    dims = {
        "dense_generic": (8, 129, 1, 1, 32), "dense_default": (8, 129, 1, 1, 128), "dense_tail": (8, 129, 2, 2, 96),
        "dense_large_d": (8, 4096, 1, 1, 256), "dense_gqa": (513, 129, 2, 1, 128), "sage": (8, 129, 2, 1, 96),
        "sage_long": (8, 16384, 2, 1, 128), "sage_wide": (8, 129, 2, 1, 256)
    }
    nq, nk, hq, hk, d = dims[name]
    cls = module.SageAttentionPlan if name.startswith("sage") else module.DenseFlashAttentionPlan
    return cls(tensor((1, nq, hq, d)), tensor((1, nk, hk, d)), tensor((1, nk, hk, d)))


@pytest.mark.parametrize("name", [
    "small_m", "quant", "topk", "single_row", "dense_generic", "dense_default", "dense_tail", "dense_large_d",
    "dense_gqa", "sage", "sage_long", "sage_wide"
])
def test_launches_follow_tensor_device_and_current_stream(plans, name):
    module, tensor, state = plans
    plan = make_plan(module, tensor, name)
    plan()
    assert state.events
    assert all(device == 1 and stream == "first" for _, device, stream in state.events)
    assert state.device == 0

    state.events.clear()
    state.streams[1] = "second"
    plan()
    assert state.events
    assert all(device == 1 and stream == "second" for _, device, stream in state.events)
    assert state.device == 0


@pytest.mark.parametrize("name,method", [("dense_default", "pack"), ("dense_default", "core"), ("sage", "quantize"),
                                         ("sage", "pack"), ("sage", "core"), ("sage", "full_cast")])
def test_direct_attention_stages_use_tensor_device(plans, name, method):
    module, tensor, state = plans
    plan = make_plan(module, tensor, name)
    getattr(plan, method)()
    assert state.events
    assert all(device == 1 for _, device, _ in state.events)
    assert state.device == 0


@pytest.mark.parametrize("name", ["single_row", "dense_default", "sage"])
def test_failed_launch_restores_callers_device(plans, name):
    module, tensor, state = plans
    plan = make_plan(module, tensor, name)
    state.fail = True
    with pytest.raises(RuntimeError, match="launch failed"):
        plan()
    assert state.device == 0
