"""Numerical and graph checks for plans called with another current device."""

import math

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")


@pytest.fixture
def two_gfx938_devices():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two gfx938 devices")
    for index in (0, 1):
        with torch.cuda.device(index):
            target = triton.runtime.driver.active.get_current_target()
            if target.backend != "hip" or target.arch != "gfx938":
                pytest.skip("requires two gfx938 devices")
    with torch.cuda.device(0):
        yield torch.device("cuda", 1)


def make_plan(kind, device):
    from triton.experimental.tle.ops.hcu import DenseFlashAttentionPlan, SingleRowGemmPlan

    if kind == "gemv":
        a = torch.randn((1, 64), device=device, dtype=torch.float16)
        b = torch.randn((37, 64), device=device, dtype=torch.float16)
        plan = SingleRowGemmPlan(a, b)
        reference = lambda: (a.double() @ b.double().T).to(a.dtype)
        return plan, a, reference
    q = torch.randn((1, 17, 1, 128), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 129, 1, 128), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    plan = DenseFlashAttentionPlan(q, k, v)

    def reference():
        scores = q.double().transpose(1, 2) @ k.double().transpose(1, 2).transpose(-1, -2)
        result = (scores / math.sqrt(128)).softmax(-1) @ v.double().transpose(1, 2)
        return result.transpose(1, 2).to(q.dtype)

    return plan, q, reference


@pytest.mark.parametrize("kind", ["gemv", "cached_attention"])
def test_plan_on_noncurrent_device(two_gfx938_devices, kind):
    device = two_gfx938_devices
    plan, _, reference = make_plan(kind, device)
    result = plan()
    torch.cuda.synchronize(device)
    assert torch.cuda.current_device() == 0
    assert result.device == device
    torch.testing.assert_close(result, reference(), atol=0.02, rtol=0.02)


@pytest.mark.parametrize("kind", ["gemv", "cached_attention"])
def test_graph_replay_refreshes_inputs_on_owning_device(two_gfx938_devices, kind):
    device = two_gfx938_devices
    plan, refresh, reference = make_plan(kind, device)
    plan()
    torch.cuda.synchronize(device)
    with torch.cuda.device(device):
        stream = torch.cuda.Stream(device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with torch.cuda.device(0):
                plan()
        refresh.neg_()
        expected = reference()
        plan.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize(device)
    assert torch.cuda.current_device() == 0
    assert torch.isfinite(plan.output).all()
    torch.testing.assert_close(plan.output, expected, atol=0.02, rtol=0.02)
