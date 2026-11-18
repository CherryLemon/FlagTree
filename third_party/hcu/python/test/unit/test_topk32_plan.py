"""Check the public plan against an independent scalar ordering reference."""

import math

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")


@pytest.fixture(autouse=True)
def require_gfx938():
    if not torch.cuda.is_available():
        pytest.skip("requires a gfx938 device")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx938":
        pytest.skip("requires a gfx938 device")


def reference(x):

    def key(value, index):
        return (math.isnan(value), value if not math.isnan(value) else 0.0, -index)

    indices = torch.tensor(
        [sorted(range(len(row)), key=lambda i: key(row[i], i), reverse=True)[:32] for row in x.float().tolist()],
        dtype=torch.int64)
    return x.gather(1, indices), indices


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("width", [32, 129, 513])
def test_values_indices_and_graph_refresh(dtype, width):
    from triton.experimental.tle.ops.hcu import TopK32Plan

    host = ((torch.arange(2 * width).reshape(2, width) % 113 - 56) / 8).to(dtype)
    host[:, :6] = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0, -0.0, float("nan")], dtype=dtype)
    x = host.cuda()
    plan = TopK32Plan(x)

    def check():
        expected_values, expected_indices = reference(host)
        assert torch.equal(plan.indices.cpu(), expected_indices)
        assert torch.equal(plan.values.cpu().view(torch.int16), expected_values.view(torch.int16))

    plan()
    check()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    host = host.flip(1).contiguous()
    x.copy_(host)
    plan.values.fill_(float("nan"))
    plan.indices.fill_(-1)
    graph.replay()
    check()
