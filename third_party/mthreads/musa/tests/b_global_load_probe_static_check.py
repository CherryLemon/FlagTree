#!/usr/bin/env python3
"""Zero-GPU / zero-compiler static verifier for the opt-in B global-load probe.

This mirrors, in pure Python, the exact-contract guard implemented in
`third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/LoadStoreOpToLLVM.cpp`
(`probeExactM64UpBGlobalLoad`) and the backend opt-in plumbing in
`third_party/mthreads/backend/compiler.py` (`MUSAOptions.b_global_load_probe`).

It proves:
  * the exact audited contract parsed from the archived current TTGIR/JSON
    (musa:31, num_ctas=1, num_stages=1, BF16, `[64,32]`, `#blocked1`, unmasked)
    is accepted by the matcher model;
  * every key near-miss is rejected (marker off, wrong target / num_ctas /
    num_stages, masked, wrong dtype / shape / layout, and the real A load);
  * the source plumbing is opt-in and default-off, and the probe cannot change
    `vec` (source scan: void-returning probe + discarded call result + no `vec`
    assignment inside the probe body).

No compiler, GPU, network, container, or path outside this worktree is used.

Run:
    python3 third_party/mthreads/musa/tests/b_global_load_probe_static_check.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
IR = REPO_ROOT / "evidence_bundle/lowering-audit/evidence/ir"
CURRENT_TTGIR = IR / "current-up.ttgir"
CURRENT_JSON = IR / "current-up.json"
CPP = (REPO_ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/"
               "LoadStoreOpToLLVM.cpp")
COMPILER = REPO_ROOT / "third_party/mthreads/backend/compiler.py"

# Exact contract constants (from the audited current M64-up TTGIR).
EXACT_TARGET = "musa:31"
EXACT_NUM_CTAS = 1
EXACT_NUM_STAGES = 1
EXACT_SHAPE = (64, 32)
EXACT_DTYPE = "bf16"
EXACT_SIZE_PER_THREAD = (8, 1)
EXACT_THREADS_PER_WARP = (8, 4)
EXACT_WARPS_PER_CTA = (1, 4)
EXACT_ORDER = (0, 1)

BLOCKED_RE = re.compile(
    r"^#(\w+)\s*=\s*#(?:ttg|triton_gpu)\.blocked<\{"
    r"sizePerThread = \[([^\]]*)\], "
    r"threadsPerWarp = \[([^\]]*)\], "
    r"warpsPerCTA = \[([^\]]*)\], "
    r"order = \[([^\]]*)\]\}>", re.M)
ANY_2D_LOAD_RE = re.compile(
    r"tt\.load ([^:]*?): tensor<(\d+)x(\d+)x!tt\.ptr<(\w+)>, #(\w+)>")
TARGET_RE = re.compile(r'ttg\.target = "([^"]+)"')
NUM_CTAS_RE = re.compile(r'"ttg\.num-ctas" = (\d+) : i32')


def _ints(text):
    return tuple(int(x) for x in text.replace(" ", "").split(",") if x)


# ---------------------------------------------------------------------------
# Pure-Python mirror of `probeExactM64UpBGlobalLoad`.
# ---------------------------------------------------------------------------
def probe_matches(contract):
    """Return True iff the C++ probe would fire for `contract`.

    `contract` keys mirror the C++ negative checks one-for-one. Every check
    precedes the AxisInfo query in C++, so a False here means the probe returns
    without touching the analysis or printing anything.
    """
    if not contract["marker"]:
        return False
    if contract["target"] != EXACT_TARGET:
        return False
    if contract["num_ctas"] != EXACT_NUM_CTAS:
        return False
    if contract["num_stages"] != EXACT_NUM_STAGES:
        return False
    if contract["masked"]:
        return False
    if not contract["is_bf16"]:
        return False
    if tuple(contract["shape"]) != EXACT_SHAPE:
        return False
    if tuple(contract["size_per_thread"]) != EXACT_SIZE_PER_THREAD:
        return False
    if tuple(contract["threads_per_warp"]) != EXACT_THREADS_PER_WARP:
        return False
    if tuple(contract["warps_per_cta"]) != EXACT_WARPS_PER_CTA:
        return False
    if tuple(contract["order"]) != EXACT_ORDER:
        return False
    return True


# ---------------------------------------------------------------------------
# Parse the archived real contract.
# ---------------------------------------------------------------------------
def parse_layouts(ttgir):
    layouts = {}
    for m in BLOCKED_RE.finditer(ttgir):
        layouts[m.group(1)] = {
            "size_per_thread": _ints(m.group(2)),
            "threads_per_warp": _ints(m.group(3)),
            "warps_per_cta": _ints(m.group(4)),
            "order": _ints(m.group(5)),
        }
    return layouts


def parse_2d_loads(ttgir):
    loads = []
    for m in ANY_2D_LOAD_RE.finditer(ttgir):
        args = m.group(1)
        loads.append({
            "shape": (int(m.group(2)), int(m.group(3))),
            "dtype": m.group(4),
            "layout": m.group(5),
            "masked": "," in args,
        })
    return loads


def parse_contract(ttgir, meta):
    target = TARGET_RE.search(ttgir)
    num_ctas = NUM_CTAS_RE.search(ttgir)
    layouts = parse_layouts(ttgir)
    b_load = None
    a_load = None
    for load in parse_2d_loads(ttgir):
        if load["shape"] == EXACT_SHAPE and load["dtype"] == EXACT_DTYPE:
            b_load = load
        if load["shape"] == (32, 64) and load["dtype"] == EXACT_DTYPE:
            a_load = load
    assert b_load is not None, "exact B load not found in archived TTGIR"
    assert target is not None and num_ctas is not None, "module attrs missing"
    layout = layouts[b_load["layout"]]
    contract = {
        "marker": True,
        "target": target.group(1),
        "num_ctas": int(num_ctas.group(1)),
        "num_stages": int(meta["num_stages"]),
        "masked": b_load["masked"],
        "is_bf16": b_load["dtype"] == "bf16",
        "shape": b_load["shape"],
        "size_per_thread": layout["size_per_thread"],
        "threads_per_warp": layout["threads_per_warp"],
        "warps_per_cta": layout["warps_per_cta"],
        "order": layout["order"],
    }
    return contract, a_load, layouts


# ---------------------------------------------------------------------------
# Source-level plumbing / non-invasiveness scan.
# ---------------------------------------------------------------------------
def probe_body(cpp):
    start = cpp.index("static void probeExactM64UpBGlobalLoad(")
    end = cpp.index("struct LoadOpConversion :", start)
    return cpp[start:end]


def check_source_plumbing():
    cpp = CPP.read_text()
    comp = COMPILER.read_text()
    body = probe_body(cpp)

    cpp_required = [
        "kMusaBGlobalLoadProbeMarker",
        '"musa.b_global_load_probe"',
        "kMusaBGlobalLoadProbeNumStages",
        "kMusaBGlobalLoadProbeTarget",
        '"musa:31"',
        "triton::gpu::AttrTargetName",
        "triton::gpu::AttrNumCTAsName",
        # default-off guard precedes the first AxisInfo query
        "if (!marker || !marker.getValue())",
        # exact-contract negative checks
        "if (op.getMask())",
        "!isa<BFloat16Type>(tensorTy.getElementType())",
        "ph1EqualsIntList(tensorTy.getShape(), {64, 32})",
        "ph1EqualsUnsignedList(blocked.getSizePerThread(), {8u, 1u})",
        "ph1EqualsUnsignedList(blocked.getThreadsPerWarp(), {8u, 4u})",
        "ph1EqualsUnsignedList(blocked.getWarpsPerCTA(), {1u, 4u})",
        "ph1EqualsUnsignedList(blocked.getOrder(), {0u, 1u})",
        # recorded values
        "axisInfoAnalysis.getContiguity(ptr)",
        "axisInfoAnalysis.getAlignment(ptr)",
        "finalVec",
        "dumpProbePointerDefinitionChain",
    ]
    comp_required = [
        "b_global_load_probe: bool = False",
        'hash_dict.pop("b_global_load_probe"',
        'mod.set_attr("musa.b_global_load_probe"',
        'mod.set_attr("musa.b_global_load_probe.num_stages"',
    ]
    for needle in cpp_required:
        assert needle in cpp, f"C++ probe plumbing missing: {needle}"
    for needle in comp_required:
        assert needle in comp, f"compiler opt-in plumbing missing: {needle}"

    # Non-invasiveness: void probe, discarded call result, no vec write inside.
    assert "static void probeExactM64UpBGlobalLoad(" in cpp
    assert "(void)probeExactM64UpBGlobalLoad(op, ptr, axisInfoAnalysis, vec);" in cpp
    for forbidden in ("vec =", "vec +=", "vec -=", "vec /=", "vec *="):
        assert forbidden not in body, f"probe body mutates vec via {forbidden!r}"

    # Default-off: the option must never default to True and the C++ marker must
    # never be inferred from anything but the module attribute.
    assert "b_global_load_probe: bool = True" not in comp
    assert "getenv" not in body, "probe must not branch on environment variables"
    return cpp, comp, body


def main():
    meta = json.loads(CURRENT_JSON.read_text())
    ttgir = CURRENT_TTGIR.read_text()
    contract, a_load, layouts = parse_contract(ttgir, meta)

    # The real archived B load must be the exact positive match.
    assert contract["target"] == EXACT_TARGET, contract
    assert contract["num_ctas"] == EXACT_NUM_CTAS, contract
    assert contract["num_stages"] == EXACT_NUM_STAGES, contract
    assert contract["masked"] is False, "exact B load must be unmasked"
    assert probe_matches(contract), f"exact B contract rejected: {contract}"

    # Real negative: the A load shares dtype but is masked and has a shape /
    # layout that must not match, so the probe can never fire on A.
    assert a_load is not None, "A load not found"
    a_layout = layouts[a_load["layout"]]
    a_contract = dict(contract)
    a_contract.update(shape=a_load["shape"], masked=a_load["masked"],
                      size_per_thread=a_layout["size_per_thread"],
                      threads_per_warp=a_layout["threads_per_warp"],
                      warps_per_cta=a_layout["warps_per_cta"],
                      order=a_layout["order"])
    assert a_load["masked"] is True, "A load is expected to be masked"
    assert not probe_matches(a_contract), "A load unexpectedly matched"

    check_source_plumbing()

    negatives = {
        "marker off (default)": {"marker": False},
        "wrong target": {"target": "musa:32"},
        "num_ctas=2": {"num_ctas": 2},
        "num_stages=2": {"num_stages": 2},
        "masked": {"masked": True},
        "non-bf16 dtype": {"is_bf16": False},
        "wrong shape": {"shape": (32, 64)},
        "wrong sizePerThread": {"size_per_thread": (1, 8)},
        "wrong threadsPerWarp": {"threads_per_warp": (4, 8)},
        "wrong warpsPerCTA": {"warps_per_cta": (4, 1)},
        "wrong order": {"order": (1, 0)},
    }
    for name, patch in negatives.items():
        mutated = dict(contract)
        mutated.update(patch)
        assert not probe_matches(mutated), f"negative unexpectedly accepted: {name}"

    print("B global-load probe static check: PASS")
    print(f"  exact contract: target={contract['target']} "
          f"num_ctas={contract['num_ctas']} num_stages={contract['num_stages']} "
          f"shape={contract['shape']} dtype={EXACT_DTYPE} unmasked")
    print("  layout: sizePerThread=[8, 1] threadsPerWarp=[8, 4] "
          "warpsPerCTA=[1, 4] order=[0, 1]")
    print("  source: opt-in default-off (MUSAOptions.b_global_load_probe=False), "
          "void probe, discarded call result, no vec write")
    print("  negative cases rejected: " + ", ".join(sorted(negatives)) +
          ", real A load")
    return 0


if __name__ == "__main__":
    sys.exit(main())
