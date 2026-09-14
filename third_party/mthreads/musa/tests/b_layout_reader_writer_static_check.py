#!/usr/bin/env python3
"""Zero-GPU static audit for the B shared-layout / SQMMA-reader pair.

This encodes the facts behind `b_layout_reader_writer_audit.md`, which asks
whether the B staging-store scalarization can be removed by co-designing the B
shared layout with the SQMMA B reader instead of changing the store lowering.

It does not touch production code. It asserts:

  1. The physical transpose is real: B's global/register layout is K-contiguous
     (`#blocked1`, order [0,1], sizePerThread [8,1]) while the current SQMMA B
     shared layout is N-contiguous (`#shared1`, order [1,0]); current must
     scatter in shared, old 3.2 avoided it by sharing the K-contiguous order and
     using the SQMMA "col" reader mode.
  2. The current lowering forces the reversed order for a plain B load
     (`allowTranspose=false` -> `newOrder=[1,0]`), so the col path is only
     reachable when a `tt.trans` is present in the B load chain.
  3. Reader mode selectors and their descriptor parameters for the current
     (row) and the hypothetical candidate (col) B layouts.
  4. The current pipeline hard-rejects col operands in
     `getExpectedSqmmaOperandContract` (used by SqmmaOperandPlan /
     TritonGPUToLLVM), so switching B to col is not a local store change.
  5. No `storeMatrixShared`/stmatrix primitive exists in the current MUSA
     backend, so there is no ready-made hardware transpose-store to reuse.
  6. The existing A-only store predicate rejects both B variants.

Run:
    python3 third_party/mthreads/musa/tests/b_layout_reader_writer_static_check.py
"""

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE.with_name(name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load("a_only_vector_store_static_check")

SHAPE = (64, 32)          # B [K, N]
ELEM_BYTES = 2
LINE_BYTES = 256


def _rhs(text, alias):
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(alias + " ="):
            return s
    raise AssertionError("alias missing: " + alias)


def _ints(rhs, key):
    m = re.search(re.escape(key) + r"\s*=\s*\[([^\]]*)\]", rhs)
    assert m, (key, rhs)
    return [int(v) for v in m.group(1).split(",") if v.strip()]


def _int(rhs, key):
    m = re.search(re.escape(key) + r"\s*=\s*(\d+)", rhs)
    assert m, (key, rhs)
    return int(m.group(1))


def resolve_carrier(order, vec, per_phase, max_phase, shape, elem_bytes):
    """Mirror resolveCanonicalPH1TMESharedCarrierConfig for a rank-2 operand.

    Returns (granularity_bytes, stride_bytes, line_bytes, leading_stride_bytes,
    elems_per_group_in_leading_dim).
    """
    if vec == 1 and per_phase == 1 and max_phase == 1:
        return (None, LINE_BYTES, LINE_BYTES, None, None)
    leading_dim = order[0]
    leading_width_bytes = shape[leading_dim] * elem_bytes
    assert per_phase > 1
    assert LINE_BYTES % per_phase == 0
    assert leading_width_bytes == LINE_BYTES // per_phase, (
        leading_width_bytes, per_phase)
    sg = LINE_BYTES // max_phase
    assert sg // elem_bytes == vec, (sg, elem_bytes, vec)
    if leading_width_bytes <= LINE_BYTES:
        num_groups = 1
        elems_per_group_leading = shape[leading_dim]
    else:
        raise AssertionError("unexpected multi-group B operand")
    leading_stride_bytes = elems_per_group_leading * elem_bytes
    return (sg, LINE_BYTES, LINE_BYTES, leading_stride_bytes,
            elems_per_group_leading)


def writer_offset(order, shape, elem_bytes, coord):
    """linearizePH1TMELinearCoords single-group: offset in elements."""
    leading = coord[order[0]]
    row = coord[order[1]]
    return row * shape[order[0]] + leading


def check_physical_transpose():
    cur = (ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
           "current-up.ttgir").read_text()
    old = (ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
           "old-up.ttgir").read_text()

    b_reg = _rhs(cur, "#blocked1")
    assert _ints(b_reg, "sizePerThread") == [8, 1]
    assert _ints(b_reg, "order") == [0, 1]

    cur_b = _rhs(cur, "#shared1")
    assert "swizzled_shared" in cur_b
    assert _ints(cur_b, "order") == [1, 0]

    old_b = _rhs(old, "#shared1")
    assert "swizzled_shared" not in old_b
    assert _ints(old_b, "order") == [0, 1]
    assert _int(old_b, "vec") == 1

    # Register fast dim is dim0 (K); shared fast dim is dim1 (N): transpose.
    assert _ints(b_reg, "order")[0] != _ints(cur_b, "order")[0]
    return cur_b


def check_candidate_vs_current_descriptor():
    cur_b = _rhs((ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
                  "current-up.ttgir").read_text(), "#shared1")
    row = resolve_carrier([1, 0], _int(cur_b, "vec"), _int(cur_b, "perPhase"),
                          _int(cur_b, "maxPhase"), SHAPE, ELEM_BYTES)
    # Candidate: order [0,1], vec=8, perPhase=2, maxPhase=16 (composeMusaSharedLayout
    # for opIdx=1 non-MN-major / K-major B).
    col = resolve_carrier([0, 1], 8, 2, 16, SHAPE, ELEM_BYTES)
    assert row[0] == 32 and row[3] == 64, row
    assert col[0] == 16 and col[3] == 128, col

    # Writer offsets: current N-contiguous, candidate K-contiguous.
    assert writer_offset([1, 0], SHAPE, ELEM_BYTES, (3, 5)) == 3 * 32 + 5
    assert writer_offset([0, 1], SHAPE, ELEM_BYTES, (3, 5)) == 5 * 64 + 3

    # Loader elemsPerSwizzlingRow = physicalShape[order[0]].
    assert SHAPE[row_order_first([1, 0])] == 32
    assert SHAPE[row_order_first([0, 1])] == 64
    return row, col


def row_order_first(order):
    return order[0]


def check_reader_selectors_and_order_forcing():
    accel = (ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUTransforms/"
             "AccelerateMUSAMatmul.cpp").read_text()
    sqmma = (ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/"
             "DotOpToLLVM/SQMMA.cpp").read_text()
    mma = (ROOT / "third_party/mthreads/musa/include/TritonMUSACommon/"
           "MMAOperandUtils.h").read_text()
    plan = (ROOT / "third_party/mthreads/musa/include/TritonMUSACommon/"
            "SqmmaOperandPlan.h").read_text()

    # Plain B load => allowTranspose false => reversed order [1,0].
    assert "if (!allowTranspose)" in accel
    assert "newOrder.push_back(static_cast<unsigned>(dim));" in accel
    assert "classifySqmmaTransLoad" in accel
    assert "cur.getDefiningOp<tt::TransOp>()" in accel
    assert "allowTransposeB = transLoadKindB != SqmmaTransLoadKind::None" in accel

    # Reader selectors keyed off inferSharedRowMajor.
    assert "inferSharedRowMajor(memDescTy)" in sqmma
    assert "loaderTransB = layoutB == triton::musa::SQMMALayout::row" in sqmma
    assert "intrinsicTransB = (layoutB == triton::musa::SQMMALayout::col) ? 1 : 0" in sqmma
    assert "groupedLeadingWidthBytes" in sqmma
    assert "resolveTMESwizzleConfigFromMatrixView" in sqmma

    # Hard row-major contract recovery: col is rejected.
    assert "if (!rowMajor)" in mma
    assert "return failure();" in mma
    assert "getExpectedSqmmaOperandContract" in plan
    assert "if (failed(expected))" in plan


def check_no_stmatrix_primitive():
    musa = ROOT / "third_party/mthreads/musa"
    hits = []
    for p in musa.rglob("*"):
        if p.is_file() and p.suffix in (".cpp", ".h", ".td"):
            t = p.read_text(errors="ignore")
            if "storeMatrixShared" in t or "chooseStMatrixLayout" in t:
                hits.append(str(p.relative_to(ROOT)))
    assert not hits, hits


def check_a_predicate_rejects_b():
    b_bases = A.build_bases(SHAPE, (0, 1), (8, 1), (8, 4), (1, 4))
    assert not A.guard_accepts(b_bases, SHAPE, (1, 0), ELEM_BYTES, 8, 16, 0)
    # Candidate order [0,1] is also rejected by the A-only predicate (order
    # hard-required [1,0] and op_idx==0).
    assert not A.guard_accepts(b_bases, SHAPE, (0, 1), ELEM_BYTES, 8, 16, 0)


def main():
    check_physical_transpose()
    row, col = check_candidate_vs_current_descriptor()
    check_reader_selectors_and_order_forcing()
    check_no_stmatrix_primitive()
    check_a_predicate_rejects_b()
    print("B layout/reader co-design static audit: PASS (NO-GO for candidate)")
    print(f"  B [K,N]={SHAPE}: register K-contiguous (#blocked1 order [0,1]);")
    print("  current shared N-contiguous (#shared1 order [1,0], swizzled)")
    print("  old 3.2 shared K-contiguous (order [0,1], non-swizzled) + reader col")
    print("  current reader: layoutB=row, loaderTransB=true, intrinsicTransB=0, "
          f"desc(gran={row[0]}B,leadingStride={row[3]}B,epsw=32)")
    print("  candidate reader: layoutB=col, loaderTransB=false, intrinsicTransB=1, "
          f"desc(gran={col[0]}B,leadingStride={col[3]}B,epsw=64)")
    print("  plain B load forces order [1,0] via !allowTranspose; col only via tt.trans")
    print("  getExpectedSqmmaOperandContract rejects !rowMajor (hard failure in plan)")
    print("  no storeMatrixShared/chooseStMatrixLayout in MUSA backend")
    print("  A-only store predicate rejects both B variants")
    print("  conclusion: col-B route is plausible but untested with PH1 swizzle and "
          "multi-site; not a minimal provable patch -> NO-GO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
