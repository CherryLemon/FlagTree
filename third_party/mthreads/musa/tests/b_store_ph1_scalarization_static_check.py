#!/usr/bin/env python3
"""Zero-GPU static verifier for the M64-up B-operand shared staging store.

Result: **NO-GO for a production B shared-store vectorization.**

The old (3.2) B staging store was `store <8 x bfloat>`; the current (3.6) store
is 16 scalar `bf16` stores. This is *not* a weakening of the store lowering: the
B shared encoding changed from a plain, register-order-matching
`shared<order=[0,1]>` to `swizzled_shared<order=[1,0]>`, and the B *register*
layout `#blocked1` is contiguous along dim0 (`sizePerThread=[8,1]`, order
`[0,1]`). Under the current pairing every per-thread register run maps to shared
offsets that are `shape[1]=32` elements apart, so *no* two of the 16 registers a
thread writes are adjacent in shared memory. A `<vec x bfloat>` contiguous store
is therefore impossible without a cross-lane shuffle or a shared-layout change,
both of which are out of scope and correctness-risky. The production store path
must keep its per-element scalar loop for B.

This verifier does not modify production semantics. It proves the claim from the
tensor/memdesc encodings and asserts the production C++ still refuses to emit a
B vector store. It mirrors the reviewed layout model of
`a_only_vector_store_static_check.py` rather than re-deriving it.

Run:
    python3 third_party/mthreads/musa/tests/b_store_ph1_scalarization_static_check.py
"""

import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]


def _load_a_model():
    spec = importlib.util.spec_from_file_location(
        "a_only_vector_store_static_check",
        HERE.with_name("a_only_vector_store_static_check.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_a_model()

# ---------------------------------------------------------------------------
# Exact B contract (audited current-up.ttgir / diagnostic stage IR).
# ---------------------------------------------------------------------------
B_SHAPE = (64, 32)          # [K, N] for the up-projection B operand
B_REG_ORDER = (0, 1)        # #blocked1: dim0 fastest
B_SIZE_PER_THREAD = (8, 1)
B_THREADS_PER_WARP = (8, 4)
B_WARPS_PER_CTA = (1, 4)
B_SHARED_ORDER = (1, 0)     # #shared1: dim1 fastest (transposed vs register)
ELEM_BYTES = 2              # bfloat16
B_GRAN_BYTES = 32           # SG_32B (vec=16 * 2 bytes)
B_STRIDE_BYTES = 256
B_LINE_BYTES = 256
B_GRANULARITY_ELEMS = B_GRAN_BYTES // ELEM_BYTES  # 16
B_NUM_REGS = 16


def per_thread_offsets(bases, shape, order):
    """Return tid -> list of 16 pre-swizzle element offsets in register order."""
    out = {}
    for tid in range(A.NUM_THREADS):
        lane = tid % A.WARP_SIZE
        warp = tid // A.WARP_SIZE
        offsets = []
        for reg in range(B_NUM_REGS):
            coord = A.apply_bases(bases, reg, lane, warp)
            offsets.append(A.linearize(coord, shape, order))
        out[tid] = offsets
    return out


def min_positive_gap(values):
    values = sorted(set(values))
    return min(b - a for a, b in zip(values, values[1:]))


def check_no_contiguous_register_run():
    bases = A.build_bases(B_SHAPE, B_REG_ORDER, B_SIZE_PER_THREAD,
                          B_THREADS_PER_WARP, B_WARPS_PER_CTA)
    assert len(bases["register"]) == 4, len(bases["register"])  # 16 regs
    offsets_by_tid = per_thread_offsets(bases, B_SHAPE, B_SHARED_ORDER)

    # Every thread's 16 elements are distinct and at least 32 bytes apart; no
    # pair is adjacent (2 bytes), so the contiguous-vector predicate that gates
    # the A path is mathematically false for B.
    for tid, offsets in offsets_by_tid.items():
        assert len(offsets) == B_NUM_REGS
        assert len(set(offsets)) == B_NUM_REGS, (tid, offsets)
        gap_bytes = min_positive_gap(offsets) * ELEM_BYTES
        assert gap_bytes >= B_GRAN_BYTES, (tid, gap_bytes)
        for a in offsets:
            for b in offsets:
                assert abs(a - b) != 1, (tid, a, b)

    # The A control layout is contiguous, which is why the same predicate
    # accepts A. This ties the model to the audited A behavior.
    a_bases = A.build_bases((32, 64), (1, 0), (1, 8), (4, 8), (4, 1))
    a_offsets = per_thread_offsets(a_bases, (32, 64), (1, 0))
    for tid, offsets in a_offsets.items():
        gap_bytes = min_positive_gap(offsets) * ELEM_BYTES
        assert gap_bytes == ELEM_BYTES, (tid, gap_bytes)
    return bases, offsets_by_tid


def check_swizzle_preserves_non_adjacency(offsets_by_tid):
    """The PH1 swizzle is a sector permutation: it never makes two elements
    closer than one sector, so it cannot create a contiguous pair either."""
    for tid, offsets in offsets_by_tid.items():
        addrs = [A.swizzle(ELEM_BYTES * off, B_GRAN_BYTES, B_STRIDE_BYTES,
                           B_LINE_BYTES) for off in offsets]
        assert len(set(addrs)) == B_NUM_REGS, (tid, addrs)
        assert min_positive_gap(addrs) >= B_GRAN_BYTES, (tid, addrs)


def check_store_is_all_ph1_swizzled():
    """The B memdesc must route through `requiresAbsoluteSwizzledSharedAccess`
    => `lowerPH1SwizzledSharedStoreFromTensor`, whose default is one scalar
    store per canonical register."""
    bases = A.build_bases(B_SHAPE, B_REG_ORDER, B_SIZE_PER_THREAD,
                          B_THREADS_PER_WARP, B_WARPS_PER_CTA)
    # The A-only production predicate rejects B on multiple grounds; the first
    # is the offset contribution of register bit 0 (32, not 1).
    assert not A.guard_accepts(bases, B_SHAPE, B_SHARED_ORDER, ELEM_BYTES,
                               8, 16, 0), "A predicate unexpectedly accepts B"
    # Sanity: the A contract itself is still accepted.
    a_bases = A.build_bases((32, 64), (1, 0), (1, 8), (4, 8), (4, 1))
    assert A.guard_accepts(a_bases, (32, 64), (1, 0), ELEM_BYTES, 8, 16, 0)


# ---------------------------------------------------------------------------
# Artifact checks: the archived TTGIR encodings must match the contract.
# ---------------------------------------------------------------------------
def _alias_rhs(text, alias):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(alias + " ="):
            return line
    raise AssertionError(f"alias not found: {alias}")


def _ints(rhs, key):
    m = re.search(re.escape(key) + r"\s*=\s*\[([^\]]*)\]", rhs)
    assert m, (key, rhs)
    return [int(v.strip()) for v in m.group(1).split(",") if v.strip()]


def _int(rhs, key):
    m = re.search(re.escape(key) + r"\s*=\s*(\d+)", rhs)
    assert m, (key, rhs)
    return int(m.group(1))


def check_archived_encodings():
    cur = (ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
           "current-up.ttgir").read_text()
    old = (ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
           "old-up.ttgir").read_text()

    b_reg = _alias_rhs(cur, "#blocked1")
    assert _ints(b_reg, "sizePerThread") == list(B_SIZE_PER_THREAD)
    assert _ints(b_reg, "threadsPerWarp") == list(B_THREADS_PER_WARP)
    assert _ints(b_reg, "warpsPerCTA") == list(B_WARPS_PER_CTA)
    assert _ints(b_reg, "order") == list(B_REG_ORDER)

    b_shared = _alias_rhs(cur, "#shared1")
    assert "swizzled_shared" in b_shared, b_shared
    assert _int(b_shared, "vec") == 16
    assert _int(b_shared, "perPhase") == 4
    assert _int(b_shared, "maxPhase") == 8
    assert _ints(b_shared, "order") == list(B_SHARED_ORDER)

    # The old B shared layout had order [0,1] matching the register order and
    # was not a swizzled_shared, so it used the generic vectorizing lowerer.
    old_shared = _alias_rhs(old, "#shared1")
    assert "swizzled_shared" not in old_shared, old_shared
    assert _ints(old_shared, "order") == [0, 1], old_shared
    assert _int(old_shared, "vec") == 1, old_shared

    # B load -> local_alloc(op_idx=1) structure is present and unchanged.
    assert "tt.load %b_ptrs_85 : tensor<64x32x!tt.ptr<bf16>, #blocked1>" in cur
    assert "ttg.local_alloc %b" in cur
    assert "sqmma.op_idx = 1" in cur
    assert "tensor<64x32xbf16, #blocked1>" in cur
    return cur, old


def check_llir_store_forms():
    old = (ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
           "old-up.llir").read_text()
    cur = (ROOT / "evidence_bundle/lowering-audit/evidence/ir/"
           "current-up.llir").read_text()
    old_vec_shared = len(re.findall(
        r"store <8 x bfloat> %\w+, ptr addrspace\(3\)", old))
    cur_scalar_shared = len(re.findall(
        r"store bfloat %\w+, ptr addrspace\(3\)", cur))
    assert old_vec_shared == 4, old_vec_shared  # A + B, 2 groups each
    assert cur_scalar_shared == 32, cur_scalar_shared  # A + B, 16 each
    assert not re.search(
        r"store <8 x bfloat> %\w+, ptr addrspace\(3\)", cur), \
        "current shared store unexpectedly vectorized"


# ---------------------------------------------------------------------------
# Production source guards (no B vector branch may exist).
# ---------------------------------------------------------------------------
def check_source_plumbing():
    cpp = (ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/"
           "LoadStoreOpToLLVM.cpp").read_text()
    required = [
        "matchExactM64UpAOperandVectorContract",
        "emitExactM64UpAOperandVectorStore",
        "lowerPH1SwizzledSharedStoreFromTensor",
        "*opIdx != 0",                      # A-only gate
        "kPH1M64UpAVectorStoreMarker",
        "isCanonicalIndex",
        "targetInfo.storeDShared",          # scalar fallback store
        "if (static_cast<unsigned>(*opIdx) == 0)",  # preserved A barrier
        "targetInfo.barrier",
        "requiresAbsoluteSwizzledSharedAccess",
    ]
    for needle in required:
        assert needle in cpp, f"C++ plumbing missing: {needle}"

    # The vector emitter must be reached only through the A-only contract
    # matcher; no broad/B vector branch may exist.
    assert cpp.count("emitExactM64UpAOperandVectorStore(") == 2, \
        "expected declaration + single guarded call"
    collapsed = " ".join(cpp.split())
    guarded = (
        "if (auto vec = matchExactM64UpAOperandVectorContract(memDescTy, "
        "regTy, diagnosticOp, *access)) { if ("
        "emitExactM64UpAOperandVectorStore(loc, rewriter, *access, inVals, "
        "srcIndices, *vec, pred, targetInfo)) return success(); }"
    )
    assert guarded in collapsed, "vector emitter is not A-contract guarded"
    return cpp


def check_diagnostic_ir_optional():
    """Fresh HEAD build IR (compile-only), if present on this host. Proves the
    B global load is already vector<8xbf16> while the B shared store is still
    scalar, i.e. the scalarization is the shared staging store, not the load."""
    diag = Path("/root/moore-private-sessions/"
                "20260913-qwen36-bglobal-diagnostic-build-20260913-b7d9e2c1/"
                "diagnostic/03-after-llvmir-post-canon.mlir")
    if not diag.is_file():
        return None
    text = diag.read_text()
    b_loads = len(re.findall(
        r"llvm.load %b_\d+ : !llvm.ptr<1> -> vector<8xbf16> loc\(#loc85\)",
        text))
    b_stores = len(re.findall(
        r"llvm.store %b_\d+, %b_\d+ : bf16, !llvm.ptr<3> loc\(#loc85\)", text))
    b_vec_stores = len(re.findall(
        r"llvm.store %b_\d+, %b_\d+ : vector<\d+xbf16>, !llvm.ptr<3> "
        r"loc\(#loc85\)", text))
    assert b_loads == 2, b_loads
    assert b_stores == 16, b_stores
    assert b_vec_stores == 0, b_vec_stores
    return b_loads, b_stores


def main():
    bases, offsets = check_no_contiguous_register_run()
    check_swizzle_preserves_non_adjacency(offsets)
    check_store_is_all_ph1_swizzled()
    check_archived_encodings()
    check_llir_store_forms()
    check_source_plumbing()
    diag = check_diagnostic_ir_optional()

    sample = offsets[0]
    print("B shared-store scalarization static check: PASS (NO-GO for vectorize)")
    print(f"  B contract: shape={B_SHAPE} reg_order={B_REG_ORDER} "
          f"shared_order={B_SHARED_ORDER} elem_bytes={ELEM_BYTES}")
    print(f"  B shared: swizzled_shared vec=16 perPhase=4 maxPhase=8 "
          f"granularity={B_GRAN_BYTES}B num_regs={B_NUM_REGS}")
    print(f"  per-thread element offsets (thread 0): {sample}")
    print(f"  min per-thread byte gap = "
          f"{min_positive_gap(sample) * ELEM_BYTES}B -> no contiguous pair")
    print("  A control layout is contiguous (2B gap) and still accepted by "
          "the A predicate")
    print("  archived TTGIR: old #shared1 order=[0,1] non-swizzled; "
          "current #shared1 order=[1,0] swizzled_shared")
    print("  archived LLIR: old 4x `store <8 x bfloat>` shared; "
          "current 32x scalar `store bfloat` shared (A+B)")
    print("  C++ guard: A-only (op_idx==0), scalar fallback + barrier preserved")
    if diag:
        print(f"  fresh HEAD IR: B global load vector<8xbf16> x{diag[0]}, "
              f"B shared scalar bf16 stores x{diag[1]}, vector B stores x0")
    else:
        print("  fresh HEAD IR: not visible; skipped (optional)")
    print("  conclusion: B staging store cannot be vectorized safely without "
          "changing B shared layout or cross-lane shuffles -> NO-GO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
