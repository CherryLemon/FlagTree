#!/usr/bin/env python3
"""Zero-GPU static verifier for the exact M64-up SQMMA A-operand vector store.

This mirrors, in pure Python, the guard implemented in
`third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/LoadStoreOpToLLVM.cpp`
(`matchExactM64UpAOperandVectorContract` / `provePH1ContiguousRegisterVectorGroups`)
and the original PH1 swizzle in `TritonMUSACommon/TMEUtils.h`.

It proves, for the audited M64-up A operand contract only:
  * the register layout maps each 8-register run to `base + {0..7}` with
    `base % 8 == 0` for every (lane, warp), and
  * grouping the 8 scalar BF16 stores in a run into one `<8 x bfloat>` store is
    byte-for-byte identical and covers the A shared region exactly once.

It also asserts that the same predicate rejects near-miss contracts (B operand,
wrong order/vec/granularity, unaligned allocation offset).

No compiler, GPU, network, or private path is used. Run:
    python3 third_party/mthreads/musa/tests/a_only_vector_store_static_check.py
"""

import itertools
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Exact contract constants (from the audited current M64-up TTGIR).
# ---------------------------------------------------------------------------
SHAPE = (32, 64)           # A tensor / memdesc shape
ORDER = (1, 0)             # shared/register order, last dim fastest
SIZE_PER_THREAD = (1, 8)
THREADS_PER_WARP = (4, 8)
WARPS_PER_CTA = (4, 1)
WARP_SIZE = 32
NUM_WARPS = 4
NUM_THREADS = WARP_SIZE * NUM_WARPS
ELEM_BYTES = 2             # bfloat16
GRANULARITY_BYTES = 16     # resolved PH1 swizzle granularity (SG_16B)
VEC = GRANULARITY_BYTES // ELEM_BYTES  # 8
SHARED_VEC = 8
SHARED_PER_PHASE = 2
SHARED_MAX_PHASE = 16
STRIDE_BYTES = 256
LINE_BYTES = 256


# ---------------------------------------------------------------------------
# Small blocked-layout model: basis XOR, exactly like LinearLayout.apply.
# ---------------------------------------------------------------------------
def _xor_merge(a, b):
    return tuple(x ^ y for x, y in zip(a, b))


def build_bases(shape, order, size_per_thread, threads_per_warp, warps_per_cta):
    """Return {in_dim: [(d0, d1), ...]} bases in final [dim0, dim1] order.

    Follows Triton's `identityStandardND` composition: each successive layout is
    shifted by the accumulated out-dim size of the previous one, then
    `ensureLayoutNotSmallerThan` appends register bases that cover the residual
    shape in each out dimension.
    """
    out_size = [1, 1]
    bases = {"register": [], "lane": [], "warp": []}

    def add(in_dim, dim_sizes):
        for dim in order:
            size = dim_sizes[dim]
            assert size >= 1
            while size > 1:
                assert size % 2 == 0, size
                coord = [0, 0]
                coord[dim] = out_size[dim]
                bases[in_dim].append(tuple(coord))
                out_size[dim] *= 2
                size //= 2

    add("register", list(size_per_thread))
    add("lane", list(threads_per_warp))
    add("warp", list(warps_per_cta))

    # `ensureLayoutNotSmallerThan`: add register bits for the residual shape.
    for dim in range(2):
        actual = out_size[dim]
        desired = shape[dim]
        if actual < desired:
            assert desired % actual == 0, (actual, desired)
            count = desired // actual
            assert count & (count - 1) == 0, count
            while count > 1:
                coord = [0, 0]
                coord[dim] = out_size[dim]
                bases["register"].append(tuple(coord))
                out_size[dim] *= 2
                count //= 2
        assert out_size[dim] == desired, (out_size[dim], desired)
    return bases


def apply_bases(bases, reg, lane, warp):
    coord = (0, 0)
    for bit in range(len(bases["register"])):
        if reg & (1 << bit):
            coord = _xor_merge(coord, bases["register"][bit])
    for bit in range(len(bases["lane"])):
        if lane & (1 << bit):
            coord = _xor_merge(coord, bases["lane"][bit])
    for bit in range(len(bases["warp"])):
        if warp & (1 << bit):
            coord = _xor_merge(coord, bases["warp"][bit])
    return coord


def linearize(coord, shape, order):
    """Mirror linearizePH1TMELinearCoords for a single leading-dim group."""
    return coord[order[1]] * shape[order[0]] + coord[order[0]]


# ---------------------------------------------------------------------------
# Guard mirror.
# ---------------------------------------------------------------------------
def guard_accepts(bases, shape, order, elem_bytes, vec, granularity_bytes,
                  alloc_offset, marker_enabled=True, target="musa:31",
                  num_ctas=1, num_stages=1, free_reg_mask=0,
                  ctas_per_cga=(1, 1)):
    # Explicit opt-in marker (no service caller sets it). Not inferable from
    # shape/op_idx/source/function name/IR hash/runtime environment.
    if not marker_enabled:
        return False
    # Target capability 31, CTA/CGA size 1, single pipeline stage.
    if target != "musa:31":
        return False
    if num_ctas != 1:
        return False
    if num_stages != 1:
        return False
    if any(c != 1 for c in ctas_per_cga):
        return False
    # Exact register contract: 16 distinct registers, zero free mask.
    if free_reg_mask != 0:
        return False
    if vec < 2 or (vec & (vec - 1)) != 0:
        return False
    if len(shape) != 2 or len(order) != 2:
        return False
    if order[0] != 1 or order[1] != 0:
        return False
    if shape[order[0]] * elem_bytes > 256:  # single leading-dim group
        return False
    if shape[1] <= 0:
        return False
    if granularity_bytes != 16:
        return False
    if elem_bytes == 0 or granularity_bytes % elem_bytes != 0:
        return False
    if granularity_bytes // elem_bytes != vec:
        return False
    if alloc_offset % granularity_bytes != 0:
        return False

    log2vec = vec.bit_length() - 1

    def offset_of(coords):
        c0, c1 = coords
        if c0 < 0 or c1 < 0:
            return None
        if c0 and c1:
            return None
        if c0 and (c0 & (c0 - 1)):
            return None
        if c1 and (c1 & (c1 - 1)):
            return None
        return c0 * shape[1] + c1

    reg_bases = bases["register"]
    if len(reg_bases) < log2vec:
        return False
    for i in range(log2vec):
        if offset_of(reg_bases[i]) != (1 << i):
            return False

    def introduces_low_bits(coords):
        off = offset_of(coords)
        if off is None:
            return True
        if off == 0:
            return False
        return (off & (vec - 1)) != 0

    for coords in reg_bases[log2vec:]:
        if introduces_low_bits(coords):
            return False
    for coords in bases["lane"]:
        if introduces_low_bits(coords):
            return False
    for coords in bases["warp"]:
        if introduces_low_bits(coords):
            return False

    num_regs = 1 << len(reg_bases)
    if num_regs != 16:
        return False
    return num_regs % vec == 0


# ---------------------------------------------------------------------------
# PH1 absolute-address swizzle (mirror of applyPH1TMESwizzleToByteAddressValue).
# ---------------------------------------------------------------------------
def swizzle(addr_bytes, gran=GRANULARITY_BYTES, stride=STRIDE_BYTES,
            line=LINE_BYTES):
    line_offset = addr_bytes % line
    line_id = addr_bytes // line
    swizzle_group = stride // gran
    swizzle_line_id = line_id % swizzle_group
    sector_in_line = line_offset // gran
    offset_in_sector = line_offset % gran
    return line_id * line + ((sector_in_line ^ swizzle_line_id) * gran) + \
        offset_in_sector


def final_relative_byte(offset_elems, alloc_offset):
    # getPH1SwizzledSharedElementPtr applies the swizzle to the absolute byte
    # address (offset + allocation offset) and subtracts the allocation offset
    # back, so the relative device byte is swizzle(alloc + 2*offset) anchored at
    # the raw shared base.
    return swizzle(alloc_offset + ELEM_BYTES * offset_elems)


# ---------------------------------------------------------------------------
# Byte-equivalence proof for one allocation offset.
# ---------------------------------------------------------------------------
def check_byte_equivalence(bases, alloc_offset, num_regs):
    scalar = {}
    vector = {}
    seen = {}
    for tid in range(NUM_THREADS):
        lane = tid % WARP_SIZE
        warp = tid // WARP_SIZE
        for g in range(num_regs // VEC):
            first = g * VEC
            base_off = linearize(apply_bases(bases, first, lane, warp), SHAPE,
                                 ORDER)
            assert base_off % VEC == 0, (tid, g, base_off)
            group_bytes = final_relative_byte(base_off, alloc_offset)
            assert group_bytes % GRANULARITY_BYTES == 0, (tid, g, group_bytes)
            values = [((tid * 16 + g * 8 + i) * 31 + 7) & 0xFFFF
                      for i in range(VEC)]
            for r in range(VEC):
                off = linearize(apply_bases(bases, first + r, lane, warp), SHAPE,
                                ORDER)
                addr = final_relative_byte(off, alloc_offset)
                assert addr == group_bytes + ELEM_BYTES * r, (tid, g, r, addr)
                for byte, val in enumerate(values[r].to_bytes(2, "little")):
                    scalar[addr + byte] = val
                    seen[addr + byte] = seen.get(addr + byte, 0) + 1
            payload = b"".join(v.to_bytes(2, "little") for v in values)
            for i, byte in enumerate(payload):
                vector[group_bytes + i] = byte
    assert scalar == vector, "scalar and vector byte images differ"
    total = SHAPE[0] * SHAPE[1] * ELEM_BYTES
    assert len(scalar) == total, (len(scalar), total)
    assert set(seen.values()) == {1}, "A region written more than once"
    if alloc_offset == 0:
        # The audited allocation starts at the raw shared base, so the swizzled
        # image is exactly [0, 4096) and matches the archived proof.
        assert sorted(scalar) == list(range(total)), "A region not fully covered"
    return total


def check_against_audited_llir_formula(bases):
    """The audited current M64-up LLIR (commit 552b8fdf) reduces both A group
    base addresses to
        base = (row_half << 8) + 2 * (mask ^ (row_half << 3))
        mask = ((row << 6) & 64) | ((tid << 3) & 56), row = (tid >> 3) & 15
    and `row_half = (row | (group << 4)) >> 1`. Verify the layout + swizzle model
    reproduces those exact bytes for all 128 threads, tying this verifier to the
    archived artifact rather than only to itself.
    """
    for tid in range(NUM_THREADS):
        row = (tid >> 3) & 15
        k_lane = (tid << 3) & 56
        mask = ((row << 6) & 64) | k_lane
        lane = tid % WARP_SIZE
        warp = tid // WARP_SIZE
        for group, logical_row in enumerate([row, row | 16]):
            row_half = logical_row >> 1
            expected = (row_half << 8) + 2 * (mask ^ (row_half << 3))
            off = linearize(apply_bases(bases, group * VEC, lane, warp), SHAPE,
                            ORDER)
            assert final_relative_byte(off, 0) == expected, (tid, group)


def check_source_plumbing():
    """Static source-level cross-check. The host has no MLIR/LLVM build, so the
    C++ cannot be compiled; this asserts the marker/guard/ctx plumbing actually
    exists in the two edited sources.

    Cache-key semantics are three separate layers and must not be conflated:
    (1) `MUSAOptions.hash()` for False drops the field and equals the old method
    result; (2) True yields a distinct options hash; (3) the full Triton key
    also hashes `backends/mthreads/compiler.py` and `libtriton`, so it rotates
    whenever those change regardless of the option. The source must not claim
    the full key is unchanged."""
    root = Path(__file__).resolve().parents[4]
    cpp = (root / "third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/"
                   "LoadStoreOpToLLVM.cpp").read_text()
    comp = (root / "third_party/mthreads/backend/compiler.py").read_text()
    cpp_required = [
        "kPH1M64UpAVectorStoreMarker",
        '"musa.ph1_m64up_a_vector_store"',
        "kPH1M64UpAVectorStoreTarget",
        '"musa:31"',
        "triton::gpu::AttrTargetName",
        "triton::gpu::AttrNumCTAsName",
        "triton::gpu::getCTAsPerCGA",
        'str_attr("register")',
        "MLIRContext *ctx = regTy.getContext();",
        "freeRegMask != 0",
        "numRegs != 16",
    ]
    comp_required = [
        "ph1_m64up_a_vector_store: bool = False",
        'hash_dict.pop("ph1_m64up_a_vector_store"',
        'mod.set_attr("musa.ph1_m64up_a_vector_store"',
        'mod.set_attr("musa.ph1_m64up_a_vector_store.num_stages"',
        "get_bool_attr(True)",
        "get_int32_attr(int(opt.num_stages))",
        "does NOT keep the full Triton cache key fixed",
    ]
    comp_forbidden = [
        # the pre-r2 misleading claim: removing the field keeps the full key
        "byte-identical to the previous",
    ]
    for needle in cpp_required:
        assert needle in cpp, f"C++ plumbing missing: {needle}"
    for needle in comp_required:
        assert needle in comp, f"compiler plumbing missing: {needle}"
    for needle in comp_forbidden:
        assert needle not in comp, f"misleading cache wording present: {needle}"
    return root


def main():
    bases = build_bases(SHAPE, ORDER, SIZE_PER_THREAD, THREADS_PER_WARP,
                        WARPS_PER_CTA)
    assert guard_accepts(bases, SHAPE, ORDER, ELEM_BYTES, VEC,
                         GRANULARITY_BYTES, 0), "A contract rejected"
    assert guard_accepts(bases, SHAPE, ORDER, ELEM_BYTES, VEC,
                         GRANULARITY_BYTES, 16), "aligned offset rejected"
    num_regs = 1 << len(bases["register"])
    assert num_regs == 16, num_regs

    check_source_plumbing()
    check_against_audited_llir_formula(bases)

    checked = []
    for alloc_offset in (0, 16, 32):
        total = check_byte_equivalence(bases, alloc_offset, num_regs)
        checked.append((alloc_offset, total))

    # Negative cases: each must be rejected by the same predicate. The first is
    # the Astra down-projection blocker: an identical A-projection layout is
    # still rejected unless the explicit opt-in marker is present.
    b_bases = build_bases((64, 32), (0, 1), (8, 1), (8, 4), (1, 4))
    negatives = {
        "down/identical layout with marker unset": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 0,
            marker_enabled=False),
        "wrong target capability": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 0,
            target="musa:32"),
        "num_ctas=2": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 0,
            num_ctas=2),
        "num_stages=2": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 0,
            num_stages=2),
        "free register mask nonzero": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 0,
            free_reg_mask=1),
        "ctas_per_cga=2": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 0,
            ctas_per_cga=(2, 1)),
        "B operand register order": guard_accepts(
            b_bases, (64, 32), (0, 1), ELEM_BYTES, VEC, GRANULARITY_BYTES, 0),
        "wrong order": guard_accepts(
            bases, SHAPE, (0, 1), ELEM_BYTES, VEC, GRANULARITY_BYTES, 0),
        "wrong vec": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, 16, 32, 0),
        "wrong granularity": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, 32, 0),
        "unaligned allocation": guard_accepts(
            bases, SHAPE, ORDER, ELEM_BYTES, VEC, GRANULARITY_BYTES, 2),
    }
    for name, accepted in negatives.items():
        assert not accepted, f"negative case unexpectedly accepted: {name}"

    print("A-only vector store static check: PASS")
    print(f"  contract: shape={SHAPE} vec={VEC} bytes={GRANULARITY_BYTES} "
          f"num_regs={num_regs}")
    print("  marker: musa.ph1_m64up_a_vector_store (default unset); "
          "target musa:31; num_ctas=1; num_stages=1; free_reg_mask=0")
    print("  layout+swizzle model reproduces audited 552b8fdf group addresses")
    print("  C++/compiler plumbing present (source scan)")
    for alloc_offset, total in checked:
        print(f"  alloc_offset={alloc_offset}: scalar==vector, "
              f"{total} bytes covered exactly once")
    print("  negative cases rejected: " + ", ".join(sorted(negatives)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
