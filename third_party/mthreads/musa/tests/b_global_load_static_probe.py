#!/usr/bin/env python3
"""Zero-GPU / zero-compiler static probe for the B global-load lowering split.

Scope: locate where the current S5000 MUSA `fused_moe_kernel` M64-up B operand
global load first diverges from the old (triton_musa 3.2) build along the same
program. Nothing here executes a compiler, GPU, container, network, or any path
outside this worktree; it only reads the already-archived evidence bundle.

Facts proven from the archived bundle (no compiler needed):
  * The B `tt.load` is structurally identical in old and current TTGIR:
    unmasked, `tensor<64x32x!tt.ptr<bf16>, #blocked1>`, and built by the same
    `make_range -> expand_dims -> extsi -> broadcast -> addi -> addptr` chain.
  * `#blocked` / `#blocked1` are identical old vs current, so the linear-encoding
    `contigPerThread[order[0]]` term in `ModuleAxisInfoAnalysis::getContiguity`
    is the same for both.
  * Post-lowering LLIR: current has 16 scalar `load bfloat, ptr addrspace(1)`
    (B) + 2 `<8 x bfloat>` (A); old has 4 `<8 x bfloat>` and 0 scalar B. The 16
    current B loads use byte offsets {2,4,6,8,10,12,14} off a base with
    `align 16`, i.e. the addresses are contiguous and 16B aligned even though
    they were scalarized.
  * `LoadOpConversion` and `getVectorSize` in
    `TritonMUSAGPUToLLVM/LoadStoreOpToLLVM.cpp` are byte-identical to the
    initial mthreads 3.6 import, and `getVectorSize` returns
    `min(ModuleAxisInfoAnalysis::getContiguity(ptr), 8)`.

Conclusion this probe asserts: the first observable divergence for the B global
load is the vector-width value returned by `ModuleAxisInfoAnalysis`
(alignment/contiguity) as consumed by `LoadOpConversion::getVectorSize`, i.e.
post-TTGIR and upstream of LLVM O3. It does NOT prove the internal AxisInfo
lattice root cause, and therefore does NOT authorize forcing vectorization.

Run:
    python3 third_party/mthreads/musa/tests/b_global_load_static_probe.py
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BUNDLE = REPO_ROOT / "evidence_bundle/lowering-audit/evidence"
IR = BUNDLE / "ir"
SOURCE = BUNDLE / "source"
CURRENT_CPP = (REPO_ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/"
                             "LoadStoreOpToLLVM.cpp")
AXISINFO_CPP = REPO_ROOT / "third_party/mthreads/lib/Analysis/AxisInfo.cpp"
INITIAL_CPP = (SOURCE / "initial-36-import/third_party/mthreads/musa/lib/"
                        "TritonMUSAGPUToLLVM/LoadStoreOpToLLVM.cpp")
REPORT_JSON = Path(__file__).resolve().parent / "b_global_load_probe_report.json"

B_LOAD_RE = re.compile(
    r"tt\.load %\S+ : tensor<64x32x!tt\.ptr<bf16>, #blocked1>(?P<mask>[^l]*)")
LAYOUT_RE = re.compile(r"^#(blocked1?)\s*=\s*#(?:ttg|triton_gpu)\.blocked<")
NORMALIZE_PREFIX = [("#triton_gpu.", "#ttg."), ("triton_gpu.", "ttg.")]


def read(path):
    return path.read_text()


def normalize(text):
    for old, new in NORMALIZE_PREFIX:
        text = text.replace(old, new)
    return text


def extract_function(text, start_re, end_re):
    start = text.index(start_re)
    end = text.index(end_re, start)
    return text[start:end]


def raw_pointer_chain(ttgir, b_make_keyword, b_addptr_keyword):
    """Return the lines of the B-pointer build chain, from the B make_range to
    the tensor<64x32> addptr that produces the load pointer, normalized."""
    lines = normalize(ttgir).splitlines()
    start = None
    for i, ln in enumerate(lines):
        if (b_make_keyword in ln and "make_range {end = 64 : i32, start = 0 : i32}"
                in ln and "slice<{dim = 1" in ln):
            start = i
            break
    assert start is not None, "B make_range not found"
    end = None
    addptr_re = re.compile(r"^\s*%\S+\s*=\s*tt\.addptr\b")
    for j in range(start, min(start + 40, len(lines))):
        if (b_addptr_keyword in lines[j] and addptr_re.match(lines[j])
                and "tensor<64x32x!tt.ptr<bf16>, #blocked1>" in lines[j]):
            end = j
            break
    assert end is not None, "B addptr not found"
    return lines[start:end + 1]


def op_signature(lines):
    """Normalize the chain to a sequence of (op, salient-params) tuples so the
    two dialect generations (triton_gpu vs ttg, %argN vs %name) compare equal."""
    sig = []
    for ln in lines:
        body = ln.split("loc(")[0].strip()
        if "make_range" in body:
            m = re.search(r"make_range \{end = (\d+) : i32, start = (\d+) : i32\}", body)
            sig.append(("make_range", m.group(2), m.group(1)))
        elif "expand_dims" in body:
            m = re.search(r"\{axis = (\d+) : i32\}", body)
            sig.append(("expand_dims", m.group(1)))
        elif "extsi" in body:
            sig.append(("extsi", None))
        elif "splat" in body:
            sig.append(("splat", None))
        elif "muli" in body:
            sig.append(("muli", None))
        elif "broadcast" in body:
            m = re.search(r": tensor<(\d+)x(\d+)xi64, #blocked1> -> "
                          r"tensor<(\d+)x(\d+)xi64, #blocked1>", body)
            sig.append(("broadcast", m.groups() if m else None))
        elif "addi" in body:
            sig.append(("addi", None))
        elif "addptr" in body:
            sig.append(("addptr", "tensor<64x32x!tt.ptr<bf16>, #blocked1>"))
    return sig


def current_b_scalar_offsets(llir):
    geps = re.findall(
        r"getelementptr inbounds nuw i8, ptr addrspace\(1\) "
        r"%\.pn64151, i64 (\d+)", llir)
    geps2 = re.findall(
        r"getelementptr inbounds nuw i8, ptr addrspace\(1\) "
        r"%\.pn48152, i64 (\d+)", llir)
    first_base = re.search(
        r"load bfloat, ptr addrspace\(1\) %\.pn64151, align (\d+)", llir)
    return sorted(int(g) for g in geps), sorted(int(g) for g in geps2), \
        int(first_base.group(1)) if first_base else None


def main():
    checks = []
    old_ttgir = read(IR / "old-up.ttgir")
    cur_ttgir = read(IR / "current-up.ttgir")
    old_llir = read(IR / "old-up.llir")
    cur_llir = read(IR / "current-up.llir")

    def check(name, cond, detail="", msg=None):
        checks.append({"name": name, "passed": bool(cond), "detail": detail})
        assert cond, f"{name}: {msg or detail}"

    # 1. Layouts identical old vs current (so the linear-encoding contiguity
    #    term feeding getContiguity is unchanged).
    old_layouts = {m.group(1): m.group(0)
                   for m in (LAYOUT_RE.match(l) for l in normalize(old_ttgir).splitlines())
                   if m}
    cur_layouts = {m.group(1): m.group(0)
                   for m in (LAYOUT_RE.match(l) for l in normalize(cur_ttgir).splitlines())
                   if m}
    check("blocked_layouts_identical",
          old_layouts.get("blocked") == cur_layouts.get("blocked")
          and old_layouts.get("blocked1") == cur_layouts.get("blocked1"),
          detail="old and current #blocked/#blocked1 definitions match",
          msg=f"old={old_layouts} cur={cur_layouts}")

    # 2. B tt.load contract identical and unmasked.
    old_b = B_LOAD_RE.search(normalize(old_ttgir))
    cur_b = B_LOAD_RE.search(normalize(cur_ttgir))
    check("b_load_present_both", old_b and cur_b,
          detail="B tt.load present in both TTGIR",
          msg="B tt.load not found")
    check("b_load_unmasked_both",
          old_b.group("mask").strip() == "" and cur_b.group("mask").strip() == "",
          detail="B tt.load unmasked in both; shape/layout #blocked1 equal",
          msg=f"old_mask={old_b.group('mask')!r} cur_mask={cur_b.group('mask')!r}")

    # 3. B pointer chains have the same normalized signature.
    old_sig = op_signature(raw_pointer_chain(old_ttgir, "tt.make_range", "tt.addptr"))
    cur_sig = op_signature(raw_pointer_chain(cur_ttgir, "tt.make_range", "tt.addptr"))
    check("b_pointer_chain_signature_identical", old_sig == cur_sig,
          detail=f"both chains = {cur_sig}",
          msg=f"old={old_sig} cur={cur_sig}")

    # 4. LLIR global-load counts: A vector8 preserved, B scalarized in current.
    cur_vec = len(re.findall(r"load <8 x bfloat>, ptr addrspace\(1\)", cur_llir))
    cur_scalar = len(re.findall(r"load bfloat, ptr addrspace\(1\)", cur_llir))
    old_vec = len(re.findall(r"load <8 x bfloat>, ptr %", old_llir))
    old_scalar = len(re.findall(r"load bfloat, ptr %", old_llir)) + \
        len(re.findall(r"load bfloat, ptr addrspace\(1\)", old_llir))
    check("current_A_vector8_B_scalar16",
          cur_vec == 2 and cur_scalar == 16,
          detail=f"current: {cur_vec} vector8 (A) + {cur_scalar} scalar bf16 (B)",
          msg=f"vec={cur_vec} scalar={cur_scalar}")
    check("old_A_and_B_vector8",
          old_vec == 4 and old_scalar == 0,
          detail=f"old: {old_vec} vector8 (A+B) + {old_scalar} scalar",
          msg=f"vec={old_vec} scalar={old_scalar}")

    # 5. The scalarized B addresses are provably contiguous and 16B aligned.
    g1, g2, align = current_b_scalar_offsets(cur_llir)
    check("current_B_addresses_contiguous_aligned",
          g1 == [2, 4, 6, 8, 10, 12, 14] and g2 == [2, 4, 6, 8, 10, 12, 14]
          and align == 16,
          detail=f"B scalar byte offsets {g1}/{g2} off base align {align}",
          msg=f"group1={g1} group2={g2} base_align={align}")

    # 6. The load-lowering code path did not change inside this repo history.
    init_text = read(INITIAL_CPP)
    cur_text = read(CURRENT_CPP)
    init_vec = extract_function(init_text, "unsigned getVectorSize(",
                                "StringRef getMusaMemcpyG2SIntrinsic(")
    cur_vec_fn = extract_function(cur_text, "unsigned getVectorSize(",
                                  "StringRef getMusaMemcpyG2SIntrinsic(")
    check("getVectorSize_unchanged", init_vec == cur_vec_fn,
          detail="getVectorSize byte-identical to initial-36 import",
          msg="getVectorSize differs from initial-36 import")
    init_loadop = extract_function(init_text, "struct LoadOpConversion :",
                                   "struct StoreOpConversion :")
    cur_loadop = extract_function(cur_text, "struct LoadOpConversion :",
                                  "struct StoreOpConversion :")
    # Direction 1 adds an explicitly marked, default-off, void diagnostic probe
    # call inside matchAndRewrite. Strip exactly that block (and nothing else)
    # before the byte-identity comparison, so the original "the lowering logic
    # is unchanged vs initial-36" evidence still holds; a residual diff means
    # something outside the probe changed.
    probe_call_block = (
        "\n    // Read-only diagnostic; disabled unless the module marker is "
        "present. The\n"
        "    // return value is intentionally discarded: the probe cannot "
        "change `vec`\n"
        "    // or any rewrite in this pattern.\n"
        "    (void)probeExactM64UpBGlobalLoad(op, ptr, axisInfoAnalysis, "
        "vec);\n")
    cur_loadop_without_probe = cur_loadop.replace(probe_call_block, "")
    check("LoadOpConversion_unchanged_except_opt_in_probe",
          init_loadop == cur_loadop_without_probe
          and probe_call_block in cur_loadop,
          detail="LoadOpConversion byte-identical to initial-36 import after "
                 "removing the marked default-off diagnostic probe call",
          msg="LoadOpConversion differs beyond the opt-in probe call")

    # 7. The decision chain exists and getVectorSize caps at 128 bits.
    check("decision_chain_present",
          "getVectorSize(ptr, axisInfoAnalysis)" in cur_text
          and "unsigned vec = getVectorSize(ptr, axisInfoAnalysis);" in cur_loadop
          and "if (vec == 1) {" in cur_loadop
          and "vec = std::min(vec, axisInfoAnalysis.getMaskAlignment" in cur_loadop
          and "128u / pointeeBitWidth" in cur_vec_fn,
          detail="LoadOpConversion vec==1/vec>1 branch + getVectorSize cap present",
          msg="load vector-size decision markers missing")

    axis_text = normalize(read(AXISINFO_CPP))
    check("axisinfo_contiguity_chain_present",
          "unsigned ModuleAxisInfoAnalysis::getContiguity(Value value)" in axis_text
          and "getContiguity(value, elemTy.getIntOrFloatBitWidth())" in axis_text
          and "unsigned ModuleAxisInfoAnalysis::getContiguity(Value offsetsValue," in axis_text
          and "unsigned ModuleAxisInfoAnalysis::getAlignment(Value value) {" in axis_text
          and "uniqueContigPerThread[order[0]]" in axis_text
          and "contiguity = std::min(align, contiguity);" in axis_text,
          detail="AxisInfo getContiguity -> getAlignment -> min(align, contig) present",
          msg="AxisInfo getContiguity/getAlignment markers missing")

    # 8. The B *shared* layout did change (out of scope, documented, untouched).
    old_shared1 = re.search(r"^#shared1\s*=\s*(.*)$",
                            normalize(old_ttgir), re.M)
    cur_shared1 = re.search(r"^#shared1\s*=\s*(.*)$",
                            normalize(cur_ttgir), re.M)
    check("b_shared_layout_changed_out_of_scope",
          old_shared1 is not None and cur_shared1 is not None
          and old_shared1.group(1) != cur_shared1.group(1),
          detail=f"old={old_shared1.group(1)} cur={cur_shared1.group(1)} "
                 "(documented, not touched)",
          msg="expected B shared layout to differ")

    report = {
        "probe": "b-global-load-static-probe-v1",
        "status": "PASS",
        "scope": "archived M64-up B global-load lowering divergence only",
        "divergence": {
            "stage": "post-TTGIR, at LoadOpConversion vector-width selection",
            "decision_chain": [
                "LoadStoreOpToLLVM.cpp: LoadOpConversion::matchAndRewrite",
                "LoadStoreOpToLLVM.cpp: getVectorSize -> axisInfoAnalysis.getContiguity(ptr)",
                "Analysis/AxisInfo.cpp: ModuleAxisInfoAnalysis::getContiguity -> getAlignment",
            ],
            "implicated": "ModuleAxisInfoAnalysis alignment/contiguity value",
            "exonerated_by_static_evidence": [
                "front-end TTGIR B load (identical shape/layout/mask/pointer chain)",
                "LoadOpConversion / getVectorSize structure (unchanged vs initial-36 import)",
                "blocked layout contigPerThread term (identical old vs current)",
                "LLVM O3 / backend-opt as a *producer* of scalar form "
                "(optimizations combine, not split, load vectors; A stays vector)",
            ],
            "not_proven": [
                "why the current AxisInfo lattice returns <8 for B "
                "(no pre-O3 IR, old 3.2 AxisInfo source unavailable locally)",
                "whether a safe forced-vec8 override exists for all M64-up variants",
            ],
        },
        "facts": {
            "layouts_identical": True,
            "b_load_identical_unmasked": True,
            "b_pointer_chain_identical": True,
            "current_vector8_global_loads": cur_vec,
            "current_scalar_bf16_global_loads": cur_scalar,
            "old_vector8_global_loads": old_vec,
            "old_scalar_bf16_global_loads": old_scalar,
            "current_b_scalar_offsets": [g1, g2],
            "current_b_base_align_bytes": align,
        },
        "checks": checks,
        "live_execution_authorized_by_this_artifact": False,
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print("B global-load static probe: PASS")
    print(f"  divergence: post-TTGIR at {report['divergence']['implicated']}")
    print(f"  current: {cur_vec} vector8 (A) + {cur_scalar} scalar bf16 (B); "
          f"old: {old_vec} vector8, {old_scalar} scalar")
    print(f"  B scalar offsets {g1}/{g2} off base align {align} -> contiguous "
          "and 16B aligned, yet scalarized")
    print("  getVectorSize identical to initial-36 import; LoadOpConversion "
          "identical after removing the marked opt-in probe call")
    print(f"  report: {REPORT_JSON.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
