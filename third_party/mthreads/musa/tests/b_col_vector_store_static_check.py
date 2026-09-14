#!/usr/bin/env python3
"""CPU gate for the experimental exact B-col writer; no device ABI claim.

The native matcher additionally enumerates the actual C++ LinearLayout. This
test uses the independent blocked-layout construction from the existing A
verifier and compares scalar/vector memory images and logical B coordinates.
"""

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Optional, Tuple

from a_only_vector_store_static_check import apply_bases, build_bases

ROOT = Path(__file__).resolve().parents[4]
OPTION = "ph1_m64up_b_col_vector_store"


def swizzle(address, granularity=16):
    line, within = divmod(address, 256)
    sector, byte = divmod(within, granularity)
    return 256 * line + granularity * (sector ^ (line % (256 // granularity))) + byte


def prove_image(bases, base, order=(0, 1), granularity=16):
    scalar, vector = {}, {}
    groups = 0
    for warp in range(4):
        for lane in range(32):
            addresses, values = [], []
            for reg in range(16):
                k, n = apply_bases(bases, reg, lane, warp)
                assert (k, n) == (8 * (lane % 8) + reg % 8,
                                   lane // 8 + 4 * warp + 16 * (reg // 8))
                linear = n * 64 + k if order == (0, 1) else k * 32 + n
                address = swizzle(base + 2 * linear, granularity)
                assert base <= address < base + 4096 and address % 2 == 0
                # Every logical B element is a distinct opaque 16-bit payload.
                value = k * 32 + n
                assert address not in scalar
                scalar[address] = value
                addresses.append(address)
                values.append(value)
            for start in (0, 8):
                first = addresses[start]
                assert first % 16 == 0
                assert addresses[start:start + 8] == list(range(first, first + 16, 2))
                groups += 1
                for item, value in enumerate(values[start:start + 8]):
                    assert first + 2 * item not in vector
                    vector[first + 2 * item] = value
    assert scalar == vector and len(vector) == 2048
    assert sorted(vector) == list(range(base, base + 4096, 2))
    # Software reader model of col/K-major SG16. Hardware descriptor decoding
    # is deliberately not modeled and remains a separate device gate.
    for k in range(64):
        for n in range(32):
            address = swizzle(base + 2 * (n * 64 + k))
            assert vector[address] == k * 32 + n
    return groups


def check_option_hash():
    compiler = ROOT / "third_party/mthreads/backend/compiler.py"
    tree = ast.parse(compiler.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MUSAOptions")
    legacy = copy.deepcopy(cls)
    legacy.body = [node for node in legacy.body
                   if not (isinstance(node, ast.AnnAssign) and node.target.id == OPTION)]
    hash_fn = next(node for node in legacy.body if isinstance(node, ast.FunctionDef) and node.name == "hash")
    hash_fn.body = [node for node in hash_fn.body
                    if not (isinstance(node, ast.If) and OPTION in ast.unparse(node))]
    with tempfile.TemporaryDirectory() as directory:
        lib = Path(directory) / "libdevice.bc"
        lib.write_bytes(b"CPU hash field test only")

        def make_class(node):
            scope = dict(dataclass=dataclass, Optional=Optional, Tuple=Tuple,
                         Path=Path, hashlib=hashlib, __file__=str(compiler),
                         knobs=SimpleNamespace(musa=SimpleNamespace(libdevice_path=str(lib))),
                         _resolve_toolchain_paths=lambda options: ("fixture-llc", "fixture-lld", None),
                         _detect_llvm_major_version=lambda path: 22,
                         _tool_version_signature=lambda path: "fixture-signature:" + path,
                         file_hash=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest())
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         str(compiler), "exec"), scope)
            return scope["MUSAOptions"]

        current, previous = make_class(cls), make_class(legacy)
        assert current().hash() == previous().hash()
        assert current(**{OPTION: False}).hash() == current().hash()
        assert current(**{OPTION: True}).hash() != current().hash()
        assert current(**{OPTION: True}, num_stages=1).hash() != current(**{OPTION: True}, num_stages=2).hash()
    # This tests the actual options-hash method, not the full triton_key: source
    # and rebuilt libtriton identities intentionally rotate the full cache key.
    return {"default_off_matches_previous_options_hash": True,
            "enabled_changes_options_hash": True, "stage_count_changes_hash": True}


def check_source():
    cpp = (ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUToLLVM/LoadStoreOpToLLVM.cpp").read_text()
    selector = (ROOT / "third_party/mthreads/musa/lib/TritonMUSAGPUTransforms/AccelerateMUSAMatmul.cpp").read_text()
    b_body = cpp[cpp.index("static bool proveExactM64UpBColVectorGroups("):cpp.index("static LogicalResult lowerPH1SwizzledSharedStoreFromTensor(")]
    for required in ("layout.apply(", "getFreeVariableMasks(regTy)", "*opIdx != 1",
                     "rowMajor.getValue()", "allocationOffset % 4096", "inVals.size() != 16",
                     "linearizePH1TMELinearCoords(", "applyPH1TMESwizzleToByteAddress("):
        assert required in b_body, required
    emitter = b_body[b_body.index("static bool emitExactM64UpBColVectorStore("):]
    assert emitter.index("addresses.push_back") < emitter.index("targetInfo.storeDShared")
    assert "return false" not in emitter[emitter.index("targetInfo.storeDShared"):]
    assert not any(word in b_body for word in ("barrier(", "warpSync(", "loadDShared(", "LocalStoreOp"))
    assert selector.index("newOrder = {0, 1}") < selector.index("bool isRowMajor", selector.index("bool useExactBCol"))
    assert "exactBTile && opIdx == 1" in selector
    assert "matrixShape->rank == 2 && m == 32" in selector
    assert "inferSqmmaLayout(newB)" in selector
    utils = (ROOT / "third_party/mthreads/musa/include/TritonMUSACommon/MMAOperandUtils.h").read_text()
    assert "std::optional<bool> rowMajor;" in utils
    return {"source_guard_scan": "PASS; structural checks only, not native execution",
            "no_new_barrier_or_shared_reload": True}


def main():
    bases = build_bases((64, 32), (0, 1), (8, 1), (8, 4), (1, 4))
    assert bases == {"register": [(1, 0), (2, 0), (4, 0), (0, 16)],
                     "lane": [(8, 0), (16, 0), (32, 0), (0, 1), (0, 2)],
                     "warp": [(0, 4), (0, 8)]}
    # All 48 4KiB-aligned bases in the documented 192KiB address envelope.
    # This is an address-domain enumeration; it does not assert current GPU
    # capacity or authorize allocating 192KiB on a device.
    groups = sum(prove_image(bases, base) for base in range(0, 196608, 4096))
    bad_bases = copy.deepcopy(bases)
    bad_bases["lane"][0] = (16, 0)
    negatives = {
        "current_row_writer_is_scattered": lambda: prove_image(bases, 4096, order=(1, 0), granularity=32),
        "allocation_272B_boundary_escape": lambda: prove_image(bases, 272),
        "wrong_lane_basis": lambda: prove_image(bad_bases, 4096),
    }
    for name, check in negatives.items():
        try:
            check()
        except AssertionError:
            continue
        raise AssertionError("counterexample was accepted: " + name)
    print(json.dumps({"status": "CPU_PASS", "elements_per_tile": 2048,
                      "vector8_groups_per_tile": 256, "allocation_bases": 48,
                      "total_vector8_groups_checked": groups,
                      "negative_cases_rejected": list(negatives),
                      "option_hash": check_option_hash(), "source": check_source(),
                      "device_SQMMA_reader_ABI": None, "native_candidate_compilation": None}, indent=2))


if __name__ == "__main__":
    main()
