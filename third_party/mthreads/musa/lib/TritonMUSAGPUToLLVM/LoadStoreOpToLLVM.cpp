#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/PH1WarpShufflePlan.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#ifdef __TLE__
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#endif // __TLE__
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <functional>
#include <initializer_list>
#ifdef __TLE__
#include <limits>
#endif // __TLE__
#include <numeric>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

inline constexpr llvm::StringLiteral kInplaceLoadAttr =
    "musa.inplace_load_candidate";

struct PH1SwizzledMatrixViewInfo {
  SmallVector<int64_t, 2> matrixPhysicalShape;
  SmallVector<unsigned, 2> matrixOrder;
  int64_t batchStrideElements = 0;
};

static FailureOr<PH1SwizzledMatrixViewInfo>
getPH1SwizzledMatrixViewInfo(MemDescType memDescTy) {
  unsigned rank = memDescTy.getRank();
  if (rank != 3)
    return failure();

  SmallVector<int64_t> physicalShape =
      triton::musa::getMemDescPhysicalShape(memDescTy);
  if (physicalShape.size() != rank)
    return failure();

  auto order = triton::gpu::getOrder(memDescTy);
  PH1SwizzledMatrixViewInfo info;
  info.matrixPhysicalShape = {physicalShape[rank - 2], physicalShape[rank - 1]};

  if (order.size() == 2) {
    for (unsigned dim : order) {
      if (dim >= 2)
        return failure();
      info.matrixOrder.push_back(dim);
    }
  } else if (order.size() == rank) {
    for (unsigned dim : order) {
      if (dim < rank - 2)
        continue;
      info.matrixOrder.push_back(dim - (rank - 2));
      if (info.matrixOrder.size() == 2)
        break;
    }
  } else {
    return failure();
  }

  if (info.matrixOrder.size() != 2)
    return failure();

  info.batchStrideElements =
      info.matrixPhysicalShape[0] * info.matrixPhysicalShape[1];
  if (info.batchStrideElements <= 0 ||
      !triton::musa::toI32(info.batchStrideElements))
    return failure();

  return info;
}

static bool requiresAbsoluteSwizzledSharedAccess(MemDescType memDescTy) {
  unsigned rank = memDescTy.getRank();
  if (rank != 2 && rank != 3)
    return false;
  auto sharedEnc = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
      memDescTy.getEncoding());
  if (!sharedEnc)
    return false;
  FailureOr<triton::musa::ResolvedTMESwizzleConfig> swizzle = failure();
  if (rank == 2) {
    swizzle = triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy);
  } else {
    auto matrixView = getPH1SwizzledMatrixViewInfo(memDescTy);
    if (succeeded(matrixView))
      swizzle = triton::musa::resolveTMESwizzleConfigFromMatrixView(
          memDescTy, matrixView->matrixPhysicalShape, matrixView->matrixOrder);
  }
  return succeeded(swizzle) && swizzle->swizzleGranularity !=
                                   triton::musa::TMESwizzleGranularity::SG_NONE;
}

struct PH1TMESwizzleValueConfig {
  Value granularityBytes;
  Value strideBytes;
  Value lineBytes;
  bool hasSwizzle = false;
};

static FailureOr<PH1TMESwizzleValueConfig>
resolvePH1TMESwizzleValueConfig(Location loc, MemDescType memDescTy,
                                ConversionPatternRewriter &rewriter,
                                ArrayRef<int64_t> matrixPhysicalShape = {},
                                ArrayRef<unsigned> matrixOrder = {}) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  PH1TMESwizzleValueConfig config;

  auto emitConfig = [&](triton::musa::TMESwizzleGranularity granularity,
                        triton::musa::TMESwizzleStride stride,
                        triton::musa::TMESwizzleLine line) {
    config.hasSwizzle =
        granularity != triton::musa::TMESwizzleGranularity::SG_NONE;
    config.granularityBytes = b.i32_val(static_cast<int32_t>(
        triton::musa::getSwizzleGranularityBytes(granularity)));
    config.strideBytes = b.i32_val(
        static_cast<int32_t>(triton::musa::getSwizzleStrideBytes(stride)));
    config.lineBytes = b.i32_val(
        static_cast<int32_t>(triton::musa::getSwizzleLineBytes(line)));
  };

  auto swizzle =
      matrixPhysicalShape.empty()
          ? triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy)
          : triton::musa::resolveTMESwizzleConfigFromMatrixView(
                memDescTy, matrixPhysicalShape, matrixOrder);
  if (failed(swizzle))
    return failure();

  emitConfig(swizzle->swizzleGranularity, swizzle->swizzleStride,
             swizzle->swizzleLine);
  return config;
}

static Value
applyPH1TMESwizzleToByteAddressValue(TritonLLVMOpBuilder &b, Value addrBytes,
                                     const PH1TMESwizzleValueConfig &config) {
  if (!config.hasSwizzle)
    return addrBytes;

  Value lineOffset = b.urem(addrBytes, config.lineBytes);
  Value lineId = b.udiv(addrBytes, config.lineBytes);
  Value swizzleGroup = b.udiv(config.strideBytes, config.granularityBytes);
  Value swizzleLineId = b.urem(lineId, swizzleGroup);
  Value sectorInLine = b.udiv(lineOffset, config.granularityBytes);
  Value offsetInSector = b.urem(lineOffset, config.granularityBytes);
  Value targetSectorInLine = b.xor_(sectorInLine, swizzleLineId);
  return b.add(b.add(b.mul(lineId, config.lineBytes),
                     b.mul(targetSectorInLine, config.granularityBytes)),
               offsetInSector);
}

Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const mlir::triton::MUSA::TargetInfo &targetInfo);

struct PH1SwizzledSharedAccess {
  Type llvmElemTy;
  Value smemElemBase;
  Value smemOffsetBytes;
  Value affineOffsetElems;
  Value elemBytesVal;
  unsigned elemBytes = 0;
  unsigned rank = 0;
  bool hasAffineOffset = false;
  SmallVector<int64_t> physicalShape;
  SmallVector<unsigned> order;
  SmallVector<int64_t, 2> matrixPhysicalShape;
  SmallVector<unsigned, 2> matrixOrder;
  int64_t batchStrideElements = 0;
  PH1TMESwizzleValueConfig swizzleConfig;
};

static FailureOr<PH1SwizzledSharedAccess>
getPH1SwizzledSharedAccess(Location loc, MemDescType memDescTy,
                           SharedMemoryObject smemObj, Type llvmElemTy,
                           ConversionPatternRewriter &rewriter,
                           Operation *diagnosticOp) {
  if (!requiresAbsoluteSwizzledSharedAccess(memDescTy))
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  unsigned rank = memDescTy.getRank();
  auto order = triton::gpu::getOrder(memDescTy);
  SmallVector<int64_t> physicalShape =
      triton::musa::getMemDescPhysicalShape(memDescTy);
  if (physicalShape.size() != rank)
    return failure();
  if (rank == 2 && order.size() != rank)
    return failure();

  unsigned elemBytes =
      std::max<unsigned>(1, memDescTy.getElementTypeBitWidth() / 8);

  SmallVector<int64_t, 2> matrixPhysicalShape;
  SmallVector<unsigned, 2> matrixOrder;
  int64_t batchStrideElements = 0;
  if (rank == 3) {
    auto matrixView = getPH1SwizzledMatrixViewInfo(memDescTy);
    if (failed(matrixView))
      return failure();
    matrixPhysicalShape = matrixView->matrixPhysicalShape;
    matrixOrder = matrixView->matrixOrder;
    batchStrideElements = matrixView->batchStrideElements;
  }

  auto swizzleConfig =
      rank == 3
          ? resolvePH1TMESwizzleValueConfig(loc, memDescTy, rewriter,
                                            matrixPhysicalShape, matrixOrder)
          : resolvePH1TMESwizzleValueConfig(loc, memDescTy, rewriter);
  if (failed(swizzleConfig) || !swizzleConfig->hasSwizzle)
    return failure();

  auto func = diagnosticOp->getParentOfType<FunctionOpInterface>();
  if (!func) {
    diagnosticOp->emitError("PH1 swizzled shared access requires parent "
                            "function for shared-memory base");
    return failure();
  }

  Value smemObjBase = smemObj.getBase();
  Value smemRawBase = LLVM::getStackPointer(rewriter, func);
  Value smemOffsetBytes =
      b.sub(b.ptrtoint(i32_ty, smemObjBase), b.ptrtoint(i32_ty, smemRawBase));
  bool hasAffineOffset =
      SharedMemoryObject::isAffineSharedMemoryAccess(memDescTy);

  PH1SwizzledSharedAccess access;
  access.llvmElemTy = llvmElemTy;
  access.smemElemBase = b.bitcast(smemObjBase, ptr_ty(ctx, 3));
  access.smemOffsetBytes = smemOffsetBytes;
  access.affineOffsetElems = smemObj.getShmemOffset(loc, rewriter, memDescTy);
  access.elemBytesVal = b.i32_val(static_cast<int32_t>(elemBytes));
  access.elemBytes = elemBytes;
  access.rank = rank;
  access.hasAffineOffset = hasAffineOffset;
  access.physicalShape = std::move(physicalShape);
  access.order = std::move(order);
  access.matrixPhysicalShape = std::move(matrixPhysicalShape);
  access.matrixOrder = std::move(matrixOrder);
  access.batchStrideElements = batchStrideElements;
  access.swizzleConfig = *swizzleConfig;
  return access;
}

static FailureOr<Value> getPH1SwizzledSharedElementPtr(
    Location loc, ConversionPatternRewriter &rewriter,
    const PH1SwizzledSharedAccess &access, ArrayRef<Value> coord) {
  if (coord.size() != access.rank)
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  FailureOr<Value> lmsOffsetInElem = failure();
  if (access.rank == 3) {
    SmallVector<Value, 2> matrixCoord{coord[access.rank - 2],
                                      coord[access.rank - 1]};
    auto matrixOffsetInElem = triton::musa::linearizePH1TMELinearCoords(
        b, matrixCoord, access.matrixPhysicalShape, access.matrixOrder,
        access.elemBytes);
    if (failed(matrixOffsetInElem))
      return failure();
    Value batchOffsetInElem = b.mul(
        coord[0], b.i32_val(static_cast<int32_t>(access.batchStrideElements)));
    Value offsetInElem = b.add(batchOffsetInElem, *matrixOffsetInElem);
    lmsOffsetInElem = offsetInElem;
  } else {
    lmsOffsetInElem = triton::musa::linearizePH1TMELinearCoords(
        b, coord, access.physicalShape, access.order, access.elemBytes);
  }
  if (failed(lmsOffsetInElem))
    return failure();

  Value offsetInElem = *lmsOffsetInElem;
  if (access.hasAffineOffset)
    offsetInElem = b.xor_(offsetInElem, access.affineOffsetElems);

  Value lmsAddrInByte =
      b.add(b.mul(offsetInElem, access.elemBytesVal), access.smemOffsetBytes);
  Value swizzledAddrInByte = applyPH1TMESwizzleToByteAddressValue(
      b, lmsAddrInByte, access.swizzleConfig);
  Value swizzledAddrRelInByte =
      b.sub(swizzledAddrInByte, access.smemOffsetBytes);
  Value swizzledElemOffset = b.udiv(swizzledAddrRelInByte, access.elemBytesVal);
  Value ptr = b.gep(access.smemElemBase.getType(), access.llvmElemTy,
                    access.smemElemBase, swizzledElemOffset);
  return ptr;
}

// Exact-contract, A-only vector grouping for PH1 swizzled shared stores.
//
// The generic helper below stores one BF16 element per canonical register. For
// the audited M64-up SQMMA A operand, the lowest `log2(vec)` register bits are
// the only producers of the lowest `log2(vec)` bits of the pre-swizzle element
// offset, and the PH1 swizzle is a permutation at `vec`-element sector
// granularity. Each aligned run of `vec` registers then maps to one contiguous,
// `vec * elemBytes`-byte-aligned sector and may be stored as a single
// `<vec x bfloat>` value.
//
// The proof below is derived purely from the tensor/memdesc encodings and the
// static shared-memory allocation offset. It never inspects the final LLIR and
// never relies on a compile-key or input hash (neither is available in this
// lowering). If any check fails the caller keeps the original per-element
// scalar loop unchanged.
static bool ph1EqualsIntList(ArrayRef<int64_t> values,
                             std::initializer_list<int64_t> expected) {
  return values.size() == expected.size() &&
         std::equal(values.begin(), values.end(), expected.begin());
}

static bool ph1EqualsUnsignedList(ArrayRef<unsigned> values,
                                  std::initializer_list<unsigned> expected) {
  return values.size() == expected.size() &&
         std::equal(values.begin(), values.end(), expected.begin());
}

static bool ph1AllOnes(ArrayRef<unsigned> values) {
  return !values.empty() &&
         std::all_of(values.begin(), values.end(),
                     [](unsigned value) { return value == 1; });
}

// Explicit opt-in marker, set on the module by the MUSA backend only when the
// caller selects the audited M64-up/A specialization. No service caller sets
// it, so this vector store stays disabled unless a future caller opts in with
// a compile option that participates in the compile cache key.
inline constexpr llvm::StringLiteral kPH1M64UpAVectorStoreMarker =
    "musa.ph1_m64up_a_vector_store";
inline constexpr llvm::StringLiteral kPH1M64UpAVectorStoreNumStages =
    "musa.ph1_m64up_a_vector_store.num_stages";
inline constexpr llvm::StringLiteral kPH1M64UpAVectorStoreTarget = "musa:31";

// Returns true iff, for every lane/warp, register run [g*vec, (g+1)*vec) maps
// to `base + {0..vec-1}` with `base % vec == 0`, where `base` is the
// pre-swizzle element offset produced by `linearizePH1TMELinearCoords`.
static bool provePH1ContiguousRegisterVectorGroups(
    RankedTensorType regTy, ArrayRef<int64_t> physicalShape,
    ArrayRef<unsigned> order, unsigned elemBytes, unsigned vec) {
  MLIRContext *ctx = regTy.getContext();
  if (vec < 2 || !llvm::isPowerOf2_32(vec))
    return false;
  if (physicalShape.size() != 2 || order.size() != 2)
    return false;
  // The offset formula below assumes the last dim is the leading (fastest)
  // dim and that a single PH1 leading-dimension group is formed.
  if (order[0] != 1 || order[1] != 0)
    return false;
  auto grouping = triton::musa::getPH1TMELeadingDimGrouping(physicalShape,
                                                            order, elemBytes);
  if (failed(grouping) || grouping->numGroups != 1)
    return false;
  if (physicalShape[1] <= 0)
    return false;

  unsigned log2vec = llvm::Log2_32(vec);
  LinearLayout regLL = toLinearLayout(regTy);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  const auto &bases = regLL.getBases();
  if (!bases.contains(kReg) || !bases.contains(kLane) || !bases.contains(kWarp))
    return false;

  // Exact register count and no replicated/free register bits. The audited A
  // operand has 16 distinct registers per thread and a zero free mask.
  unsigned numRegs = regLL.getInDimSize(kReg);
  if (numRegs != 16)
    return false;
  auto freeRegMasks = getFreeVariableMasks(regTy);
  uint32_t freeRegMask = freeRegMasks.lookup(str_attr("register"));
  if (freeRegMask != 0)
    return false;

  // Maps one input basis vector to its contribution to the pre-swizzle element
  // offset. The contribution must be 0 or a single power of two so that XOR and
  // integer addition agree on the bits we reason about.
  auto offsetContribution = [&](ArrayRef<int32_t> basis,
                                int64_t &offset) -> bool {
    if (basis.size() != 2)
      return false;
    int64_t c0 = basis[0];
    int64_t c1 = basis[1];
    if (c0 < 0 || c1 < 0)
      return false;
    if (c0 != 0 && c1 != 0)
      return false;
    if (c0 != 0 && !llvm::isPowerOf2_64(static_cast<uint64_t>(c0)))
      return false;
    if (c1 != 0 && !llvm::isPowerOf2_64(static_cast<uint64_t>(c1)))
      return false;
    // linearizePH1TMELinearCoords == dim0 * shape[1] + dim1 for order [1, 0]
    // with a single leading-dimension group.
    offset = c0 * physicalShape[1] + c1;
    return offset == 0 || llvm::isPowerOf2_64(static_cast<uint64_t>(offset));
  };

  const auto &regBases = bases.find(kReg)->second;
  if (regBases.size() < log2vec)
    return false;
  // The low `log2(vec)` offset bits must come one-for-one from the low
  // `log2(vec)` register bits, so a register run only changes those bits.
  for (unsigned i = 0; i < log2vec; ++i) {
    int64_t offset = 0;
    if (!offsetContribution(regBases[i], offset))
      return false;
    if (offset != static_cast<int64_t>(1u << i))
      return false;
  }

  auto introducesLowBits = [&](ArrayRef<int32_t> basis) -> bool {
    int64_t offset = 0;
    if (!offsetContribution(basis, offset))
      return true; // Unmodelable basis: treat as unsafe.
    if (offset == 0)
      return false; // Free/broadcast bit: does not touch the low bits.
    return (offset & (static_cast<int64_t>(vec) - 1)) != 0;
  };
  for (unsigned i = log2vec; i < regBases.size(); ++i)
    if (introducesLowBits(regBases[i]))
      return false;
  for (const auto &basis : bases.find(kLane)->second)
    if (introducesLowBits(basis))
      return false;
  for (const auto &basis : bases.find(kWarp)->second)
    if (introducesLowBits(basis))
      return false;

  return (numRegs % vec) == 0;
}

// Returns the number of elements per vector store iff the memdesc/register pair
// is exactly the audited current M64-up SQMMA A operand and every grouping
// precondition is statically provable. `std::nullopt` means "keep the scalar
// loop"; the caller must not emit any store on failure.
static std::optional<unsigned> matchExactM64UpAOperandVectorContract(
    MemDescType memDescTy, RankedTensorType regTy, Operation *diagnosticOp,
    const PH1SwizzledSharedAccess &access) {
  // A-only: exactly the SQMMA A-operand shared allocation.
  if (!isa<triton::gpu::LocalAllocOp>(diagnosticOp))
    return std::nullopt;
  auto opIdx = triton::musa::getSqmmaOpIdx(diagnosticOp);
  if (!opIdx || *opIdx != 0)
    return std::nullopt;

  // Explicit opt-in marker. It is absent for every service caller, so the
  // vector store is disabled unless a caller selects the audited
  // specialization with a compile option that is part of the cache key.
  ModuleOp module = diagnosticOp->getParentOfType<ModuleOp>();
  if (!module)
    return std::nullopt;
  auto marker = module->getAttrOfType<BoolAttr>(kPH1M64UpAVectorStoreMarker);
  if (!marker || !marker.getValue())
    return std::nullopt;
  // Exact target capability 31, CTA/CGA size 1, and one pipeline stage, all
  // from module IR (the stage count is plumbed by the backend from the same
  // options object that feeds the compile cache key).
  auto target = module->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
  if (!target || target.getValue() != kPH1M64UpAVectorStoreTarget)
    return std::nullopt;
  auto numCTAs =
      module->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumCTAsName);
  if (!numCTAs || numCTAs.getInt() != 1)
    return std::nullopt;
  auto numStages =
      module->getAttrOfType<IntegerAttr>(kPH1M64UpAVectorStoreNumStages);
  if (!numStages || numStages.getInt() != 1)
    return std::nullopt;
  if (!ph1AllOnes(triton::gpu::getCTAsPerCGA(regTy.getEncoding())) ||
      !ph1AllOnes(triton::gpu::getCTAsPerCGA(memDescTy.getEncoding())))
    return std::nullopt;

  if (!isa<BFloat16Type>(memDescTy.getElementType()))
    return std::nullopt;
  if (memDescTy.getRank() != 2 || regTy.getRank() != 2)
    return std::nullopt;
  if (!ph1EqualsIntList(regTy.getShape(), {32, 64}))
    return std::nullopt;
  if (!ph1EqualsIntList(access.physicalShape, {32, 64}))
    return std::nullopt;
  if (access.hasAffineOffset)
    return std::nullopt;

  auto blocked = dyn_cast<BlockedEncodingAttr>(regTy.getEncoding());
  if (!blocked)
    return std::nullopt;
  if (!ph1EqualsUnsignedList(blocked.getSizePerThread(), {1u, 8u}) ||
      !ph1EqualsUnsignedList(blocked.getThreadsPerWarp(), {4u, 8u}) ||
      !ph1EqualsUnsignedList(blocked.getWarpsPerCTA(), {4u, 1u}) ||
      !ph1EqualsUnsignedList(triton::gpu::getOrder(regTy), {1u, 0u}))
    return std::nullopt;

  auto shared = dyn_cast<SwizzledSharedEncodingAttr>(memDescTy.getEncoding());
  if (!shared)
    return std::nullopt;
  if (shared.getVec() != 8 || shared.getPerPhase() != 2 ||
      shared.getMaxPhase() != 16 ||
      !ph1EqualsUnsignedList(shared.getOrder(), {1u, 0u}))
    return std::nullopt;

  auto swizzle = triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy);
  if (failed(swizzle))
    return std::nullopt;
  unsigned granularityBytes = static_cast<unsigned>(
      triton::musa::getSwizzleGranularityBytes(swizzle->swizzleGranularity));
  if (granularityBytes != 16 || access.elemBytes == 0 ||
      granularityBytes % access.elemBytes != 0)
    return std::nullopt;
  unsigned vec = granularityBytes / access.elemBytes;
  if (vec != 8)
    return std::nullopt;

  if (!provePH1ContiguousRegisterVectorGroups(
          regTy, access.physicalShape, access.order, access.elemBytes, vec))
    return std::nullopt;

  // `smemOffsetBytes` in getPH1SwizzledSharedElementPtr is exactly this
  // allocation's static offset below the raw shared base. The group base is
  // `allocation.offset + elemBytes * (multiple of vec)`; requiring the offset
  // to be granularity-aligned makes every emitted vector sector-aligned.
  auto allocOffsetAttr =
      diagnosticOp->getAttrOfType<IntegerAttr>("allocation.offset");
  if (!allocOffsetAttr)
    return std::nullopt;
  if ((allocOffsetAttr.getValue().getZExtValue() % granularityBytes) != 0)
    return std::nullopt;

  return vec;
}

// Emits one `<vec x bfloat>` store per proven group. All group addresses are
// computed before the first store, and no store is emitted if any address
// cannot be built, so a failure leaves the memory untouched for the scalar
// fallback.
static bool emitExactM64UpAOperandVectorStore(
    Location loc, ConversionPatternRewriter &rewriter,
    const PH1SwizzledSharedAccess &access, ArrayRef<Value> inVals,
    ArrayRef<SmallVector<Value>> srcIndices, unsigned vec, Value pred,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  if (vec == 0 || inVals.size() % vec != 0)
    return false;
  unsigned numGroups = inVals.size() / vec;
  SmallVector<Value, 4> groupBases;
  groupBases.reserve(numGroups);
  for (unsigned g = 0; g < numGroups; ++g) {
    unsigned baseIdx = g * vec;
    if (baseIdx >= srcIndices.size())
      return false;
    auto ptr = getPH1SwizzledSharedElementPtr(loc, rewriter, access,
                                              srcIndices[baseIdx]);
    if (failed(ptr))
      return false;
    groupBases.push_back(*ptr);
  }
  for (unsigned g = 0; g < numGroups; ++g) {
    Value packed =
        packLLVector(loc, ArrayRef<Value>(inVals).slice(g * vec, vec),
                     rewriter);
    targetInfo.storeDShared(rewriter, loc, groupBases[g], std::nullopt, packed,
                            pred);
  }
  return true;
}


// Exact, default-off B row/SG32 writer. Runtime M64/up identity is attested by
// the audited caller; this guard independently checks the compiler geometry,
// actual LinearLayout, physical allocation and complete address coverage.
static constexpr llvm::StringLiteral kPH1BRowShuffle =
    "musa.ph1_m64up_b_row_shuffle_vector_store";

static bool provePH1BRowShuffleGroups(
    RankedTensorType regTy, int64_t allocationOffset,
    const triton::musa::ResolvedTMESwizzleConfig &swizzle) {
  auto *ctx = regTy.getContext();
  LinearLayout layout = toLinearLayout(regTy);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");
  const auto &bases = layout.getBases();
  if (layout.getNumInDims() != 4 || layout.getNumOutDims() != 2)
    return false;
  for (auto dim : {kReg, kLane, kWarp, kBlock})
    if (!bases.contains(dim))
      return false;
  if (layout.getInDimSize(kReg) != 16 || layout.getInDimSize(kLane) != 32 ||
      layout.getInDimSize(kWarp) != 4 || layout.getInDimSize(kBlock) != 1)
    return false;
  for (auto [dim, mask] : getFreeVariableMasks(regTy))
    if (mask != 0)
      return false;
  // A's unchanged shared contract gives the raw base 4KiB alignment. Requiring
  // this allocation to share that alignment also covers the SG32 period.
  if (allocationOffset < 0 || allocationOffset > 0x7ffff000LL ||
      allocationOffset % 4096 != 0)
    return false;
  SmallVector<bool, 2048> seen(2048, false);
  for (int32_t warp = 0; warp < 4; ++warp) {
    for (int32_t lane = 0; lane < 32; ++lane) {
      int64_t groupBase = 0;
      for (int32_t reg = 0; reg < 16; ++reg) {
        auto input = layout.apply(
            {{kReg, reg}, {kLane, lane}, {kWarp, warp}, {kBlock, 0}});
        if (input.size() != 2 || input[0].first != str_attr("dim0") ||
            input[1].first != str_attr("dim1") ||
            input[0].second != 8 * (lane % 8) + reg % 8 ||
            input[1].second != lane / 8 + 4 * warp + 16 * (reg / 8))
          return false;
        // The plan exchanges packed-register bits 0/1 with lane bits 3/4.
        // Its resulting register j holds N=j, with even/odd K still packed.
        int64_t k = 8 * (lane % 8) + 2 * (lane / 8) + (reg / 4) % 2;
        int64_t n = 4 * warp + reg % 4 + 16 * (reg / 8);
        auto offset = triton::musa::linearizePH1TMELinearCoords(
            ArrayRef<int64_t>({k, n}), ArrayRef<int64_t>({64, 32}),
            ArrayRef<unsigned>({1, 0}), 2);
        if (failed(offset) || *offset != k * 32 + n)
          return false;
        int64_t addr = triton::musa::applyPH1TMESwizzleToByteAddress(
            allocationOffset + 2 * *offset, swizzle.swizzleGranularity,
            swizzle.swizzleStride, swizzle.swizzleLine);
        int64_t relative = addr - allocationOffset;
        if (relative < 0 || relative >= 4096 || relative % 2 != 0 ||
            seen[relative / 2])
          return false;
        seen[relative / 2] = true;
        if (reg % 4 == 0) {
          groupBase = addr;
          if (groupBase % 8 != 0)
            return false;
        } else if (addr != groupBase + 2 * (reg % 4)) {
          return false;
        }
      }
    }
  }
  return llvm::all_of(seen, [](bool value) { return value; });
}

static bool matchPH1BRowShuffleContract(
    MemDescType memDescTy, RankedTensorType regTy, Operation *op,
    const PH1SwizzledSharedAccess &access) {
  if (!isa<triton::gpu::LocalAllocOp>(op))
    return false;
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto enabled = module->getAttrOfType<BoolAttr>(kPH1BRowShuffle);
  if (!enabled || !enabled.getValue())
    return false;
  auto matchesInt = [&](StringRef name, int64_t value) {
    auto attr = module->getAttrOfType<IntegerAttr>(name);
    return attr && attr.getInt() == value;
  };
  auto target = module->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
  auto backendOpt = module->getAttrOfType<BoolAttr>(
      "musa.ph1_m64up_b_row_shuffle_vector_store.backend_opt");
  auto aEnabled = module->getAttrOfType<BoolAttr>(kPH1M64UpAVectorStoreMarker);
  if (!target || target.getValue() != "musa:31" ||
      !backendOpt || !backendOpt.getValue() || !aEnabled || !aEnabled.getValue() ||
      !matchesInt(triton::gpu::AttrNumCTAsName, 1) ||
      !matchesInt(triton::gpu::AttrNumWarpsName, 4) ||
      !matchesInt(triton::gpu::AttrNumThreadsPerWarp, 32) ||
      !matchesInt("musa.ph1_m64up_b_row_shuffle_vector_store.num_stages", 1) ||
      !matchesInt("musa.ph1_m64up_b_row_shuffle_vector_store.warp_size", 32))
    return false;
  auto opIdx = triton::musa::getSqmmaOpIdx(op);
  auto elemBytes = triton::musa::getSqmmaElemBytes(op);
  auto rowMajor = op->getAttrOfType<BoolAttr>(triton::musa::kSqmmaRowMajorAttr);
  if (!opIdx || *opIdx != 1 || !elemBytes || *elemBytes != 2 ||
      !rowMajor || !rowMajor.getValue())
    return false;
  if (regTy.getRank() != 2 || !regTy.getElementType().isBF16() ||
      !ph1EqualsIntList(regTy.getShape(), {64, 32}) ||
      memDescTy.getRank() != 2 || !memDescTy.getElementType().isBF16() ||
      !ph1EqualsIntList(memDescTy.getShape(), {64, 32}) ||
      !ph1EqualsIntList(access.physicalShape, {64, 32}) ||
      !ph1EqualsUnsignedList(access.order, {1u, 0u}) ||
      access.elemBytes != 2 || access.hasAffineOffset ||
      !ph1AllOnes(triton::gpu::getCTAsPerCGA(regTy.getEncoding())) ||
      !ph1AllOnes(triton::gpu::getCTAsPerCGA(memDescTy.getEncoding())))
    return false;
  auto blocked = dyn_cast<BlockedEncodingAttr>(regTy.getEncoding());
  auto shared = dyn_cast<SwizzledSharedEncodingAttr>(memDescTy.getEncoding());
  if (!blocked ||
      !ph1EqualsUnsignedList(blocked.getSizePerThread(), {8u, 1u}) ||
      !ph1EqualsUnsignedList(blocked.getThreadsPerWarp(), {8u, 4u}) ||
      !ph1EqualsUnsignedList(blocked.getWarpsPerCTA(), {1u, 4u}) ||
      !ph1EqualsUnsignedList(blocked.getOrder(), {0u, 1u}) ||
      !shared || shared.getVec() != 16 || shared.getPerPhase() != 4 ||
      shared.getMaxPhase() != 8 ||
      !ph1EqualsUnsignedList(shared.getOrder(), {1u, 0u}))
    return false;
  auto swizzle = triton::musa::resolveTMESwizzleConfigFromEncoding(memDescTy);
  auto grouping = triton::musa::getPH1TMELeadingDimGrouping(
      access.physicalShape, access.order, access.elemBytes);
  if (failed(swizzle) || failed(grouping) || grouping->numGroups != 1 ||
      grouping->elemsPerGroupInLeadingDim != 32 ||
      triton::musa::getSwizzleGranularityBytes(swizzle->swizzleGranularity) != 32 ||
      triton::musa::getSwizzleStrideBytes(swizzle->swizzleStride) != 256 ||
      triton::musa::getSwizzleLineBytes(swizzle->swizzleLine) != 256)
    return false;
  auto offset = op->getAttrOfType<IntegerAttr>("allocation.offset");
  return offset && provePH1BRowShuffleGroups(regTy, offset.getInt(), *swizzle);
}

static bool emitPH1BRowShuffleVectorStore(
    Location loc, ConversionPatternRewriter &rewriter,
    const PH1SwizzledSharedAccess &access, ArrayRef<Value> inVals,
    ArrayRef<SmallVector<Value>> srcIndices, Value pred,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  if (inVals.size() != 16 || srcIndices.size() != 16 ||
      llvm::any_of(inVals, [](Value value) { return !value.getType().isBF16(); }) ||
      llvm::any_of(srcIndices, [](ArrayRef<Value> coord) { return coord.size() != 2; }))
    return false;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // The guard proves every register/lane/warp is canonical, making the
  // redundant-thread predicate uniformly true in the supported full-warp
  // exact-up K loop. There is no lane-dependent branch around the shuffles.
  Value thread = getThreadId(rewriter, loc);
  Value lane = b.and_(thread, b.i32_val(31));
  Value warp = b.lshr(thread, b.i32_val(5));
  Value k = b.add(b.mul(b.and_(lane, b.i32_val(7)), b.i32_val(8)),
                  b.mul(b.lshr(lane, b.i32_val(3)), b.i32_val(2)));
  Value n = b.mul(warp, b.i32_val(4));
  SmallVector<Value, 4> addresses;
  for (unsigned group = 0; group < 4; ++group) {
    SmallVector<Value, 2> coord{
        b.add(k, b.i32_val(group % 2)),
        b.add(n, b.i32_val(16 * (group / 2)))};
    auto ptr = getPH1SwizzledSharedElementPtr(loc, rewriter, access, coord);
    if (failed(ptr))
      return false;
    addresses.push_back(*ptr);
  }
  // All failure exits precede any shuffle or store. Bitcasts only: no BF16
  // arithmetic, widening/narrowing conversion, or NaN canonicalization.
  std::array<Value, 8> pairs;
  for (unsigned pair = 0; pair < 8; ++pair)
    pairs[pair] = b.bitcast(packLLVector(loc, inVals.slice(2 * pair, 2), rewriter),
                           rewriter.getI32Type());
  auto result = triton::musa::transposePH1BRowPairs(
      pairs,
      [&](unsigned bit) {
        return b.icmp_eq(b.and_(lane, b.i32_val(bit)), b.i32_val(0));
      },
      [&](Value condition, Value low, Value high) {
        return b.select(condition, low, high);
      },
      [&](Value value, unsigned bit) {
        return targetInfo.shuffleXor(rewriter, loc, value, bit);
      });
  auto pairType = VectorType::get({2}, inVals[0].getType());
  for (unsigned group = 0; group < 4; ++group) {
    SmallVector<Value, 4> elements;
    for (unsigned j = 0; j < 4; ++j) {
      Value pair = b.bitcast(result[4 * (group / 2) + j], pairType);
      elements.push_back(b.extract_element(pair, b.i32_val(group % 2)));
    }
    Value packed = packLLVector(loc, elements, rewriter);
    targetInfo.storeDShared(rewriter, loc, addresses[group], std::nullopt,
                           packed, pred);
  }
  return true;
}

static LogicalResult lowerPH1SwizzledSharedStoreFromTensor(
    Location loc, SharedMemoryObject smemObj, MemDescType memDescTy,
    RankedTensorType regTy, Value llvmSrc,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const mlir::triton::MUSA::TargetInfo &targetInfo, Operation *diagnosticOp) {
  Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto access = getPH1SwizzledSharedAccess(loc, memDescTy, smemObj, llvmElemTy,
                                           rewriter, diagnosticOp);
  if (failed(access))
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto inVals = ::mlir::unpackLLElements(loc, llvmSrc, rewriter);
  auto srcIndices =
      emitIndices(loc, rewriter, targetInfo, regTy.getEncoding(), regTy, false);
  if (srcIndices.size() != inVals.size())
    return failure();

  auto freeVarMasks = getFreeVariableMasks(regTy);
  uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
  Value threadPred =
      emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
  Value pred = threadPred ? threadPred : b.true_val();
  if (auto vec = matchExactM64UpAOperandVectorContract(memDescTy, regTy,
                                                       diagnosticOp, *access)) {
    if (emitExactM64UpAOperandVectorStore(loc, rewriter, *access, inVals,
                                          srcIndices, *vec, pred, targetInfo))
      return success();
  }
  if (matchPH1BRowShuffleContract(memDescTy, regTy, diagnosticOp, *access) &&
      emitPH1BRowShuffleVectorStore(loc, rewriter, *access, inVals, srcIndices,
                                   pred, targetInfo))
    return success();
  for (auto [idx, coord] : llvm::enumerate(srcIndices)) {
    if (!isCanonicalIndex(static_cast<unsigned>(idx), regMask))
      continue;
    auto ptr = getPH1SwizzledSharedElementPtr(loc, rewriter, *access, coord);
    if (failed(ptr))
      return failure();
    targetInfo.storeDShared(rewriter, loc, *ptr, std::nullopt, inVals[idx],
                            pred);
  }
  return success();
}

static FailureOr<Value> lowerPH1SwizzledSharedLoadToTensor(
    Location loc, Value llvmMemDesc, MemDescType memDescTy,
    RankedTensorType regTy, const LLVMTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter,
    const mlir::triton::MUSA::TargetInfo &targetInfo, Operation *localLoadOp) {
  Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, llvmMemDesc,
                                                       llvmElemTy, rewriter);
  auto access = getPH1SwizzledSharedAccess(loc, memDescTy, smemObj, llvmElemTy,
                                           rewriter, localLoadOp);
  if (failed(access))
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto srcIndices =
      emitIndices(loc, rewriter, targetInfo, regTy.getEncoding(), regTy, false);
  SmallVector<Value> outVals;
  outVals.reserve(srcIndices.size());
  for (ArrayRef<Value> coord : srcIndices) {
    auto ptr = getPH1SwizzledSharedElementPtr(loc, rewriter, *access, coord);
    if (failed(ptr))
      return failure();
    outVals.push_back(targetInfo.loadDShared(rewriter, loc, *ptr, std::nullopt,
                                             llvmElemTy, b.true_val(),
                                             localLoadOp));
  }

  return ::mlir::packLLElements(loc, typeConverter, outVals, rewriter, regTy);
}

static LogicalResult lowerLocalAllocSrcToShared(
    Location loc, MLIRContext *ctx, Value regVal, MemDescType memDescTy,
    SharedMemoryObject smemObj, ArrayRef<Value> inVals,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  auto regTy = cast<RankedTensorType>(regVal.getType());
  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());

  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kOffset = str_attr("offset");
  auto regLayout = toLinearLayout(regTy);
  auto paddedEnc =
      dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(memDescTy.getEncoding());
  LinearLayout cvt = LinearLayout::empty();
  if (paddedEnc) {
    const auto &sharedLL = paddedEnc.getLinearComponent();
    cvt = regLayout.invertAndCompose(sharedLL);
  } else {
    auto sharedLayout = toLinearLayout(memDescTy);
    cvt = regLayout.invertAndCompose(sharedLayout);
  }
  auto kBlock = str_attr("block");
  if (!cvt.isTrivialOver({kBlock}))
    return failure();
  cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});
  lowerLocalLdSt(loc, ctx, cvt, inVals, llvmElemTy, memDescTy, smemObj,
                 rewriter, targetInfo);
  return success();
}

static std::optional<int>
getResidualDotOperandLocalLoadMaxVecElems(MemDescType memDescTy) {
  unsigned elemBitWidth = memDescTy.getElementTypeBitWidth();
  if (elemBitWidth == 8)
    return 1;
  return std::nullopt;
}

static SmallVector<Value>
getBlockedThreadIds(Value threadId, ArrayRef<unsigned> shapePerCTATile,
                    ArrayRef<unsigned> sizePerThread, ArrayRef<unsigned> order,
                    ConversionPatternRewriter &rewriter, Location loc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> threadIds(order.size());
  for (unsigned i = 0; i < order.size() - 1; ++i) {
    unsigned dim = order[i];
    Value dimSize = b.i32_val(
        static_cast<int32_t>(shapePerCTATile[dim] / sizePerThread[dim]));
    Value rem = b.urem(threadId, dimSize);
    threadId = b.udiv(threadId, dimSize);
    threadIds[dim] = rem;
  }
  unsigned dim = order.back();
  threadIds[dim] =
      b.urem(threadId, b.i32_val(static_cast<int32_t>(shapePerCTATile[dim] /
                                                      sizePerThread[dim])));
  return threadIds;
}

static SmallVector<unsigned> getShapePerCTATile(BlockedEncodingAttr layout) {
  SmallVector<unsigned> shapePerCTATile;
  for (auto [reg, thread, warp] :
       llvm::zip(layout.getSizePerThread(), layout.getThreadsPerWarp(),
                 layout.getWarpsPerCTA())) {
    shapePerCTATile.push_back(reg * thread * warp);
  }
  return shapePerCTATile;
}

static unsigned getShapePerCTATileForMN(BlockedEncodingAttr layout, bool isM) {
  auto order = layout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(layout);
  unsigned mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  unsigned nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  return isM ? mShapePerCTATile : nShapePerCTATile;
}

static unsigned getSizePerThreadForMN(BlockedEncodingAttr layout, bool isM) {
  auto order = layout.getOrder();
  auto sizePerThread = layout.getSizePerThread();
  unsigned mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  unsigned nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  return isM ? mSizePerThread : nSizePerThread;
}

static FailureOr<SmallVector<unsigned>>
getStaticStrides(ArrayRef<int64_t> shape, ArrayRef<unsigned> order) {
  if (shape.size() != order.size())
    return failure();

  SmallVector<unsigned> strides(shape.size());
  uint64_t stride = 1;
  for (unsigned dim : order) {
    if (dim >= shape.size() || shape[dim] <= 0 ||
        stride > std::numeric_limits<unsigned>::max())
      return failure();
    strides[dim] = static_cast<unsigned>(stride);
    stride *= static_cast<uint64_t>(shape[dim]);
  }
  return strides;
}

static unsigned product(ArrayRef<unsigned> values) {
  return std::accumulate(values.begin(), values.end(), 1u,
                         std::multiplies<unsigned>());
}

static FailureOr<Value> lowerResidualFMAOperandLoad(
    triton::gpu::LocalLoadOp op, Value llSrc, MemDescType memDescTy,
    RankedTensorType regTy, DotOperandEncodingAttr dotEnc,
    const LLVMTypeConverter *typeConverter, ConversionPatternRewriter &rewriter,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  if (memDescTy.getShape().size() != 2 || regTy.getShape().size() != 2)
    return failure();

  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto dLayout = dyn_cast<BlockedEncodingAttr>(dotEnc.getParent());
  if (!dLayout)
    return failure();

  Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, llSrc, llvmElemTy, rewriter);
  auto sharedOrder = triton::gpu::getOrder(memDescTy);
  if (sharedOrder.size() != 2)
    return failure();
  auto strides = getStaticStrides(
      memDescTy.getAllocShape().take_back(memDescTy.getRank()), sharedOrder);
  if (failed(strides))
    return failure();

  Value threadId = getThreadId(rewriter, loc);
  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = llvm::to_vector(dLayout.getSizePerThread());
  auto order = llvm::to_vector(dLayout.getOrder());
  auto threadIds = getBlockedThreadIds(threadId, shapePerCTATile, sizePerThread,
                                       order, rewriter, loc);

  auto shape = memDescTy.getShape();
  unsigned opIdx = dotEnc.getOpIdx();
  SmallVector<Value> vals;
  Value smemBase = smemObj.getShmemAffineBase(loc, rewriter, memDescTy);
  Type ptrTy = smemBase.getType();
  auto inRepOrder = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto repOrder = expandMatrixOrderWithBatch(dLayout.getRepOrder());

  if (opIdx == 0) {
    Value strideAM = b.i32_val(static_cast<int32_t>((*strides)[0]));
    Value strideAK = b.i32_val(static_cast<int32_t>((*strides)[1]));
    unsigned kExtent = static_cast<unsigned>(shape[1]);
    unsigned mExtent = static_cast<unsigned>(shape[0]);
    unsigned mShapePerCTATile = getShapePerCTATileForMN(dLayout, true);
    unsigned mSizePerThread = getSizePerThreadForMN(dLayout, true);
    unsigned mContig = mSizePerThread;
    Value threadIdM = b.urem(
        threadIds[0],
        b.i32_val(static_cast<int32_t>(llvm::divideCeil(mExtent, mContig))));
    Value threadBaseM = b.mul(threadIdM, b.i32_val(mContig));

    SmallVector<unsigned> perRepShape = {1, mSizePerThread, kExtent};
    SmallVector<unsigned> repetitions = {
        1, llvm::divideCeil(mExtent, mShapePerCTATile), 1};
    unsigned elemsPerRep = product(perRepShape);
    unsigned totalElems = elemsPerRep * product(repetitions);
    vals.reserve(totalElems);
    for (unsigned idx = 0; idx < totalElems; ++idx) {
      auto inRepIdx =
          mlir::LLVM::delinearize(idx % elemsPerRep, perRepShape, inRepOrder);
      auto repIdx =
          mlir::LLVM::delinearize(idx / elemsPerRep, repetitions, repOrder);
      Value mCoord = b.add(
          b.add(threadBaseM, b.i32_val(static_cast<int32_t>(inRepIdx[1]))),
          b.i32_val(static_cast<int32_t>(repIdx[1] * mShapePerCTATile)));
      Value kCoord = b.i32_val(static_cast<int32_t>(inRepIdx[2]));
      Value offset = b.add(b.mul(mCoord, strideAM), b.mul(kCoord, strideAK));
      Value ptr = b.gep(ptrTy, llvmElemTy, smemBase, offset);
      vals.push_back(targetInfo.loadDShared(rewriter, loc, ptr, std::nullopt,
                                            llvmElemTy, b.true_val(),
                                            op.getOperation()));
    }
  } else if (opIdx == 1) {
    Value strideBK = b.i32_val(static_cast<int32_t>((*strides)[0]));
    Value strideBN = b.i32_val(static_cast<int32_t>((*strides)[1]));
    unsigned kExtent = static_cast<unsigned>(shape[0]);
    unsigned nExtent = static_cast<unsigned>(shape[1]);
    unsigned nShapePerCTATile = getShapePerCTATileForMN(dLayout, false);
    unsigned nSizePerThread = getSizePerThreadForMN(dLayout, false);
    unsigned nContig = nSizePerThread;
    Value threadIdN = b.urem(
        threadIds[1],
        b.i32_val(static_cast<int32_t>(llvm::divideCeil(nExtent, nContig))));
    Value threadBaseN = b.mul(threadIdN, b.i32_val(nContig));

    SmallVector<unsigned> perRepShape = {1, kExtent, nSizePerThread};
    SmallVector<unsigned> repetitions = {
        1, 1, llvm::divideCeil(nExtent, nShapePerCTATile)};
    unsigned elemsPerRep = product(perRepShape);
    unsigned totalElems = elemsPerRep * product(repetitions);
    vals.reserve(totalElems);
    for (unsigned idx = 0; idx < totalElems; ++idx) {
      auto inRepIdx =
          mlir::LLVM::delinearize(idx % elemsPerRep, perRepShape, inRepOrder);
      auto repIdx =
          mlir::LLVM::delinearize(idx / elemsPerRep, repetitions, repOrder);
      Value kCoord = b.i32_val(static_cast<int32_t>(inRepIdx[1]));
      Value nCoord = b.add(
          b.add(threadBaseN, b.i32_val(static_cast<int32_t>(inRepIdx[2]))),
          b.i32_val(static_cast<int32_t>(repIdx[2] * nShapePerCTATile)));
      Value offset = b.add(b.mul(kCoord, strideBK), b.mul(nCoord, strideBN));
      Value ptr = b.gep(ptrTy, llvmElemTy, smemBase, offset);
      vals.push_back(targetInfo.loadDShared(rewriter, loc, ptr, std::nullopt,
                                            llvmElemTy, b.true_val(),
                                            op.getOperation()));
    }
  } else {
    return failure();
  }

  return packLLElements(loc, typeConverter, vals, rewriter, regTy);
}

Value maybeAnd(ConversionPatternRewriter &rewriter, Location loc, Value a,
               Value b) {
  if (!a)
    return b;
  if (!b)
    return a;
  if (matchPattern(a, m_One()))
    return b;
  if (matchPattern(b, m_One()))
    return a;
  if (matchPattern(a, m_Zero()))
    return a;
  if (matchPattern(b, m_Zero()))
    return b;

  if (a && b) {
    return TritonLLVMOpBuilder(loc, rewriter).and_(a, b);
  }
  return a ? a : b;
}

template <typename Fn>
void emitIfPredicated(RewriterBase &rewriter, Location loc, Value pred,
                      Fn &&emitFn) {
  if (!pred) {
    emitFn();
    return;
  }
  if (matchPattern(pred, m_One())) {
    emitFn();
    return;
  }

  Block *curBlock = rewriter.getInsertionBlock();
  Block *afterBlock =
      rewriter.splitBlock(curBlock, rewriter.getInsertionPoint());
  Block *thenBlock = rewriter.createBlock(afterBlock);

  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, thenBlock, afterBlock);

  rewriter.setInsertionPointToEnd(thenBlock);
  emitFn();
  LLVM::BrOp::create(rewriter, loc, afterBlock);

  rewriter.setInsertionPointToStart(afterBlock);
}

bool asyncCopyMaskNeedsPredicate(Value mask) {
  if (!mask)
    return false;

  if (matchPattern(mask, m_One()))
    return false;

  if (auto splatOp = mask.getDefiningOp<triton::SplatOp>())
    return !matchPattern(splatOp.getSrc(), m_One());

  return true;
}

Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const mlir::triton::MUSA::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred;
  int32_t laneMask = freeVarMasks.lookup(kLane);
  if (laneMask != 0) {
    Value dimPred = b.icmp_eq(b.and_(laneId, b.i32_val(laneMask)), zero);
    pred = maybeAnd(rewriter, loc, pred, dimPred);
  }
  int32_t warpMask = freeVarMasks.lookup(kWarp);
  if (warpMask != 0) {
    Value dimPred = b.icmp_eq(b.and_(warpId, b.i32_val(warpMask)), zero);
    pred = maybeAnd(rewriter, loc, pred, dimPred);
  }
  int32_t blockMask = freeVarMasks.lookup(kBlock);
  if (blockMask != 0) {
    Value dimPred = b.icmp_eq(b.and_(blockId, b.i32_val(blockMask)), zero);
    pred = maybeAnd(rewriter, loc, pred, dimPred);
  }

  return pred;
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

unsigned getVectorSize(Value ptr, ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return 1;
  unsigned contiguity = axisInfoAnalysis.getContiguity(ptr);
  unsigned pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  if (pointeeBitWidth == 0)
    return 1;
  // Keep vectorized memory operations within 128-bit lanes.
  unsigned maxVec = std::max(1u, 128u / pointeeBitWidth);
  return std::min(contiguity, maxVec);
}

StringRef getMusaMemcpyG2SIntrinsic(Type elemTy) {
  switch (getIntOrFloatOrPtrBitWidth(elemTy)) {
  case 8:
  case 16:
  case 32:
  case 64:
    return "llvm.musa.memcpy.g2s";
  default:
    return {};
  }
}

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp> {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const mlir::triton::MUSA::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisInfoAnalysis,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto *ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));

    Value ptr = op.getPtr();
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    auto ptrElems = ::mlir::unpackLLElements(loc, llPtr, rewriter);
    unsigned numElems = ptrElems.size();

    unsigned vec = getVectorSize(ptr, axisInfoAnalysis);
    vec = std::min(vec, numElems);

    bool forceScalarSharedByteLoad = false;
    if (!ptrElems.empty()) {
      if (auto llPtrTy =
              dyn_cast<LLVM::LLVMPointerType>(ptrElems.front().getType())) {
        constexpr unsigned kSharedAddrSpace = 3;
        forceScalarSharedByteLoad =
            llPtrTy.getAddressSpace() == kSharedAddrSpace &&
            valueElemTy.getIntOrFloatBitWidth() == 8;
      }
    }

    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min(vec, axisInfoAnalysis.getMaskAlignment(op.getMask()));
      maskElems = ::mlir::unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    SmallVector<Value> otherElems;
    if (llOther) {
      otherElems = ::mlir::unpackLLElements(loc, llOther, rewriter);
      assert(otherElems.size() == numElems);
    }

    while (vec > 1 && (numElems % vec != 0)) {
      vec /= 2;
    }
    vec = std::max(1u, vec);
    if (forceScalarSharedByteLoad)
      vec = 1;

    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));
    bool useInplaceLoad = op->hasAttr(kInplaceLoadAttr);

    SmallVector<Value> loaded;
    loaded.reserve(numElems);

    if (vec == 1) {
      for (unsigned i = 0; i < numElems; ++i) {
        if (auto canonical = getCanonicalIndex(i, regMask); i != canonical) {
          loaded.push_back(loaded[canonical]);
          continue;
        }

        Value pred = b.true_val();
        if (!maskElems.empty())
          pred = maybeAnd(rewriter, loc, pred, maskElems[i]);
        Value falseVal =
            otherElems.empty()
                ? LLVM::ConstantOp::create(rewriter, loc, valueElemTy,
                                           rewriter.getZeroAttr(valueElemTy))
                : otherElems[i];
        Value val = useInplaceLoad
                        ? LLVM::MUSA::llInplaceLoad(rewriter, loc, ptrElems[i],
                                                    valueElemTy, pred, falseVal)
                        : LLVM::MUSA::llLoad(rewriter, loc, ptrElems[i],
                                             valueElemTy, pred, falseVal);
        loaded.push_back(val);
      }
    } else {
      auto vecTy = LLVM::getVectorType(valueElemTy, vec);
      for (unsigned vecStart = 0; vecStart < numElems; vecStart += vec) {
        unsigned canonicalVecStart = getCanonicalIndex(vecStart, regMask);
        if (vecStart != canonicalVecStart) {
          for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx)
            loaded.push_back(loaded[canonicalVecStart + elemIdx]);
          continue;
        }

        Value pred = b.true_val();
        if (!maskElems.empty())
          pred = maybeAnd(rewriter, loc, pred, maskElems[vecStart]);

        Value falseVal;
        if (otherElems.empty()) {
          auto zeroAttr = rewriter.getZeroAttr(valueElemTy);
          auto dense =
              DenseElementsAttr::get(cast<ShapedType>(vecTy), zeroAttr);
          falseVal = LLVM::ConstantOp::create(rewriter, loc, vecTy, dense);
        } else {
          falseVal = LLVM::UndefOp::create(rewriter, loc, vecTy);
          for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx) {
            falseVal = b.insert_element(vecTy, falseVal,
                                        otherElems[vecStart + elemIdx],
                                        b.i32_val(elemIdx));
          }
        }

        Value vecVal =
            useInplaceLoad
                ? LLVM::MUSA::llInplaceLoad(rewriter, loc, ptrElems[vecStart],
                                            vecTy, pred, falseVal)
                : LLVM::MUSA::llLoad(rewriter, loc, ptrElems[vecStart], vecTy,
                                     pred, falseVal);
        for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx) {
          loaded.push_back(
              b.extract_element(valueElemTy, vecVal, b.i32_val(elemIdx)));
        }
      }
    }

    Value packed = ::mlir::packLLElements(loc, typeConverter, loaded, rewriter,
                                          op.getType());
    rewriter.replaceOp(op, packed);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp> {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const mlir::triton::MUSA::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getValue();
    Value llMask = adaptor.getMask();

    auto ptrElems = ::mlir::unpackLLElements(loc, llPtr, rewriter);
    auto valElems = ::mlir::unpackLLElements(loc, llVal, rewriter);
    unsigned numElems = ptrElems.size();
    assert(numElems == valElems.size());

    unsigned vec = getVectorSize(ptr, axisInfoAnalysis);
    vec = std::min(vec, numElems);

    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = ::mlir::unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
      vec = std::min(vec, axisInfoAnalysis.getMaskAlignment(op.getMask()));
    }

    while (vec > 1 && (numElems % vec != 0)) {
      vec /= 2;
    }
    vec = std::max(1u, vec);

    auto *ctx = rewriter.getContext();
    auto freeVarMasks = getFreeVariableMasks(ptr.getType());
#ifdef __TLE__
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    if (!isa<RankedTensorType>(ptr.getType())) {
      auto partitions =
          op->getParentOfType<triton::gpu::WarpSpecializePartitionsOp>();
      auto ws = partitions ? dyn_cast<triton::gpu::WarpSpecializeOp>(
                                 partitions->getParentOp())
                           : triton::gpu::WarpSpecializeOp();
      if (ws && ws->hasAttr("musa_tle.static_warp_specialize")) {
        ModuleOp module = op->getParentOfType<ModuleOp>();
        auto numWarps =
            module->getAttrOfType<IntegerAttr>(triton::gpu::AttrNumWarpsName);
        auto threadsPerWarp = module->getAttrOfType<IntegerAttr>(
            triton::gpu::AttrNumThreadsPerWarp);
        if (!numWarps || !threadsPerWarp || numWarps.getInt() <= 0 ||
            threadsPerWarp.getInt() <= 0) {
          return op.emitOpError(
              "static WS partition store requires ttg.num-warps and "
              "ttg.threads-per-warp");
        }
        if (numWarps.getInt() >
            std::numeric_limits<int32_t>::max() / threadsPerWarp.getInt())
          return op.emitOpError(
              "static WS partition leader exceeds int32 range");
        int64_t firstPartitionThread =
            numWarps.getInt() * threadsPerWarp.getInt();
        Value rawThreadId = ::mlir::gpu::ThreadIdOp::create(
            rewriter, loc, ::mlir::gpu::Dimension::x);
        rawThreadId =
            arith::IndexCastOp::create(rewriter, loc, i32_ty, rawThreadId);
        threadPred = b.icmp_eq(
            rawThreadId, b.i32_val(static_cast<int32_t>(firstPartitionThread)));
      }
    }
#else
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
#endif // __TLE__
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    if (vec == 1) {
      for (unsigned i = 0; i < numElems; ++i) {
        if (!isCanonicalIndex(i, regMask))
          continue;
        Value pred = threadPred ? threadPred : b.true_val();
        if (!maskElems.empty()) {
          pred = maybeAnd(rewriter, loc, pred, maskElems[i]);
        }
        LLVM::MUSA::llStore(rewriter, loc, ptrElems[i], valElems[i], pred);
      }
      rewriter.eraseOp(op);
      return success();
    }

    Type valueElemTy =
        getTypeConverter()->convertType(getElementTypeOrSelf(op.getValue()));
    auto vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (unsigned vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask))
        continue;

      Value pred = threadPred ? threadPred : b.true_val();
      if (!maskElems.empty()) {
        pred = maybeAnd(rewriter, loc, pred, maskElems[vecStart]);
      }

      Value storeVal = LLVM::UndefOp::create(rewriter, loc, vecTy);
      for (unsigned elemIdx = 0; elemIdx < vec; ++elemIdx) {
        storeVal = b.insert_element(
            vecTy, storeVal, valElems[vecStart + elemIdx], b.i32_val(elemIdx));
      }
      LLVM::MUSA::llStore(rewriter, loc, ptrElems[vecStart], storeVal, pred);
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp> {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const mlir::triton::MUSA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisInfoAnalysis,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicCASOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  Value convertToAtomicType(Location loc, Value val, Type elemTy,
                            ConversionPatternRewriter &rewriter) const {
    if (elemTy.isIntOrIndex()) {
      return val;
    }
    auto intTy = rewriter.getIntegerType(elemTy.getIntOrFloatBitWidth());
    return LLVM::BitcastOp::create(rewriter, loc, intTy, val);
  }

  Value convertFromAtomicType(Location loc, Value val, Type elemTy,
                              ConversionPatternRewriter &rewriter) const {
    if (elemTy.isIntOrIndex()) {
      return val;
    }
    return LLVM::BitcastOp::create(rewriter, loc, elemTy, val);
  }

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto ptrElements =
        ::mlir::unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto cmpElements =
        ::mlir::unpackLLElements(loc, adaptor.getCmp(), rewriter);
    auto valElements =
        ::mlir::unpackLLElements(loc, adaptor.getVal(), rewriter);

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    auto atomicOrdering =
        getMemoryOrdering(op.getSem()).value_or(LLVM::AtomicOrdering::acq_rel);
    auto successOrdering = atomicOrdering;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    Value storePred = threadPred ? threadPred : b.true_val();
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    auto emitPredicated = [&](Value pred, Type retTy,
                              auto emitAtomic) -> Value {
      if (!pred) {
        return emitAtomic();
      }
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument(retTy, loc);

      rewriter.setInsertionPointToEnd(curBlock);
      Value undefVal = LLVM::UndefOp::create(rewriter, loc, retTy);
      LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock,
                             undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      Value atom = emitAtomic();
      LLVM::BrOp::create(rewriter, loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      return endBlock->getArgument(0);
    };

    if (!tensorTy) {
      Value casPtr = ptrElements.front();
      Value casCmp = cmpElements.front();
      Value casVal = valElements.front();
      auto atomicCmp = convertToAtomicType(loc, casCmp, valueElemTy, rewriter);
      auto atomicVal = convertToAtomicType(loc, casVal, valueElemTy, rewriter);
      Type atomicTy = atomicCmp.getType();

      Value retVal = emitPredicated(threadPred, valueElemTy, [&]() {
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, atomicCmp, atomicVal, successOrdering,
            failureOrdering);
        Value old = b.extract_val(atomicTy, cmpxchg, 0);
        return convertFromAtomicType(loc, old, valueElemTy, rewriter);
      });

      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      if (!op->hasAttr("allocation.offset")) {
        return rewriter.notifyMatchFailure(
            op, "missing allocation.offset for scalar atomic result");
      }

      Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
      targetInfo.storeDShared(rewriter, loc, atomPtr, std::nullopt, retVal,
                              storePred);
      b.barrier(triton::gpu::AddrSpace::Local);
      Value ret = b.load(valueElemTy, atomPtr);
      rewriter.replaceOp(op, {ret});
      return success();
    }

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; ++i) {
      if (auto canonical = getCanonicalIndex(i, regMask); canonical != i) {
        resultVals[i] = resultVals[canonical];
        continue;
      }
      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      Value casVal = valElements[i];
      auto atomicCmp = convertToAtomicType(loc, casCmp, valueElemTy, rewriter);
      auto atomicVal = convertToAtomicType(loc, casVal, valueElemTy, rewriter);
      Type atomicTy = atomicCmp.getType();

      Value oldVal = emitPredicated(threadPred, valueElemTy, [&]() {
        auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
            rewriter, loc, casPtr, atomicCmp, atomicVal, successOrdering,
            failureOrdering);
        Value old = b.extract_val(atomicTy, cmpxchg, 0);
        return convertFromAtomicType(loc, old, valueElemTy, rewriter);
      });
      resultVals[i] = oldVal;
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, storePred, targetInfo, getTypeConverter());
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp> {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const mlir::triton::MUSA::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisInfoAnalysis,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::AtomicRMWOp>(converter, benefit),
        targetInfo(targetInfo), axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();
    auto maybeKind = matchAtomicOp(atomicRmwAttr);
    if (!maybeKind) {
      return rewriter.notifyMatchFailure(op, "unsupported atomic op");
    }
    auto atomicOrdering =
        getMemoryOrdering(op.getSem()).value_or(LLVM::AtomicOrdering::acq_rel);

    auto ptrElements =
        ::mlir::unpackLLElements(loc, adaptor.getPtr(), rewriter);
    auto valElements =
        ::mlir::unpackLLElements(loc, adaptor.getVal(), rewriter);
    SmallVector<Value> maskElements;
    if (adaptor.getMask()) {
      maskElements = ::mlir::unpackLLElements(loc, adaptor.getMask(), rewriter);
    }

    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    Value storePred = threadPred ? threadPred : b.true_val();
    uint32_t regMask = freeVarMasks.lookup(str_attr("reg"));

    auto emitPredicated = [&](Value pred, Type retTy,
                              auto emitAtomic) -> Value {
      if (!pred) {
        return emitAtomic();
      }
      auto *curBlock = rewriter.getInsertionBlock();
      auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
      auto *atomicBlock = rewriter.createBlock(
          curBlock->getParent(), std::next(Region::iterator(curBlock)));
      endBlock->addArgument(retTy, loc);

      rewriter.setInsertionPointToEnd(curBlock);
      Value undefVal = LLVM::UndefOp::create(rewriter, loc, retTy);
      LLVM::CondBrOp::create(rewriter, loc, pred, atomicBlock, endBlock,
                             undefVal);

      rewriter.setInsertionPointToEnd(atomicBlock);
      Value atom = emitAtomic();
      LLVM::BrOp::create(rewriter, loc, atom, endBlock);

      rewriter.setInsertionPointToStart(endBlock);
      return endBlock->getArgument(0);
    };

    if (!tensorTy) {
      Value rmwPtr = ptrElements.front();
      Value rmwVal = valElements.front();
      Value rmwMask =
          maskElements.empty() ? b.true_val() : maskElements.front();
      rmwMask = maybeAnd(rewriter, loc, rmwMask, threadPred);

      auto emitAtomic = [&]() -> Value {
        if (*maybeKind == LLVM::AtomicBinOp::fadd &&
            (valueElemTy.isF16() || valueElemTy.isF32() ||
             valueElemTy.isF64())) {
          StringRef funcName;
          Type fpType = valueElemTy;
          if (valueElemTy.isF16()) {
            funcName = "__mt_atomicAdd_f16";
          } else if (valueElemTy.isF32()) {
            funcName = "__mt_atomicAdd_f32";
          } else {
            funcName = "__mt_atomicAdd_f64";
          }
          auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
          auto funcType = LLVM::LLVMFunctionType::get(fpType, {ptrTy, fpType});
          LLVM::LLVMFuncOp funcOp =
              appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
          Value addressCast =
              LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, rmwPtr);
          return LLVM::CallOp::create(rewriter, loc, funcOp,
                                      ValueRange{addressCast, rmwVal})
              .getResult();
        }
        return LLVM::AtomicRMWOp::create(rewriter, loc, *maybeKind, rmwPtr,
                                         rmwVal, atomicOrdering)
            .getResult();
      };

      Value retVal = emitPredicated(rmwMask, valueElemTy, emitAtomic);

      if (op.getResult().use_empty()) {
        rewriter.eraseOp(op);
        return success();
      }

      if (!op->hasAttr("allocation.offset")) {
        return rewriter.notifyMatchFailure(
            op, "missing allocation.offset for scalar atomic result");
      }

      auto *ctx = rewriter.getContext();
      Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                op.getOperation());
      atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
      targetInfo.storeDShared(rewriter, loc, atomPtr, std::nullopt, retVal,
                              rmwMask);
      b.barrier(triton::gpu::AddrSpace::Local);
      Value ret = b.load(valueElemTy, atomPtr);
      rewriter.replaceOp(op, {ret});
      return success();
    }

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; ++i) {
      if (auto canonical = getCanonicalIndex(i, regMask); canonical != i) {
        resultVals[i] = resultVals[canonical];
        continue;
      }
      Value rmwPtr = ptrElements[i];
      Value rmwVal = valElements[i];
      Value rmwMask = maskElements.empty() ? b.true_val() : maskElements[i];
      rmwMask = maybeAnd(rewriter, loc, rmwMask, threadPred);

      auto emitAtomic = [&]() -> Value {
        if (*maybeKind == LLVM::AtomicBinOp::fadd &&
            (valueElemTy.isF16() || valueElemTy.isF32() ||
             valueElemTy.isF64())) {
          StringRef funcName;
          Type fpType = valueElemTy;
          if (valueElemTy.isF16()) {
            funcName = "__mt_atomicAdd_f16";
          } else if (valueElemTy.isF32()) {
            funcName = "__mt_atomicAdd_f32";
          } else {
            funcName = "__mt_atomicAdd_f64";
          }
          auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
          auto funcType = LLVM::LLVMFunctionType::get(fpType, {ptrTy, fpType});
          LLVM::LLVMFuncOp funcOp =
              appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
          Value addressCast =
              LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrTy, rmwPtr);
          return LLVM::CallOp::create(rewriter, loc, funcOp,
                                      ValueRange{addressCast, rmwVal})
              .getResult();
        }
        return LLVM::AtomicRMWOp::create(rewriter, loc, *maybeKind, rmwPtr,
                                         rmwVal, atomicOrdering)
            .getResult();
      };

      Value atom = emitPredicated(rmwMask, valueElemTy, emitAtomic);
      resultVals[i] = atom;
    }

    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, storePred, targetInfo, getTypeConverter());
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct SqmmaLocalAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp> {
  SqmmaLocalAllocOpConversion(LLVMTypeConverter &converter,
                              const mlir::triton::MUSA::TargetInfo &targetInfo,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalAllocOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isSharedMemoryAlloc() || !op.getSrc())
      return failure();

    bool isSqmma = triton::musa::hasSqmmaOpIdxAttr(op.getOperation());
    auto srcTensorTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    bool isSqmmaAccumulatorSpill =
        !isSqmma && srcTensorTy &&
        isa<triton::gpu::MUSASqmmaEncodingAttr>(srcTensorTy.getEncoding());

    bool needsPH1SwizzledAccess =
        requiresAbsoluteSwizzledSharedAccess(cast<MemDescType>(op.getType()));
    if (!needsPH1SwizzledAccess && !isSqmmaAccumulatorSpill)
      return failure();
    Location loc = op.getLoc();
    auto memDescTy = cast<MemDescType>(op.getType());
    auto llvmElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());

    Value smemBase =
        LLVM::getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
    auto smemObj = SharedMemoryObject(smemBase, llvmElemTy, memDescTy.getRank(),
                                      loc, rewriter);

    if (isSqmmaAccumulatorSpill) {
      auto *ctx = op.getContext();
      auto inVals = ::mlir::unpackLLElements(loc, adaptor.getSrc(), rewriter);
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
      if (failed(lowerLocalAllocSrcToShared(loc, ctx, op.getSrc(), memDescTy,
                                            smemObj, inVals, getTypeConverter(),
                                            rewriter, targetInfo))) {
        return failure();
      }
      auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
      rewriter.replaceOp(op, retVal);
      return success();
    }

    auto regTy = cast<RankedTensorType>(op.getSrc().getType());

    if (!needsPH1SwizzledAccess || regTy.getRank() != memDescTy.getRank())
      return failure();

    if (isSqmma) {
      auto opIdx = triton::musa::getSqmmaOpIdx(op.getOperation());
      if (!opIdx)
        return failure();
      if (static_cast<unsigned>(*opIdx) == 0)
        targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    }

    if (failed(lowerPH1SwizzledSharedStoreFromTensor(
            loc, smemObj, memDescTy, regTy, adaptor.getSrc(),
            getTypeConverter(), rewriter, targetInfo, op.getOperation())))
      return failure();

    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct DotOperandLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
  DotOperandLocalLoadOpConversion(
      LLVMTypeConverter &converter,
      const mlir::triton::MUSA::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Keep SQMMA local_load paths on the generic lowering.
    if (triton::musa::hasSqmmaOpIdxAttr(op.getOperation()))
      return failure();

    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    auto memDescTy = dyn_cast<MemDescType>(op.getSrc().getType());
    if (!memDescTy)
      return failure();
    if (requiresAbsoluteSwizzledSharedAccess(memDescTy))
      return failure();

    auto regTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!regTy)
      return failure();
    auto dotEnc = dyn_cast<DotOperandEncodingAttr>(regTy.getEncoding());
    if (!dotEnc || !isa<BlockedEncodingAttr>(dotEnc.getParent()))
      return failure();

    auto typeConverter = getTypeConverter();
    auto fmaLoad = lowerResidualFMAOperandLoad(op, adaptor.getSrc(), memDescTy,
                                               regTy, dotEnc, typeConverter,
                                               rewriter, targetInfo);
    if (succeeded(fmaLoad)) {
      rewriter.replaceOp(op, *fmaLoad);
      return success();
    }

    std::optional<int> maxVecElems =
        getResidualDotOperandLocalLoadMaxVecElems(memDescTy);
    if (!maxVecElems)
      return failure();

    auto sharedEnc =
        dyn_cast<triton::gpu::SharedEncodingTrait>(memDescTy.getEncoding());
    if (!sharedEnc)
      return failure();

    Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         llvmElemTy, rewriter);

    auto kReg = str_attr("register");
    auto kLane = str_attr("lane");
    auto kWarp = str_attr("warp");
    auto kOffset = str_attr("offset");
    auto kBlock = str_attr("block");
    auto regLayout = toLinearLayout(regTy);

    LinearLayout cvt = LinearLayout::empty();
    if (auto paddedEnc =
            dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc)) {
      const auto &sharedLL = paddedEnc.getLinearComponent();
      cvt = regLayout.invertAndCompose(sharedLL);
    } else {
      auto sharedLayout = toLinearLayout(memDescTy);
      cvt = regLayout.invertAndCompose(sharedLayout);
    }
    if (!cvt.isTrivialOver({kBlock}))
      return failure();
    cvt = cvt.sublayout({kReg, kLane, kWarp}, {kOffset});

    Value affineOffset = smemObj.getShmemOffset(loc, rewriter, memDescTy);
    uint64_t maskSpanAffineOffset = smemObj.getMaskSpanOffsets(memDescTy);
    SmallVector<std::pair<unsigned, unsigned>> paddingShifts;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            memDescTy.getEncoding())) {
      auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
      paddingShifts = getPaddedSharedShifts(paddedEnc, bitwidth,
                                            /*offsetInBytes=*/true);
    }

    SmallVector<Value> outVals = lowerLdStShared(
        loc, ctx, cvt, /*valsArray=*/{}, llvmElemTy, smemObj.getBase(),
        paddingShifts, affineOffset, maskSpanAffineOffset, rewriter, targetInfo,
        maxVecElems, op.getOperation());
    Value result =
        ::mlir::packLLElements(loc, typeConverter, outVals, rewriter, regTy);
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct PH1SwizzledLocalLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp> {
  PH1SwizzledLocalLoadOpConversion(
      LLVMTypeConverter &converter,
      const mlir::triton::MUSA::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalLoadOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memDescTy = dyn_cast<MemDescType>(op.getSrc().getType());
    auto regTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!memDescTy || !regTy ||
        !requiresAbsoluteSwizzledSharedAccess(memDescTy))
      return failure();

    auto result = lowerPH1SwizzledSharedLoadToTensor(
        op.getLoc(), adaptor.getSrc(), memDescTy, regTy, getTypeConverter(),
        rewriter, targetInfo, op.getOperation());
    if (failed(result))
      return failure();

    rewriter.replaceOp(op, *result);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct PH1SwizzledLocalStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp> {
  PH1SwizzledLocalStoreOpConversion(
      LLVMTypeConverter &converter,
      const mlir::triton::MUSA::TargetInfo &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::LocalStoreOp>(converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto memDescTy = dyn_cast<MemDescType>(op.getDst().getType());
    auto regTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!memDescTy || !regTy ||
        !requiresAbsoluteSwizzledSharedAccess(memDescTy))
      return failure();

    Type llvmElemTy =
        getTypeConverter()->convertType(memDescTy.getElementType());
    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        op.getLoc(), adaptor.getDst(), llvmElemTy, rewriter);
    if (failed(lowerPH1SwizzledSharedStoreFromTensor(
            op.getLoc(), smemObj, memDescTy, regTy, adaptor.getSrc(),
            getTypeConverter(), rewriter, targetInfo, op.getOperation())))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp> {
  AsyncCopyGlobalToLocalOpConversion(
      LLVMTypeConverter &converter,
      const mlir::triton::MUSA::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo),
        axisInfoAnalysis(axisInfoAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = cast<MemDescType>(op.getResult().getType());
    auto llvmElemTy = getTypeConverter()->convertType(dstTy.getElementType());
    StringRef memcpyIntrinsic = getMusaMemcpyG2SIntrinsic(llvmElemTy);
    if (memcpyIntrinsic.empty())
      return op.emitError("async_copy unsupported element bitwidth for "
                          "llvm.musa.memcpy.g2s.* intrinsic");

    Value llDst = adaptor.getResult();
    Value llSrc = adaptor.getSrc();
    Value llMask = adaptor.getMask();

    auto srcElems = ::mlir::unpackLLElements(loc, llSrc, rewriter);
    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = ::mlir::unpackLLElements(loc, llMask, rewriter);

    auto ptrTy = srcElems.front().getType();
    auto structTy =
        LLVM::LLVMStructType::getLiteral(ctx, ArrayRef<Type>{ptrTy, i1_ty});

    SmallVector<Value> vals;
    vals.reserve(srcElems.size());
    for (size_t i = 0; i < srcElems.size(); ++i) {
      Value packed = LLVM::UndefOp::create(rewriter, loc, structTy);
      packed = b.insert_val(packed, srcElems[i], 0);
      Value maskElem = llMask ? maskElems[i] : b.true_val();
      packed = b.insert_val(packed, maskElem, 1);
      vals.push_back(packed);
    }

    auto srcLayout = toLinearLayout(srcTy);
    auto removeBroadcastSrc = actionRemoveBroadcastedRegs(srcLayout);
    srcLayout = removeBroadcastSrc.apply(srcLayout);
    vals = removeBroadcastSrc.apply(vals);

    unsigned maxVec = getVectorSize(op.getSrc(), axisInfoAnalysis);
    if (op.getMask())
      maxVec =
          std::min(maxVec, axisInfoAnalysis.getMaskAlignment(op.getMask()));
    maxVec = std::max(maxVec, op.getContiguity());
    maxVec = std::max(1u, maxVec);
    int vecBytes = maxVec * llvmElemTy.getIntOrFloatBitWidth() / 8;
    if (vecBytes < 4) {
      return op.emitError(
                 "async_copy does not support transfers smaller than 4 bytes")
             << "; calculated " << vecBytes << " bytes";
    }

    Value threadPred = b.true_val();

    bool usePred = asyncCopyMaskNeedsPredicate(op.getMask());

    if (requiresAbsoluteSwizzledSharedAccess(dstTy)) {
      auto sharedEnc = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
          dstTy.getEncoding());
      if (!sharedEnc)
        return op.emitError(
            "PH1 swizzled async_copy expects swizzled shared destination");

      auto srcIndices =
          emitIndices(loc, rewriter, targetInfo, srcTy.getEncoding(), srcTy,
                      /*withCTAOffset=*/false);
      if (srcIndices.size() != srcElems.size())
        return failure();

      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, llDst, llvmElemTy, rewriter);
      auto access = getPH1SwizzledSharedAccess(loc, dstTy, smemObj, llvmElemTy,
                                               rewriter, op.getOperation());
      if (failed(access))
        return failure();

      unsigned inVec = std::max<unsigned>(1, op.getContiguity());
      unsigned outVec = sharedEnc.getVec();
      unsigned minVec = inVec;
      if (outVec > 1)
        minVec = std::min(outVec, inVec);
      unsigned numElems = getTotalElemsPerThread(srcTy);

      for (unsigned elemIdx = 0; elemIdx < numElems;) {
        unsigned vecElems = std::min(minVec, numElems - elemIdx);
        auto idx = srcIndices[elemIdx];
        auto basePtr =
            getPH1SwizzledSharedElementPtr(loc, rewriter, *access, idx);
        if (failed(basePtr))
          return failure();

        auto maxBitWidth =
            std::max<unsigned>(128, llvmElemTy.getIntOrFloatBitWidth());
        auto vecBitWidth = llvmElemTy.getIntOrFloatBitWidth() * vecElems;
        auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
        auto numWords = vecBitWidth / bitWidth;
        auto numWordElems = bitWidth / llvmElemTy.getIntOrFloatBitWidth();
        auto byteWidth = bitWidth / 8;

        for (unsigned wordIdx = 0; wordIdx < numWords; ++wordIdx) {
          unsigned wordElemIdx = wordIdx * numWordElems;
          Value packedVal = vals[elemIdx + wordElemIdx];
          Value srcPtr = b.extract_val(ptrTy, packedVal, 0);
          Value maskElem = b.extract_val(i1_ty, packedVal, 1);
          Value dst = b.gep(basePtr->getType(), llvmElemTy, *basePtr,
                            b.i32_val(static_cast<int32_t>(wordElemIdx)));
          Value copyPred = b.true_val();
          if (usePred) {
            Value notMask = b.xor_(maskElem, b.true_val());
            Value zeroPred = notMask;
            Value zeroElem = LLVM::ConstantOp::create(
                rewriter, loc, llvmElemTy, rewriter.getZeroAttr(llvmElemTy));
            auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
            Value dstBase = b.bitcast(dst, elemPtrTy);
            for (unsigned elem = 0; elem < numWordElems; ++elem) {
              Value dstPtr =
                  b.gep(elemPtrTy, llvmElemTy, dstBase, b.i32_val(elem));
              LLVM::MUSA::llStore(rewriter, loc, dstPtr, zeroElem, zeroPred);
            }
            copyPred = b.and_(copyPred, maskElem);
          }

          Value cpSize = b.i32_val(static_cast<int32_t>(byteWidth));
          Value prefetchSize = b.i32_val(0);
          emitIfPredicated(rewriter, loc, copyPred, [&]() {
            auto funcType = LLVM::LLVMFunctionType::get(
                void_ty(ctx),
                {dst.getType(), srcPtr.getType(), cpSize.getType(),
                 prefetchSize.getType()},
                /*isVarArg=*/false);
            auto funcOp = appendOrGetExternFuncOp(rewriter, op, memcpyIntrinsic,
                                                  funcType);
            LLVM::CallOp::create(rewriter, loc, funcOp,
                                 ValueRange{dst, srcPtr, cpSize, prefetchSize});
          });
        }
        elemIdx += vecElems;
      }

      rewriter.replaceOp(op, b.i32_val(0));
      return success();
    }

    auto emitAsyncCopy = [&](RewriterBase &rewriter, Location emitLoc,
                             ArrayRef<Value> values, Value shmemAddr,
                             int startIdx,
                             VectorType vecTy) -> SmallVector<Value> {
      auto tb = TritonLLVMOpBuilder(emitLoc, rewriter);
      unsigned elemsPerVec = vecTy.getNumElements();
      unsigned byteWidth = elemsPerVec * llvmElemTy.getIntOrFloatBitWidth() / 8;

      Value packedVal = values[startIdx];
      Value srcPtr = tb.extract_val(ptrTy, packedVal, 0);
      Value maskElem = tb.extract_val(i1_ty, packedVal, 1);
      Value copyPred = threadPred ? threadPred : tb.true_val();

      if (usePred) {
        Value notMask = tb.xor_(maskElem, tb.true_val());
        Value zeroPred = threadPred ? tb.and_(threadPred, notMask) : notMask;
        Value zeroElem = LLVM::ConstantOp::create(
            rewriter, emitLoc, llvmElemTy, rewriter.getZeroAttr(llvmElemTy));
        auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
        Value dstBase = tb.bitcast(shmemAddr, elemPtrTy);
        for (unsigned elem = 0; elem < elemsPerVec; ++elem) {
          Value dstPtr =
              tb.gep(elemPtrTy, llvmElemTy, dstBase, tb.i32_val(elem));
          LLVM::MUSA::llStore(rewriter, emitLoc, dstPtr, zeroElem, zeroPred);
        }
        copyPred = tb.and_(copyPred, maskElem);
      }

      Value cpSize = tb.i32_val(byteWidth);
      Value prefetchSize = tb.i32_val(0);
      emitIfPredicated(rewriter, emitLoc, copyPred, [&]() {
        auto funcType = LLVM::LLVMFunctionType::get(
            void_ty(ctx),
            {shmemAddr.getType(), srcPtr.getType(), cpSize.getType(),
             prefetchSize.getType()},
            /*isVarArg=*/false);
        auto funcOp =
            appendOrGetExternFuncOp(rewriter, op, memcpyIntrinsic, funcType);
        LLVM::CallOp::create(
            rewriter, emitLoc, funcOp,
            ValueRange{shmemAddr, srcPtr, cpSize, prefetchSize});
      });
      return {};
    };

    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, llDst, llvmElemTy, rewriter);
    auto smemLayout = toLinearLayout(dstTy);
    auto cvt = srcLayout.invertAndCompose(smemLayout);
    if (!cvt.isTrivialOver({str_attr("block")}))
      return op.emitError(
          "async_copy does not support non-trivial block dimension");
    cvt = cvt.sublayout(
        {str_attr("register"), str_attr("lane"), str_attr("warp")},
        {str_attr("offset")});

    Value affineOffset = smemObj.getShmemOffset(loc, rewriter, dstTy);
    uint64_t maskSpanAffineOffset = smemObj.getMaskSpanOffsets(dstTy);
    std::optional<int> maybeMaxVecElems;
    SmallVector<std::pair<unsigned, unsigned>> paddingShifts;
    if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      maybeMaxVecElems = paddedEnc.getMinInterval();
      auto bitwidth = getIntOrFloatOrPtrBitWidth(llvmElemTy);
      paddingShifts =
          getPaddedSharedShifts(paddedEnc, bitwidth, /*offsetInBytes=*/true);
    }

    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    lowerLdSt(loc, ctx, cvt, vals, llvmElemTy, smemObj.getBase(), paddingShifts,
              affineOffset, maskSpanAffineOffset, laneId, warpId, rewriter,
              targetInfo, maxVec, emitAsyncCopy);

    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisInfoAnalysis;
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

struct AsyncWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncWaitOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::AsyncWaitOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto waitFnTy = LLVM::LLVMFunctionType::get(void_ty(ctx), {}, false);
    auto waitFn = appendOrGetExternFuncOp(
        rewriter, op, "llvm.musa.memcpy.g2s.wait", waitFnTy);
    LLVM::CallOp::create(rewriter, loc, waitFn, ValueRange{});
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    int /*computeCapability*/, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit) {
  patterns.add<SqmmaLocalAllocOpConversion>(
      typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 2));
  patterns.add<DotOperandLocalLoadOpConversion>(
      typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 3));
  patterns
      .add<PH1SwizzledLocalLoadOpConversion, PH1SwizzledLocalStoreOpConversion>(
          typeConverter, targetInfo, PatternBenefit(benefit.getBenefit() + 2));
  patterns.add<LoadOpConversion, StoreOpConversion, AtomicCASOpConversion,
               AtomicRMWOpConversion, AsyncCopyGlobalToLocalOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
  patterns.add<AsyncCommitGroupOpConversion, AsyncWaitOpConversion>(
      typeConverter, benefit);
}
