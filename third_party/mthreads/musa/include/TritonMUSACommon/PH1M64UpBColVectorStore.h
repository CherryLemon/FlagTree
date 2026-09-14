#ifndef TRITONMUSA_COMMON_PH1_M64UP_B_COL_VECTOR_STORE_H
#define TRITONMUSA_COMMON_PH1_M64UP_B_COL_VECTOR_STORE_H

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>

namespace mlir::triton::musa {

// Experimental caller opt-in only. The caller must select the M64 up arm;
// tensor tiles alone cannot recover the runtime token count or the MoE arm.
inline constexpr llvm::StringLiteral kPH1M64UpBColVectorStore =
    "musa.ph1_m64up_b_col_vector_store";
inline constexpr llvm::StringLiteral kPH1M64UpBColVectorStoreStages =
    "musa.ph1_m64up_b_col_vector_store.num_stages";
inline constexpr llvm::StringLiteral kPH1M64UpBColVectorStoreBackendOpt =
    "musa.ph1_m64up_b_col_vector_store.backend_opt";
inline constexpr llvm::StringLiteral kPH1M64UpBColVectorStoreSelected =
    "musa.ph1_m64up_b_col_vector_store.selected";

inline bool isPH1M64UpBColVectorStoreEnabled(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto enabled = module->getAttrOfType<BoolAttr>(kPH1M64UpBColVectorStore);
  auto backendOpt =
      module->getAttrOfType<BoolAttr>(kPH1M64UpBColVectorStoreBackendOpt);
  auto target =
      module->getAttrOfType<StringAttr>(gpu::AttrTargetName);
  auto matchesInt = [&](StringRef name, int64_t expected) {
    auto attr = module->getAttrOfType<IntegerAttr>(name);
    return attr && attr.getInt() == expected;
  };
  return enabled && enabled.getValue() && backendOpt && backendOpt.getValue() &&
         target && target.getValue() == "musa:31" &&
         matchesInt(gpu::AttrNumCTAsName, 1) &&
         matchesInt(gpu::AttrNumWarpsName, 4) &&
         matchesInt(gpu::AttrNumThreadsPerWarp, 32) &&
         matchesInt(kPH1M64UpBColVectorStoreStages, 1);
}

inline bool isPH1M64UpBColRegisterType(RankedTensorType type) {
  if (!type || type.getRank() != 2 || !type.getElementType().isBF16() ||
      type.getShape()[0] != 64 || type.getShape()[1] != 32)
    return false;
  auto blocked = dyn_cast<gpu::BlockedEncodingAttr>(type.getEncoding());
  if (!blocked)
    return false;
  auto equals = [](ArrayRef<unsigned> values, unsigned a, unsigned b) {
    return values.size() == 2 && values[0] == a && values[1] == b;
  };
  auto ctas = gpu::getCTAsPerCGA(type.getEncoding());
  return equals(blocked.getSizePerThread(), 8, 1) &&
         equals(blocked.getThreadsPerWarp(), 8, 4) &&
         equals(blocked.getWarpsPerCTA(), 1, 4) &&
         equals(blocked.getOrder(), 0, 1) && !ctas.empty() &&
         std::all_of(ctas.begin(), ctas.end(),
                     [](unsigned value) { return value == 1; });
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_PH1_M64UP_B_COL_VECTOR_STORE_H
