#ifndef TRITONMUSA_COMMON_PH1_WARP_SHUFFLE_PLAN_H
#define TRITONMUSA_COMMON_PH1_WARP_SHUFFLE_PLAN_H

#include <array>

namespace mlir::triton::musa {

// Exact MP31 B tile only: exchange packed-register bits 0/1 with lane bits
// 3/4. The caller proves a full 32-lane warp with 8 live i32 values per lane.
// Adjacent K BF16 values occupy the low/high halves of each i32. Keeping them
// packed reduces the transpose to eight 32-bit shuffles per participating
// thread; there is no memory or barrier operation in this plan.
//
// This operation schedule is shared by the LLVM emitter and the CPU test.
// The test interprets a Value as an entire warp, so shuffle reads another
// lane of the *same SSA value*, never a destination-dependent source register.
template <typename Value, typename IsLowLane, typename Select, typename Shuffle>
std::array<Value, 8> transposePH1BRowPairs(std::array<Value, 8> values,
                                       IsLowLane isLowLane, Select select,
                                       Shuffle shuffle) {
  for (unsigned stage = 0; stage < 2; ++stage) {
    unsigned registerBit = 1u << stage;
    unsigned laneBit = 8u << stage;
    auto lowLane = isLowLane(laneBit);
    for (unsigned low = 0; low < 8; ++low) {
      if (low & registerBit)
        continue;
      unsigned high = low | registerBit;
      Value sent = select(lowLane, values[high], values[low]);
      Value received = shuffle(sent, laneBit);
      Value lowResult = select(lowLane, values[low], received);
      Value highResult = select(lowLane, received, values[high]);
      values[low] = lowResult;
      values[high] = highResult;
    }
  }
  return values;
}

} // namespace mlir::triton::musa

#endif
