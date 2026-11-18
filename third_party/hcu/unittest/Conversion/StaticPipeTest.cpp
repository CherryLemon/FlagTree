#include "TritonHCUGPUToLLVM/StaticPipe.h"

#include <gtest/gtest.h>
#include <limits>

namespace mlir::triton::HCU::static_pipe {
namespace {

class StaticPipeIntegerTest : public ::testing::Test {
protected:
  StaticPipeIntegerTest() : builder(&context) {
    context.loadDialect<arith::ArithDialect>();
    builder.setInsertionPointToStart(&block);
  }

  Value literal(int64_t value, unsigned width = 32) {
    return arith::ConstantIntOp::create(builder, builder.getUnknownLoc(), value,
                                        width);
  }

  Value induction() {
    return block.addArgument(builder.getI32Type(), builder.getUnknownLoc());
  }

  MLIRContext context;
  OpBuilder builder;
  Block block;
};

TEST_F(StaticPipeIntegerTest, PreservesCanonicalStagePhaseAndOffset) {
  auto loc = builder.getUnknownLoc();
  Value iv = induction(), two = literal(2), zero = literal(0);
  auto stage = arith::RemSIOp::create(builder, loc, iv, two);
  auto epoch = arith::DivSIOp::create(builder, loc, iv, two);
  auto parity = arith::RemSIOp::create(builder, loc, epoch, two);
  auto phase = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                     parity, zero);
  auto offset = arith::MulIOp::create(builder, loc, iv, literal(64));

  for (int64_t i = 0; i < 4096; ++i) {
    EXPECT_EQ(evaluate(stage, iv, i), i % 2);
    EXPECT_EQ(evaluate(phase, iv, i), (i / 2) % 2);
    EXPECT_EQ(evaluate(offset, iv, i), i * 64);
  }
}

TEST_F(StaticPipeIntegerTest,
       DoesNotProveAnOverflowingIterationIsTheLoopIndex) {
  auto loc = builder.getUnknownLoc();
  Value iv = induction(), scale = literal(std::numeric_limits<int32_t>::max());
  // The public pipe API derives phase from the supplied iteration. Unbounded
  // arithmetic incorrectly proves (i * INT32_MAX) / INT32_MAX == i.
  auto product = arith::MulIOp::create(builder, loc, iv, scale);
  auto iteration = arith::DivSIOp::create(builder, loc, product, scale);
  auto epoch = arith::DivSIOp::create(builder, loc, iteration, literal(2));
  auto phase = arith::RemSIOp::create(builder, loc, epoch, literal(2));

  EXPECT_EQ(evaluate(iteration, iv, 1), 1);
  EXPECT_EQ(evaluate(product, iv, 2), -2);
  EXPECT_EQ(evaluate(iteration, iv, 2), 0);
  EXPECT_EQ(evaluate(phase, iv, 2), 0);
  EXPECT_NE(evaluate(phase, iv, 2), (2 / 2) % 2);
}

TEST_F(StaticPipeIntegerTest, AppliesTruncationBeforeComputingTheStage) {
  auto loc = builder.getUnknownLoc();
  Value iv = induction();
  auto narrow = arith::TruncIOp::create(builder, loc, builder.getI8Type(), iv);
  auto wide =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), narrow);
  auto lost = arith::SubIOp::create(builder, loc, wide, iv);
  auto correction = arith::DivSIOp::create(builder, loc, lost, literal(256));
  auto iteration = arith::AddIOp::create(builder, loc, iv, correction);
  auto stage = arith::RemSIOp::create(builder, loc, iteration, literal(2));

  EXPECT_EQ(evaluate(stage, iv, 255), 1);
  EXPECT_EQ(evaluate(wide, iv, 256), 0);
  EXPECT_EQ(evaluate(stage, iv, 256), 1);
  EXPECT_NE(evaluate(stage, iv, 256), 256 % 2);
}

TEST_F(StaticPipeIntegerTest, DistinguishesSignAndZeroExtension) {
  auto loc = builder.getUnknownLoc();
  Value byte = literal(-1, 8), bit = literal(1, 1);
  auto signedByte =
      arith::ExtSIOp::create(builder, loc, builder.getI32Type(), byte);
  auto unsignedByte =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), byte);
  auto signedBit =
      arith::ExtSIOp::create(builder, loc, builder.getI32Type(), bit);
  auto unsignedBit =
      arith::ExtUIOp::create(builder, loc, builder.getI32Type(), bit);

  EXPECT_EQ(constant(signedByte), -1);
  EXPECT_EQ(constant(unsignedByte), 255);
  EXPECT_EQ(constant(signedBit), -1);
  EXPECT_EQ(constant(unsignedBit), 1);
}

TEST_F(StaticPipeIntegerTest, UsesSignednessForDivisionRemainderAndShift) {
  auto loc = builder.getUnknownLoc();
  Value a = literal(-3, 8), two = literal(2, 8), one = literal(1, 8);
  EXPECT_EQ(constant(arith::DivSIOp::create(builder, loc, a, two)), -1);
  EXPECT_EQ(constant(arith::DivUIOp::create(builder, loc, a, two)), 126);
  EXPECT_EQ(constant(arith::RemSIOp::create(builder, loc, a, two)), -1);
  EXPECT_EQ(constant(arith::RemUIOp::create(builder, loc, a, two)), 1);
  EXPECT_EQ(constant(arith::ShRSIOp::create(builder, loc, a, one)), -2);
  EXPECT_EQ(constant(arith::ShRUIOp::create(builder, loc, a, one)), 126);
}

TEST_F(StaticPipeIntegerTest, RejectsUndefinedArithmetic) {
  auto loc = builder.getUnknownLoc();
  Value one = literal(1), zero = literal(0);
  Value min = literal(std::numeric_limits<int32_t>::min());
  EXPECT_FALSE(constant(arith::DivSIOp::create(builder, loc, one, zero)));
  EXPECT_FALSE(constant(arith::DivUIOp::create(builder, loc, one, zero)));
  EXPECT_FALSE(constant(arith::RemSIOp::create(builder, loc, one, zero)));
  EXPECT_FALSE(constant(arith::RemUIOp::create(builder, loc, one, zero)));
  EXPECT_FALSE(
      constant(arith::DivSIOp::create(builder, loc, min, literal(-1))));
  EXPECT_FALSE(
      constant(arith::RemSIOp::create(builder, loc, min, literal(-1))));
  EXPECT_FALSE(constant(arith::ShLIOp::create(builder, loc, one, literal(32))));
  EXPECT_FALSE(
      constant(arith::ShRUIOp::create(builder, loc, one, literal(-1))));
}

TEST_F(StaticPipeIntegerTest, RejectsViolatedOverflowFlags) {
  auto loc = builder.getUnknownLoc();
  auto nsw = arith::IntegerOverflowFlagsAttr::get(
      &context, arith::IntegerOverflowFlags::nsw);
  auto nuw = arith::IntegerOverflowFlagsAttr::get(
      &context, arith::IntegerOverflowFlags::nuw);
  auto add =
      arith::AddIOp::create(builder, loc, literal(127, 8), literal(1, 8));
  EXPECT_EQ(constant(add), -128);
  add.setOverflowFlagsAttr(nsw);
  EXPECT_FALSE(constant(add));
  auto sub = arith::SubIOp::create(builder, loc, literal(0, 8), literal(1, 8));
  EXPECT_EQ(constant(sub), -1);
  sub.setOverflowFlagsAttr(nuw);
  EXPECT_FALSE(constant(sub));
  auto trunc =
      arith::TruncIOp::create(builder, loc, builder.getI8Type(), literal(256));
  EXPECT_EQ(constant(trunc), 0);
  trunc.setOverflowFlagsAttr(nuw);
  EXPECT_FALSE(constant(trunc));
}

TEST_F(StaticPipeIntegerTest, UsesTheHcuIndexWidth) {
  auto loc = builder.getUnknownLoc();
  auto index = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                          literal(int64_t{1} << 32, 64));
  EXPECT_EQ(constant(index), 0);
  auto signedIndex = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), literal(-1));
  auto extended = arith::IndexCastUIOp::create(
      builder, loc, builder.getI64Type(), signedIndex);
  EXPECT_EQ(constant(extended), 4294967295LL);
}

} // namespace
} // namespace mlir::triton::HCU::static_pipe
