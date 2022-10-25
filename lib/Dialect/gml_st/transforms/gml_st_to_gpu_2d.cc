/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "imex/Dialect/gml_st/IR/gml_st_ops.h"
#include "imex/Dialect/gml_st/transforms/passes.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#define GEN_PASS_DEF_GMLSTTOGPU2DPASS
#include "imex/Dialect/gml_st/transforms/passes.h.inc"

using namespace mlir;
using namespace mlir::gml_st;
using mlir::gpu::LaunchOp;

namespace {
/// Converts a sequence of 2 nested gml_st.parallel ops into a gpu.launch op.
/// The loops directly correspond to the blocks and threads of the 2-level
/// threading model used in the gpu.launch op by default.
///
/// Each gml_st.parallel is expected to have up to 3 induction variables, to be
/// mapped onto the X, Y and Z dimensions respectively.
///
/// All operations from within the nested gml_st.parallel regions are copied
/// directly into the gpu.launch region, with induction variables replaced by
/// equivalent values computed using the threading indices. Thus, the 2 nested
/// parallel regions are effectively flattened into a single level of nesting
/// within the gpu.launch region.
struct ParallelOpToGpu2dPattern : public OpRewritePattern<ParallelOp> {
  using OpRewritePattern<ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelOp root,
                                PatternRewriter &rewriter) const override;
};

/// Implements the GmlStToGpu2dPass declared in
/// include/imex/Dialect/gml_st/transforms/passes.td.
struct GmlStToGpu2dPass
    : public ::impl::GmlStToGpu2dPassBase<GmlStToGpu2dPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ParallelOpToGpu2dPattern>(&getContext());
    ConversionTarget target(getContext());
    target.addIllegalDialect<GmlStDialect>();
    // FIXME: restore type == getReductionIteratorTypeName(); with a limit to
    // non-parallelized dimensions.
    target.addDynamicallyLegalOp<linalg::GenericOp>([](linalg::GenericOp op) {
      return llvm::none_of(op.getIteratorTypes().getAsValueRange<StringAttr>(),
                           [](StringRef type) { return false; });
    });
    // We're producing new ops (clones of original ops in gml_st.parallel
    // loops), so we have to mark them explicitly legal, otherwise the
    // conversion fails even if doing partial conversion.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

/// Creates an initial gpu.launch op with launch configuration set to a single
/// thread. The idea is to update those later, as we discover the correct values
/// from the nesting structure.
static LaunchOp createInitialGpuLaunchOp(Location loc, Value defaultSize,
                                         PatternRewriter &rewriter) {
  auto launch =
      rewriter.create<LaunchOp>(loc, defaultSize, defaultSize, defaultSize,
                                defaultSize, defaultSize, defaultSize);
  Block *body = &launch.getBody().front();
  rewriter.setInsertionPointToEnd(body);
  rewriter.create<gpu::TerminatorOp>(loc);
  rewriter.setInsertionPointToStart(body);
  return launch;
}

/// Matches the `launchIdx`-th iteration space of `launch` to the iteration
/// space of `parallel`. Returns an SSA value that is a part of the `launch`'s
/// region, and represents the value of `parallel`'s induction variable.
static Value matchLaunchSpaceToLoop(ParallelOp parallel,
                                    const BlockAndValueMapping &bvm,
                                    unsigned launchIdx, unsigned dimIdx,
                                    LaunchOp launch,
                                    PatternRewriter &rewriter) {
  Location loc = parallel.getLoc();
  Value upperBound = bvm.lookupOrDefault(parallel.getUpperBound()[dimIdx]);
  Value lowerBound = bvm.lookupOrDefault(parallel.getLowerBound()[dimIdx]);
  Value step = bvm.lookupOrDefault(parallel.getStep()[dimIdx]);

  // Compute the value that gml_st.parallel's induction variable would have in
  // each iteration, and make it available to operations within the gpu.launch
  // region.
  AffineMap inductionVarMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/2,
      rewriter.getAffineDimExpr(0) * rewriter.getAffineSymbolExpr(1) +
          rewriter.getAffineSymbolExpr(0));
  Value inductionVar = rewriter.create<AffineApplyOp>(
      loc, inductionVarMap,
      ValueRange{launch.getBody().getArgument(launchIdx + dimIdx), lowerBound,
                 step});

  // TODO: add a check against imperfect tiling here. The current code unsafely
  // assumes the input is evenly tiled
  AffineMap launchBoundMap = AffineMap::get(
      /*dimCount=*/1, /*symbolCount=*/2,
      (rewriter.getAffineDimExpr(0) - rewriter.getAffineSymbolExpr(0))
          .ceilDiv(rewriter.getAffineSymbolExpr(1)));
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(launch);
  launch.setOperand(
      launchIdx + dimIdx,
      rewriter.create<AffineApplyOp>(loc, launchBoundMap,
                                     ValueRange{upperBound, lowerBound, step}));
  return inductionVar;
}

// Converts the 2 nested gml_st.parallel ops rooted at `root` into a
// gpu.launch op. We do this by creating an empty gpu.launch region and
// copying all the operations in gml_st.parallel into that region,
// recursively copying the bodies of any nested gml_st.parallel regions that
// we encounter.
LogicalResult
ParallelOpToGpu2dPattern::matchAndRewrite(ParallelOp root,
                                          PatternRewriter &rewriter) const {
  Location loc = root.getLoc();

  Value defaultSize = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  LaunchOp launch = createInitialGpuLaunchOp(loc, defaultSize, rewriter);

  BlockAndValueMapping bvm;
  // We need to keep track of which value in the gpu.launch region represents
  // which level of the induction variable in the nested region. This is because
  // we might have multiple gml_st.parallel operations on the same level, and
  // their induction variables should map to the same value in the flattened
  // gpu.launch region.
  SmallVector<Value, 4> nestingLevelToInductionVarMap;
  // This is our stack holding in-flight operations of gml_st.parallel regions
  // that we started to copy over to the gpu.launch region, but are on hold
  // while we are processing a nested gml_st.parallel.
  SmallVector<iterator_range<Block::iterator>, 2> loopIterators;

  // This functor implements the processing of a single parallel op:
  // 1)  update of GPU launch bounds according to the interation space
  // 2)  addition of a nesting level to `loopIterators`, with the iterator
  //     over `parallel`'s body
  auto processParallelOp = [&](ParallelOp parallel) {
    unsigned nestingLevel = loopIterators.size();
    unsigned inductionVarIdx = 2 * nestingLevel;
    if (parallel.getNumLoops() != 2) {
      return rewriter.notifyMatchFailure(
          parallel, "should always have 2 induction variables");
    }

    for (unsigned dimIdx = 0; dimIdx < 2; ++dimIdx) {
      Value currentBound = launch.getOperand(inductionVarIdx + dimIdx);
      // We are encountering a loop at this level of nesting for the first time.
      assert(currentBound == defaultSize &&
             "launch bound should use the default size");
      nestingLevelToInductionVarMap.push_back(matchLaunchSpaceToLoop(
          parallel, bvm, inductionVarIdx, dimIdx, launch, rewriter));

      bvm.map(parallel.getInductionVars()[dimIdx],
              nestingLevelToInductionVarMap[inductionVarIdx + dimIdx]);
    }
    loopIterators.push_back(parallel.getBody()->without_terminator());
    return success();
  };

  if (failed(processParallelOp(root)))
    return failure();

  while (!loopIterators.empty()) {
    auto currentLoop = loopIterators.pop_back_val();
    for (Operation &op : currentLoop) {
      if (auto nestedParallel = dyn_cast<ParallelOp>(&op)) {
        // Push the current state back to loopIterator and start the next level
        // of nesting.
        loopIterators.push_back(
            llvm::make_range(std::next(op.getIterator()), currentLoop.end()));
        if (failed(processParallelOp(nestedParallel)))
          return failure();
        break;
      }
      // TODO(b/244314146): Figure out what we need to do for operations
      // encountered on upper nesting levels to correctly lower them after the
      // rewrite to gpu.launch.
      Operation *clone = rewriter.clone(op, bvm);
      bvm.map(op.getResults(), clone->getResults());
    }
  }

  rewriter.eraseOp(root);
  return success();
}
