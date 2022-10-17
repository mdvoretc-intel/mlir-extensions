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

#ifndef IMEX_DIALECT_KGEN_TRANSFORMS_TILING_H
#define IMEX_DIALECT_KGEN_TRANSFORMS_TILING_H

#include <functional>

#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace kgen {

struct TilingResult {
  Operation *tiledOp;
  Operation *loop;
};

/// Options to use to control tiling.
struct TilingOptions {
  using TileSizeComputationFn =
      std::function<SmallVector<Value>(OpBuilder &, Operation *)>;

  /// Function to materialize the tile sizes for a given operation. This allows
  /// to infer tile sizes statically, e.g. based on an operation's rank, and
  /// also dynamically based, e.g. based on a tensor's shape at runtime.
  TileSizeComputationFn tileSizeComputationFn = nullptr;

  /// If `true`, generate a `kgen.parallel` loop nest.
  bool distribute = true;
};

/// Create tiled operation based on the specified tiling options. The result is
/// equivalent to original op.
FailureOr<TilingResult> tile(const TilingOptions &options,
                             PatternRewriter &rewriter, TilingInterface op);

/// Populate tiling patterns.
void populateTilingPatterns(
    MLIRContext *context,
    llvm::function_ref<LogicalResult(Operation *)> filterFn,
    const TilingOptions &opts, RewritePatternSet *patterns);

} // namespace kgen
} // namespace mlir

#endif // IMEX_DIALECT_KGEN_TRANSFORMS_TILING_H
