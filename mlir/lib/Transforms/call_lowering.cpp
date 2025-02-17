// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "imex/Transforms/call_lowering.hpp"
#include "imex/Dialect/imex_util/dialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>

static mlir::Value skipCasts(mlir::Value val) {
  auto getArg = [](mlir::Value arg) -> mlir::Value {
    auto cast = arg.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!cast)
      return {};

    auto inputs = cast.getInputs();
    if (inputs.size() != 1)
      return {};

    return inputs.front();
  };

  while (auto arg = getArg(val))
    val = arg;

  return val;
};

mlir::LogicalResult imex::ExpandCallVarargs::matchAndRewrite(
    plier::PyCallOp op, mlir::PatternRewriter &rewriter) const {
  auto vararg = op.getVarargs();
  if (!vararg)
    return mlir::failure();

  vararg = skipCasts(vararg);

  auto varargType = vararg.getType().dyn_cast<mlir::TupleType>();
  if (!varargType)
    return mlir::failure();

  auto argsCount = op.getArgs().size();
  auto varargsCount = varargType.size();
  llvm::SmallVector<mlir::Value> args(argsCount + varargsCount);
  llvm::copy(op.getArgs(), args.begin());

  auto loc = op.getLoc();
  for (auto i : llvm::seq<size_t>(0, varargsCount)) {
    auto type = varargType.getType(i);
    auto index = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, static_cast<int64_t>(i));
    args[argsCount + i] =
        rewriter.create<imex::util::TupleExtractOp>(loc, type, vararg, index);
  }

  auto resType = op.getType();
  rewriter.replaceOpWithNewOp<plier::PyCallOp>(
      op, resType, op.getFunc(), args, mlir::Value(), op.getKwargs(),
      op.getFuncName(), op.getKwNames());
  return mlir::success();
}

mlir::LogicalResult
imex::CallOpLowering::matchAndRewrite(plier::PyCallOp op,
                                      mlir::PatternRewriter &rewriter) const {
  if (op.getVarargs())
    return mlir::failure();

  auto funcName = op.getFuncName();

  llvm::SmallVector<mlir::Value> args;
  args.reserve(op.getArgs().size() + 1);
  auto func = op.getFunc();
  if (func) {
    auto getattr = func.getDefiningOp<plier::GetattrOp>();
    if (getattr)
      args.emplace_back(skipCasts(getattr.getOperand()));
  }

  for (auto arg : op.getArgs())
    args.emplace_back(skipCasts(arg));

  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Value>> kwargs;
  for (auto it : llvm::zip(op.getKwargs(), op.getKwNames())) {
    auto arg = skipCasts(std::get<0>(it));
    auto name = std::get<1>(it).cast<mlir::StringAttr>();
    kwargs.emplace_back(name.getValue(), arg);
  }

  auto loc = op.getLoc();
  return resolveCall(op, funcName, loc, rewriter, args, kwargs);
}
