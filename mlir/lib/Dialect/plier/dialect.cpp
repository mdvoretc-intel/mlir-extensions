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

#include "imex/Dialect/plier/dialect.hpp"
#include "imex/Dialect/imex_util/dialect.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/InliningUtils.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <llvm/ADT/TypeSwitch.h>

namespace MemoryEffects = ::mlir::MemoryEffects;

namespace {
struct PlierInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Region *, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return true;
  }
  bool isLegalToInline(mlir::Operation *op, mlir::Region *, bool,
                       mlir::BlockAndValueMapping &) const final override {
    return !mlir::isa<plier::ArgOp>(op);
  }
};
} // namespace

namespace plier {
namespace detail {
struct PyTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::StringRef;

  PyTypeStorage(mlir::StringRef name) : name(name) {}

  bool operator==(const KeyTy &key) const { return key == name; }

  static PyTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                  const KeyTy &key) {
    return new (allocator.allocate<PyTypeStorage>())
        PyTypeStorage(allocator.copyInto(key));
  }

  mlir::StringRef name;
};
} // namespace detail

mlir::ArrayRef<detail::OperatorNamePair> getOperators() {
  return llvm::makeArrayRef(detail::OperatorNames);
}

void PlierDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "imex/Dialect/plier/PlierOps.cpp.inc"
      >();
  addTypes<plier::PyType, SliceType>();
  addInterfaces<PlierInlinerInterface>();
}

mlir::Type PlierDialect::parseType(mlir::DialectAsmParser &parser) const {
  parser.emitError(parser.getNameLoc(), "unknown type");
  return mlir::Type();
}

void PlierDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter &os) const {
  llvm::TypeSwitch<mlir::Type>(type)
      .Case<plier::PyType>(
          [&](auto t) { os << "PyType<" << t.getName() << ">"; })
      .Case<plier::SliceType>([&](auto) { os << "SliceType"; })
      .Default([](auto) { llvm_unreachable("unexpected type"); });
}

mlir::Operation *PlierDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  if (mlir::arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<mlir::arith::ConstantOp>(loc, type, value);

  return nullptr;
}

PyType PyType::get(mlir::MLIRContext *context, llvm::StringRef name) {
  assert(!name.empty());
  return Base::get(context, name);
}

PyType PyType::getUndefined(mlir::MLIRContext *context) {
  return Base::get(context, "");
}

llvm::StringRef PyType::getName() const { return getImpl()->name; }

SliceType SliceType::get(mlir::MLIRContext *context) {
  assert(context);
  return Base::get(context);
}

void ArgOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  unsigned index, mlir::StringRef name) {
  ArgOp::build(builder, state, PyType::getUndefined(state.getContext()), index,
               name);
}

mlir::OpFoldResult ArgOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto func = getOperation()->getParentOfType<mlir::func::FuncOp>();
  if (func) {
    auto ind = getIndex();
    if (ind < func.getNumArguments() &&
        func.getArgument(ind).getType() == getType()) {
      return func.getArgument(ind);
    }
  }
  return nullptr;
}

void ConstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,

                    mlir::Attribute val) {
  ConstOp::build(builder, state, PyType::getUndefined(state.getContext()), val);
}

void GlobalOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     mlir::StringRef name) {
  GlobalOp::build(builder, state, PyType::getUndefined(state.getContext()),
                  name);
}

void BinOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs, mlir::StringRef op) {
  BinOp::build(builder, state, PyType::getUndefined(state.getContext()), lhs,
               rhs, op);
}

void UnaryOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value value, mlir::StringRef op) {
  UnaryOp::build(builder, state, PyType::getUndefined(state.getContext()),
                 value, op);
}

static mlir::Value propagateCasts(mlir::Value val, mlir::Type thisType);

template <typename T>
static mlir::Value foldPrevCast(mlir::Value val, mlir::Type thisType) {
  if (auto prevOp = val.getDefiningOp<T>()) {
    auto prevArg = prevOp->getOperand(0);
    if (prevArg.getType() == thisType)
      return prevArg;

    auto res = propagateCasts(prevArg, thisType);
    if (res)
      return res;
  }
  return {};
}

static mlir::Value propagateCasts(mlir::Value val, mlir::Type thisType) {
  using fptr = mlir::Value (*)(mlir::Value, mlir::Type);
  const fptr handlers[] = {
      &foldPrevCast<imex::util::SignCastOp>,
      &foldPrevCast<CastOp>,
      &foldPrevCast<mlir::UnrealizedConversionCastOp>,
  };

  for (auto h : handlers) {
    auto res = h(val, thisType);
    if (res)
      return res;
  }

  return {};
}

mlir::OpFoldResult CastOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  auto arg = getValue();
  auto opType = arg.getType();
  auto retType = getType();
  if (opType == retType && opType != PyType::getUndefined(getContext()))
    return arg;

  if (auto res = propagateCasts(arg, retType))
    return res;

  return nullptr;
}

namespace {
struct PropagateCasts
    : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputs = op.getInputs();
    if (inputs.size() != 1 || op->getNumResults() != 1)
      return mlir::failure();

    auto thisType = op.getType(0);
    auto arg = inputs[0];
    auto res = propagateCasts(arg, thisType);
    if (!res)
      return mlir::failure();

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};
} // namespace

void CastOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  results.insert<PropagateCasts>(context);
}

void PyCallOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value func,
    llvm::StringRef func_name, mlir::ValueRange args, mlir::Value varargs,
    mlir::ArrayRef<std::pair<std::string, mlir::Value>> kwargs) {
  PyCallOp::build(builder, state,
                  plier::PyType::getUndefined(builder.getContext()), func,
                  func_name, args, varargs, kwargs);
}

void PyCallOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type type,
    mlir::Value func, llvm::StringRef func_name, mlir::ValueRange args,
    mlir::Value varargs,
    mlir::ArrayRef<std::pair<std::string, mlir::Value>> kwargs) {
  auto ctx = builder.getContext();

  llvm::SmallVector<mlir::Value> kwArgsVals(kwargs.size());
  llvm::copy(llvm::make_second_range(kwargs), kwArgsVals.begin());

  llvm::SmallVector<mlir::Attribute> kwNames;
  kwNames.reserve(kwargs.size());
  for (auto &a : kwargs)
    kwNames.push_back(mlir::StringAttr::get(ctx, a.first));

  PyCallOp::build(builder, state, type, func, args, varargs, kwArgsVals,
                  func_name, mlir::ArrayAttr::get(ctx, kwNames));
}

void BuildTupleOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         ::mlir::ValueRange args) {
  BuildTupleOp::build(builder, state, PyType::getUndefined(state.getContext()),
                      args);
}

void GetItemOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      ::mlir::Value value, ::mlir::Value index) {
  GetItemOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value, index);
}

void GetiterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      ::mlir::Value value) {
  GetiterOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value);
}

void IternextOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       ::mlir::Value value) {
  IternextOp::build(builder, state, PyType::getUndefined(state.getContext()),
                    value);
}

void PairfirstOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        ::mlir::Value value) {
  PairfirstOp::build(builder, state, PyType::getUndefined(state.getContext()),
                     value);
}

void PairsecondOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         ::mlir::Value value) {
  PairsecondOp::build(builder, state, PyType::getUndefined(state.getContext()),
                      value);
}

void GetattrOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value value, mlir::StringRef name) {
  GetattrOp::build(builder, state, PyType::getUndefined(state.getContext()),
                   value, name);
}

void ExhaustIterOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          mlir::Value value, int64_t count) {
  ExhaustIterOp::build(builder, state, PyType::getUndefined(state.getContext()),
                       value, builder.getI64IntegerAttr(count));
}

mlir::OpFoldResult
ExhaustIterOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/) {
  if (getType() == getOperand().getType() &&
      getType() != plier::PyType::getUndefined(getContext())) {
    return getOperand();
  }
  return nullptr;
}

namespace {
struct GetattrGlobalRewrite : public mlir::OpRewritePattern<GetattrOp> {
  using mlir::OpRewritePattern<GetattrOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(GetattrOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto prevOp = mlir::dyn_cast_or_null<plier::GlobalOp>(
        op.getOperand().getDefiningOp());
    if (prevOp) {
      auto newName = llvm::Twine(prevOp.getName() + "." + op.getName()).str();
      auto newOp =
          rewriter.create<plier::GlobalOp>(op.getLoc(), op.getType(), newName);
      rewriter.replaceOp(op, newOp.getResult());
      return mlir::success();
    }
    return mlir::failure();
  }
};
} // namespace

void GetattrOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                            ::mlir::MLIRContext *context) {
  results.insert<GetattrGlobalRewrite>(context);
}

void BuildSliceOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value begin, mlir::Value end,
                         mlir::Value stride) {
  auto type = SliceType::get(builder.getContext());
  BuildSliceOp::build(builder, state, type, begin, end, stride);
}

namespace {
struct SliceGetitemPropagate
    : public mlir::OpRewritePattern<plier::SliceGetItemOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(plier::SliceGetItemOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op.getArray().getType().isa<mlir::ShapedType>())
      return mlir::failure();

    auto index = mlir::getConstantIntValue(op.getIndex());
    if (!index)
      return mlir::failure();

    auto i = *index;
    if (i < 0 || i >= 3)
      return mlir::failure();

    auto buildSlice = op.getSlice().getDefiningOp<plier::BuildSliceOp>();
    if (!buildSlice)
      return mlir::failure();

    auto loc = op.getLoc();
    auto getInd = [&](int64_t val) -> mlir::Value {
      return rewriter.create<mlir::arith::ConstantIndexOp>(loc, val);
    };

    auto src = buildSlice.getOperand(static_cast<unsigned>(i));
    auto srcType = src.getType();
    if (srcType.isa<mlir::NoneType>()) {
      if (i == 0) {
        rewriter.replaceOp(op, getInd(0));
      } else if (i == 1) {
        auto size = [&]() -> mlir::Value {
          if (op.getArray().getType().isa<mlir::TensorType>())
            return rewriter.create<mlir::tensor::DimOp>(loc, op.getArray(),
                                                        op.getDim());
          return rewriter.create<mlir::memref::DimOp>(loc, op.getArray(),
                                                      op.getDim());
        }();
        rewriter.replaceOp(op, size);
      } else { // i == 2
        rewriter.replaceOp(op, getInd(1));
      }
    } else {
      if (auto intType = srcType.dyn_cast<mlir::IntegerType>()) {
        if (!intType.isSignless()) {
          auto signless =
              mlir::IntegerType::get(intType.getContext(), intType.getWidth());
          src = rewriter.create<imex::util::SignCastOp>(loc, signless, src);
        }
        auto indexType = rewriter.getIndexType();
        src = rewriter.create<mlir::arith::IndexCastOp>(loc, indexType, src);
      } else if (srcType.isa<mlir::IndexType>()) {
        // Nothing
      } else {
        return mlir::failure();
      }
      rewriter.replaceOp(op, src);
    }

    return mlir::success();
  }
};
} // namespace

void SliceGetItemOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<SliceGetitemPropagate>(context);
}
} // namespace plier

#include "imex/Dialect/plier/PlierOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "imex/Dialect/plier/PlierOps.cpp.inc"

//#include "imex/Dialect/plier/PlierOpsEnums.cpp.inc"
