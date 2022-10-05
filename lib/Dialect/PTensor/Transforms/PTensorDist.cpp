//===- PTensorDist.cpp - PTensorToDist Transform  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transform of the PTensor dialect to a combination of
/// PTensor and Dist dialects.
/// operation
///
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/internal/PassWrapper.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

namespace imex {

namespace {

// *******************************
// ***** Some helper functions ***
// *******************************

inline ::mlir::Value createInitWithRT(::mlir::Location &loc,
                                      ::mlir::OpBuilder &builder, uint64_t rank,
                                      ::mlir::Value gshape) {
  /*
    auto rankv = createInt(loc, builder, rank);
    llvm::SmallVector<int64_t> shp(rank, -1);
    auto fsa = builder.getStringAttr("_idtr_init_dtensor");
    auto tnsr = builder.create<::mlir::tensor::CastOp>(loc,
    ::mlir::RankedTensorType::get({-1}, builder.getIndexType()), gshape); auto
    call = builder.create<::mlir::func::CallOp>(loc, fsa, builder.getI64Type(),
    ::mlir::ValueRange({tnsr, rankv})); return call.getResult(0);
    */
  return builder.create<::imex::dist::RegisterPTensorOp>(
      loc, builder.getI64Type(), gshape);
}

inline ::mlir::Value callGetRankedData(::mlir::Location &loc,
                                       ::mlir::OpBuilder &builder,
                                       const char *func, ::mlir::Value guid,
                                       uint64_t rank) {
  auto rankv_ = createInt(loc, builder, rank);
  auto rankv = builder.create<::mlir::arith::IndexCastOp>(
      loc, builder.getIndexType(), rankv_);
  auto tnsr = builder.create<::mlir::linalg::InitTensorOp>(
      loc, ::mlir::ValueRange({rankv}), builder.getI64Type());
  auto fsa = builder.getStringAttr(func);
  (void)builder.create<::mlir::func::CallOp>(
      loc, fsa, ::mlir::TypeRange(), ::mlir::ValueRange({guid, tnsr, rankv_}));
  return tnsr;
}

inline ::mlir::Value createGetLocalShape(::mlir::Location &loc,
                                         ::mlir::OpBuilder &builder,
                                         ::mlir::Value guid, uint64_t rank) {
  //  return callGetRankedData(loc, builder, "_idtr_local_shape", guid, rank);

  return builder.create<::imex::dist::LocalShapeOp>(
      loc, ::mlir::RankedTensorType::get({rank}, builder.getIndexType()), guid);
}

inline ::mlir::Value createGetLocalOffsets(::mlir::Location &loc,
                                           ::mlir::OpBuilder &builder,
                                           ::mlir::Value guid, uint64_t rank) {
  //  return callGetRankedData(loc, builder, "_idtr_local_offsets", guid, rank);
  return builder.create<::imex::dist::LocalOffsetsOp>(
      loc, ::mlir::RankedTensorType::get({rank}, builder.getIndexType()), guid);
}

inline ::mlir::Value createAllReduce(::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Attribute op,
                                     ::mlir::Value pTnsr) {
  auto pTnsrTyp = pTnsr.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(pTnsrTyp);
  auto rTnsr = builder.create<::imex::ptensor::ExtractRTensorOp>(
      loc, pTnsrTyp.getRtensor(), pTnsr);
  /*
    auto opV = builder.create<::mlir::arith::ConstantOp>(loc, op);
    auto dtype = createInt<sizeof(int)*8>(loc, builder, 5); // FIXME getDType
    auto fsa = builder.getStringAttr("_idtr_reduce_all");
    (void) builder.create<::mlir::func::CallOp>(loc, fsa, ::mlir::TypeRange(),
    ::mlir::ValueRange({rTnsr, dtype, opV})); return rTnsr;
    */
  return builder.create<::imex::dist::AllReduceOp>(loc, rTnsr.getType(), op,
                                                   rTnsr);
}

inline ::mlir::Value createGetLocal(::mlir::Location &loc,
                                    ::mlir::OpBuilder &builder,
                                    ::mlir::Value pt) {
  auto ptTyp = pt.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(ptTyp);
  if (ptTyp.getDist()) {
    auto rtnsr = builder.create<::imex::ptensor::ExtractRTensorOp>(
        loc, ptTyp.getRtensor(), pt);
    // FIXME: device
    return builder.create<::imex::ptensor::MkPTensorOp>(loc, rtnsr);
  }
  // not dist
  return pt;
}

inline ::mlir::Value createMkTnsr(::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder, ::mlir::Value pt,
                                  ::mlir::Value guid) {
  auto ptTyp = pt.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(ptTyp);
  auto rTnsr = builder.create<::imex::ptensor::ExtractRTensorOp>(
      loc, ptTyp.getRtensor(), pt);
  auto dmy = createInt<1>(loc, builder, 0);
  return builder.create<::imex::ptensor::MkPTensorOp>(loc, false, true, rTnsr,
                                                      dmy, dmy, guid);
}

// *******************************
// ***** Individual patterns *****
// *******************************

struct DistARange : public ::mlir::OpRewritePattern<::imex::ptensor::ARangeOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ARangeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // nothing to do if no team
    auto team = op.getTeam();
    if (!team)
      return ::mlir::failure();

    // get operands
    auto start = op.getStart();
    auto step = op.getStep();
    // compute global count (so we know the shape)
    auto count = createCountARange(rewriter, loc, start, op.getStop(), step);
    auto dtype = rewriter.getI64Type();
    // result shape is 1d
    uint64_t rank = 1;
    auto gShpTnsr = rewriter.create<::mlir::linalg::InitTensorOp>(
        loc, ::mlir::ValueRange({count}), dtype);
    auto gShape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, gShpTnsr);
    // so is the local shape
    llvm::SmallVector<mlir::Value> lShapeVVec(1);
    // get guid
    auto guid = createInitWithRT(loc, rewriter, 1, gShape);
    // get local shape
    auto lShapeVVec_mr = createGetLocalShape(loc, rewriter, guid, rank);
    auto zero = createIndex(loc, rewriter, 0);
    auto lSz_ = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, rewriter.getIndexType(), lShapeVVec_mr,
        ::mlir::ValueRange({zero}));
    auto lSz = rewriter.create<::mlir::arith::IndexCastOp>(loc, dtype, lSz_);
    // get local offsets
    auto offsets = createGetLocalOffsets(loc, rewriter, guid, rank);
    // create start from offset
    auto off_ = rewriter.create<::mlir::tensor::ExtractOp>(
        loc, rewriter.getIndexType(), offsets, ::mlir::ValueRange({zero}));
    auto off = rewriter.create<::mlir::arith::IndexCastOp>(loc, dtype, off_);
    auto tmp =
        rewriter.create<::mlir::arith::MulIOp>(loc, off, step); // off * step
    start = rewriter.create<::mlir::arith::AddIOp>(
        loc, start, tmp); // start + (off * stride)
    // create stop
    auto tmp2 = rewriter.create<::mlir::arith::MulIOp>(
        loc, lSz, step); // step * lShape[0]
    auto stop = rewriter.create<::mlir::arith::AddIOp>(
        loc, start, tmp2); // start + (lShape[0] * stride)
    // auto one = createInt(loc, rewriter, 1);
    // auto stop = rewriter.create<::mlir::arith::AddIOp>(loc, tmp3, one); //
    // increment by one to not miss the last
    //  get type of local tensor
    ::llvm::ArrayRef<int64_t> lShape({-1});
    auto artype = ::imex::ptensor::PTensorType::get(
        rewriter.getContext(), ::mlir::RankedTensorType::get({-1}, dtype),
        false, false);
    // finally create local arange
    auto dmy = ::mlir::Value(); // createInt<1>(loc, rewriter, 0);
    auto arres = rewriter.create<::imex::ptensor::ARangeOp>(
        loc, artype, start, stop, step, dmy, dmy, dmy);
    rewriter.replaceOp(op, createMkTnsr(loc, rewriter, arres, guid));
    return ::mlir::success();
  }
};

struct DistEWBinOp : public ::mlir::OpRewritePattern<::imex::ptensor::EWBinOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhsPtTyp =
        op.getLhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto rhsPtTyp =
        op.getRhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    // return success if wrong ops or not distributed
    if (!lhsPtTyp || !rhsPtTyp || !lhsPtTyp.getDist() || !lhsPtTyp.getDist()) {
      return ::mlir::success();
    }

    // result shape
    auto gShapeARef = lhsPtTyp.getRtensor().getShape();
    auto gShapeAttr = rewriter.getIndexVectorAttr(gShapeARef);
    auto gShape = rewriter.create<::mlir::shape::ConstShapeOp>(loc, gShapeAttr);
    // auto dtype = lhsPtTyp.getRtensor().getElementType();
    // Init our new dist tensor
    auto guid = createInitWithRT(loc, rewriter, 1, gShape);
    // local ewb op
    auto lLhs = createGetLocal(loc, rewriter, op.getLhs());
    auto lRhs = createGetLocal(loc, rewriter, op.getRhs());
    // return type same as lhs for now
    auto retPtTyp = lLhs.getType(); // FIXME
    auto ewbres = rewriter.create<::imex::ptensor::EWBinOp>(
        loc, retPtTyp, op.getOp(), lLhs, lRhs);
    rewriter.replaceOp(op, createMkTnsr(loc, rewriter, ewbres, guid));
    return ::mlir::success();
  }
};

struct DistReductionOp
    : public ::mlir::OpRewritePattern<::imex::ptensor::ReductionOp> {
  using OpRewritePattern::OpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // FIXME reduction over individual dimensions is not supported
    auto loc = op.getLoc();
    // get input
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!inpPtTyp || !inpPtTyp.getDist()) {
      return ::mlir::success();
    }

    // result shape is 0d
    auto gShapeAttr = rewriter.getIndexTensorAttr({});
    auto gShape = rewriter.create<::mlir::shape::ConstShapeOp>(loc, gShapeAttr);
    // Init our new dist tensor
    auto guid = createInitWithRT(loc, rewriter, 1, gShape);
    // Local reduction
    auto local = createGetLocal(loc, rewriter, op.getInput());
    // return type 0d with same dtype as input
    auto dtype = inpPtTyp.getRtensor().getElementType();
    auto retPtTyp = ::imex::ptensor::PTensorType::get(
        rewriter.getContext(), ::mlir::RankedTensorType::get({}, dtype), false,
        false);
    auto redPTnsr = rewriter.create<::imex::ptensor::ReductionOp>(
        loc, retPtTyp, op.getOp(), local);
    // global reduction
    auto retRTnsr = createAllReduce(loc, rewriter, op.getOp(), redPTnsr);
    // finish
    auto dmy = createInt<1>(loc, rewriter, 0);
    rewriter.replaceOpWithNewOp<::imex::ptensor::MkPTensorOp>(
        op, false, true, retRTnsr, dmy, dmy, guid);
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct PTensorDistPass : public ::imex::PTensorDistBase<PTensorDistPass> {

  PTensorDistPass() = default;

  void runOnOperation() override {
    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<DistARange, DistEWBinOp, DistReductionOp>(getContext(),
                                                             patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);
  }
};

} // namespace

/// Populate the given list with patterns that eliminate Dist ops
void populatePTensorDistPatterns(::mlir::LLVMTypeConverter &converter,
                                 ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createPTensorDistPass() {
  return std::make_unique<PTensorDistPass>();
}

} // namespace imex
