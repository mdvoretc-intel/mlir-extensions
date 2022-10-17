//===-- PassDetail.h - kgen pass details --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes for kgen dialect passes.
///
//===----------------------------------------------------------------------===//

#ifndef _kgen_PASSDETAIL_H_INCLUDED_
#define _kgen_PASSDETAIL_H_INCLUDED_

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

class AffineDialect;

namespace arith {
class ArithmeticDialect;
} // namespace arith

// FIXME define other dependent MLIR dialects

} // namespace mlir

namespace imex {

#define GEN_PASS_CLASSES
#include <imex/Dialect/kgen/Transforms/Passes.h.inc>

} // namespace imex

#endif // _kgen_PASSDETAIL_H_INCLUDED_
