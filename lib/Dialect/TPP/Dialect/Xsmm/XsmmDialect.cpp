//===- XsmmDialect.cpp - Xsmm dialect ---------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "imex/Dialect/TPP/Dialect/Xsmm/XsmmDialect.h"
#include "imex/Dialect/TPP/Dialect/Xsmm/XsmmOps.h"

using namespace mlir;
using namespace mlir::xsmm;

//===----------------------------------------------------------------------===//
// Xsmm dialect.
//===----------------------------------------------------------------------===//

void XsmmDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "imex/Dialect/TPP/Dialect/Xsmm/XsmmOps.cpp.inc"
      >();
}

#include "imex/Dialect/TPP/Dialect/Xsmm/XsmmOpsDialect.cpp.inc"
