//===- kgenOps.h - kgen dialect  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the kgen dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _kgen_OPS_H_INCLUDED_
#define _kgen_OPS_H_INCLUDED_

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace imex {
namespace kgen {} // namespace kgen
} // namespace imex

#include <imex/Dialect/kgen/IR/kgenOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/kgen/IR/kgenOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/kgen/IR/kgenOps.h.inc>

#endif // _kgen_OPS_H_INCLUDED_
