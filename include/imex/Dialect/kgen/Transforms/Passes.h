//===-- Passes.h - kgen pass declaration file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines prototypes that expose pass constructors for the
/// kgen dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _kgen_PASSES_H_INCLUDED_
#define _kgen_PASSES_H_INCLUDED_

#include <mlir/Pass/Pass.h>

namespace mlir {
class LLVMTypeConverter;
class MLIRContext;
class ModuleOp;
template <typename T> class OperationPass;
class RewritePatternSet;
} // namespace mlir

namespace imex {

void registerKgenTilingInterfaceExternalModels(DialectRegistry &registry);

//===----------------------------------------------------------------------===//
/// kgen passes.
//===----------------------------------------------------------------------===//

/// Pass to tile ops using TilingInterface.
std::unique_ptr<OperationPass<func::FuncOp>>
createTilingPass(StringRef opName = "", StringRef opLabel = "",
                 bool distribute = true, ArrayRef<int64_t> tileSizes = {});

/// Pass to tile a linalg.generic reduction.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingReductionPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include <imex/Dialect/kgen/Transforms/Passes.h.inc>

} // namespace imex

#endif // _kgen_PASSES_H_INCLUDED_
