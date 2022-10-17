//===-- Passes.td - kgen pass definition file --------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the kgen dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _kgen_PASSES_TD_INCLUDED_
#define _kgen_PASSES_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/Pass/PassBase.td"

def TilingInterface : OpInterface<"TilingInterface"> {
  let description = [{
    Interface for operations to expose information needed to tile them.
  }];
  let cppNamespace = "::mlir::gml_st";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of operands into which the result of the
          tiled implementation is written into. With `tensor`
          operands, this will be used as the initial tensor into which
          the tiled results are inserted into. With `memref` operands,
          this will be the operand into which the result of the tiled
          operation is written into.
        }],
        /*retType=*/"SmallVector<Value>",
        /*methodName=*/"getDestinationOperands",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of iterator types that describe the number of loops.
        }],
        /*retType=*/"SmallVector<utils::IteratorType>",
        /*methodName=*/"getLoopIteratorTypes",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getIterationDomain",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate the tiled implementation of an operation.

          The iteration space of the operation is returned by
          `getIterationDomain`. The caller provides the information of the
          tile within this iteration space whose implementation the
          caller needs.
          - `offsets` provides the offset of the tile in the coordinate system
            of the original iteration space, i.e., if an iteration space
            dimension had non-zero offset, it must be included in the offset
            provided here (as opposed to zero-based offset "relative" to the
            iteration space).
          - `sizes` provides the size of the tile.

          The method returns the operation that is the tiled
          implementation.
        }],
        /*retType=*/"mlir::gml_st::TilingInterface",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ArrayRef<OpFoldResult>":$offsets,
            "ArrayRef<OpFoldResult>":$sizes)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates the IR that computes the tile of a result of the
          operation.  The `offsets` and `sizes` describe the tile of
          the output required. This is different from
          `getTiledImplementation` which generates the tiled
          implementation of the operation given a tile of the
          iteration space. This method generates a tiled
          implementation of the operation based on the tile of the
          result required. This method enables fusion by using tile
          and fuse. The method returns failure if the operation can't be
          tiled to generate the result tile. In practical terms this
          implies it cannot be tiled and fused with its consumers.

          - `offsets` provides the offset of the tile in the coordinate system
            of the original iteration space, i.e., if an iteration space
            dimension had non-zero offset, it must be included in the offset
            provided here (as opposed to zero-based offset "relative" to the
            iteration space).
          - `sizes` provides the size of the tile.
        }],
        /*retType=*/"FailureOr<Value>",
        /*methodName=*/"generateResultTileValue",
        /*args=*/(ins
          "OpBuilder &":$b,
          "unsigned":$resultNumber,
          "ArrayRef<OpFoldResult>":$offsets,
          "ArrayRef<OpFoldResult>":$sizes)
      >
  ];
}

def TilingPass : Pass<"kgen-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile operations using TilingInterface to produce kgen.for";
  let constructor = "::mlir::kgen::createTilingPass()";
  let options = [
    Option<"opName", "op-name", "std::string", /*default=*/"",
           "Operation with this name is the anchor to latch on.">,
    Option<"opLabel", "op-label", "std::string", /*default=*/"",
           "Operation with this label is the anchor to latch on.">,
    Option<"distribute", "distribute", "bool", /*default=*/"true",
           "Generate kgen.parallel or kgen.for">,
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
  ];
}

def TilingReductionPass : Pass<"kgen-tiling-reduction", "mlir::func::FuncOp"> {
  let summary = "Tile linalg.generic reductions";
  let constructor = "::mlir::kgen::createTilingReductionPass()";
  let dependentDialects = ["::mlir::kgen::kgen_Dialect",
                           "::mlir::arith::ArithDialect"];
}

def KgenToGpu2dPass : Pass<"kgen-to-gpu-2d", "mlir::func::FuncOp"> {
  let summary = "Lower nested 2D `kgen.parallel` to `gpu.launch`";
  let dependentDialects = ["::mlir::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
}

#endif // _kgen_PASSES_TD_INCLUDED_
