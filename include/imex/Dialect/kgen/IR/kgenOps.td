//===- kgenOps.td - kgen dialect  -------*- tablegen -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the basic operations for the kgen dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _kgen_OPS_TD_INCLUDED_
#define _kgen_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def ComposeSetInterface : OpInterface<"ComposeSetInterface"> {
  let description = [{
    This interface should be implemented by all set operations that can be
    composed with their superset operand.
  }];
  let cppNamespace = "::mlir::kgen";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns a composition of this set with its superset operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"compose",
      /*args=*/(ins "OpBuilder&":$builder)>,
  ];
}

def RankedTensorOrVector
  : AnyTypeOf<[AnyRankedTensor, AnyVector], "", "::mlir::ShapedType">;
def RankedTensorOrVectorOrScalar : AnyTypeOf<[
  AnyRankedTensor, AnyVector, AnyFloat, AnyInteger, AnyComplex
]>;

// Provide a definition of the 'kgen' dialect in the ODS framework so that we
// can define our operations.
def kgen_Dialect : Dialect {
    // The namespace of our dialect
    let name = "kgen";

    // A short one-line summary of our dialect.
    let summary = "ML workload tiling for GPU";

    // A longer description of our dialect.
    let description = [{
            The kgen dialect takes linalg primitives as inputs and
            provides tiling and initial outlining for GPU offloading
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::kgen";

    let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
    let useDefaultTypePrinterParser = 1;
    let hasConstantMaterializer = 1;
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class kgen_Op<string mnemonic, list<Trait> traits = []> :
    Op<kgen_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

def kgen_LoopOp : kgen_Op<"loop", [
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveSideEffects,
     SingleBlockImplicitTerminator<"kgen::YieldOp">
    ]> {
  let summary = "Loop-like operation";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the input and the output tensors or memrefs and the attributes
    to specify the iterator types.

    Parsing LoopOp will set all elements of the `iterator_types` attribute
    to "parallel" type, when it is absent from the custom format.

    Tensor-based version:

    The body region of the loop contains `extract_slice` operations applied to
    every tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `kgen.yield` with the operands resulting from `insert_slice` operations.

    Example:

    ```mlir
    %0 = kgen.loop (%i) = (%c0) to (%c24) step (%c4)
        ins(%lhs, %rhs : tensor<24x64xi8>, tensor<24x64xi8>)
        outs(%out : tensor<24x64xi8>)
        iterators("parallel")
        distribution("block_x") {
      %lhs_sub = tensor.extract_slice %lhs[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>
      %rhs_sub = tensor.extract_slice %rhs[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>
      %out_sub = tensor.extract_slice %out[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>

      %result_sub = linalg.generic ...

      %result = tensor.insert_slice %result_sub into %out[%i, 0][%c4, %c64][1, 1]
        : tensor<?x?xi8> into tensor<24x64xi8>
      kgen.yield %result : tensor<24x64xi8>
    }
    ```

    MemRef-based version:

    The body region of the loop contains `subview` operations applied to
    every memref argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `kgen.yield` with no operands.

    Example:

    ```mlir
    kgen.loop (%i) = (%c0) to (%c24) step (%c4)
        ins(%lhs, %rhs : memref<24x64xi8>, memref<24x64xi8>)
        outs(%out : memref<24x64xi8>)
        iterators("parallel")
        distribution("block_x") {
      %lhs_sub = subview %lhs[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>
      %rhs_sub = subview %rhs[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>
      %out_sub = subview %out[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>

      %result_sub = linalg.generic ...
      kgen.yield
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       ArrayAttr:$iterator_types,
                       OptionalAttr<ArrayAttr>:$distribution_types);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$iteratorTypes, "Optional<ArrayAttr>":$distributionTypes,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*inputs=*/ValueRange, /*outputs=*/ValueRange)>",
        "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$iteratorTypes,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*inputs=*/ValueRange, /*outputs=*/ValueRange)>",
        "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return getStep().size(); }

    /// Number of input operands
    unsigned getNumInputs() { return getInputs().size(); }

    /// Number of output operands
    unsigned getNumOutputs() { return getOutputs().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }
    ValueRange getRegionInputArgs() {
      return getBody()->getArguments().slice(getNumLoops(), getInputs().size());
    }
    ValueRange getRegionOutputArgs() {
      return getBody()->getArguments().take_back(getOutputs().size());
    }

    void setDistributionTypes(Builder& b, ArrayRef<StringRef> types) {
      assert(types.size() == getNumLoops() &&
             "expected distribution type for every dimension");
      setDistributionTypesAttr(b.getStrArrayAttr(types));
    }

    void setLowerBounds(ValueRange lowerBounds) {
      unsigned numLoops = getNumLoops();
      assert(lowerBounds.size() == numLoops &&
             "expected lower bounds for every loop dimension");
      for (unsigned i = 0; i < numLoops; ++i)
        setOperand(i, lowerBounds[i]);
    }

    void setUpperBounds(ValueRange upperBounds) {
      unsigned numLoops = getNumLoops();
      assert(upperBounds.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, upperBounds[i]);
    }

    void setSteps(ValueRange steps) {
      unsigned numLoops = getNumLoops();
      assert(steps.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = 2 * numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, steps[i]);
    }

    /// Operand that corresponds to the `bbArg` block argument.
    OpOperand& getTiedOperand(BlockArgument& bbArg) {
      return getOperation()->getOpOperand(getNumControlOperands() +
                                          bbArg.getArgNumber() - getNumLoops());
    }

    /// Block argument that corresponds to the `input` or `output` operand.
    BlockArgument getTiedBlockArgument(OpOperand& operand) {
      auto operandIndex = operand.getOperandNumber();
      assert(
          operandIndex >= getNumControlOperands() &&
          operandIndex < getNumOperands() &&
          "tied block arg is defined only for `input` and `output` arguments");
      return getBody()->getArgument(operandIndex - 2 * getNumLoops());
    }

   /// Result that corresponds to the `outputs` argument of tensor type.
   OpResult getTiedOpResult(OpOperand& opOperand) {
      // No result can correspond to a memref argument.
      if (opOperand.get().getType().isa<MemRefType>()) return OpResult();

      // Check whether the operand index is in bounds of `outputs()` arg.
      int operandIndex = opOperand.getOperandNumber();
      int outputIndexStart =
          getNumControlOperands() + getInputs().size();
      int outputIndexEnd = outputIndexStart + getOutputs().size();
      if (operandIndex < outputIndexStart || operandIndex >= outputIndexEnd)
        return OpResult();

      // Count tensor arguments in `outputs` to compute the result index.
      int tensorId = -1;
      for (int i = outputIndexStart; i <= operandIndex; ++i)
        tensorId += getOperand(i).getType().isa<RankedTensorType>();
      return getOperation()->getResult(tensorId);
    }

    /// Append `operand` to the `input` arguments.
    OpOperand& appendInputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(getNumControlOperands() + numInputs,
                                     operand);
      getBody()->insertArgument(numLoops + numInputs, operand.getType(),
                                getLoc());
      getOperation()->setAttr(
          LoopOp::getOperandSegmentSizeAttr(),
          builder.getDenseI32ArrayAttr(
              {numLoops, numLoops, numLoops, numInputs + 1, numOutputs}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs);
    }

    /// Append `operand` to the `output` arguments.
    OpOperand& appendOutputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(
          getNumControlOperands() + numInputs + numOutputs, operand);
      getBody()->insertArgument(numLoops + numInputs + numOutputs,
                                operand.getType(), getLoc());
      getOperation()->setAttr(
          LoopOp::getOperandSegmentSizeAttr(),
          builder.getDenseI32ArrayAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs + 1}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs +
                                          numOutputs);
    }

    /// Erase `operand` from the `input` or `output` arguments.
    void eraseOperand(OpBuilder& builder, OpOperand& operand) {
      int numInputs = getNumInputs();
      int numLoops = getNumLoops();
      int numOutputs = getNumOutputs();
      int numControlOperands = getNumControlOperands();

      int operandIndex = operand.getOperandNumber();
      assert(operandIndex >= numControlOperands &&
             operandIndex < static_cast<int>(getNumOperands()) &&
             "Can erase only `input` or `output` operand");

      if (operandIndex >= numControlOperands + numInputs)
        --numOutputs;
      else
        --numInputs;

      getOperation()->eraseOperand(operandIndex);
      getBody()->eraseArgument(operandIndex - 2 * numLoops);
      getOperation()->setAttr(
          LoopOp::getOperandSegmentSizeAttr(),
          builder.getDenseI32ArrayAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs}));
    }

    OpOperand* findInputOperand(Value value) {
      OperandRange::iterator it = llvm::find(getInputs(), value);
      if (it == getInputs().end()) return nullptr;
      return it.getBase();
    }

    OpOperand* findOutputOperand(Value value) {
      OperandRange::iterator it = llvm::find(getOutputs(), value);
      if (it == getOutputs().end()) return nullptr;
      return it.getBase();
    }

    /// Return whether the op has only MemRef input and outputs.
    bool hasBufferSemantics() {
      Operation* op = this->getOperation();
      return op->getNumResults() == 0 &&
             llvm::all_of(op->getOpOperands(), [&](OpOperand & operand) {
               return !operand.get().getType().template isa<ShapedType>() ||
                      operand.get().getType().template isa<MemRefType>();
             });
    }

    static constexpr StringRef getParallelIteratorTypeName() {
      return "parallel";
    }
    static constexpr StringRef getDistributionTypesAttrStrName() {
      return "distribution_types";
    }
    static constexpr StringRef getIteratorTypesAttrStrName() {
      return "iterator_types";
    }


    /// Return whether the loop dimension is parallel or not.
    bool isParallelDimension(unsigned dim) {
      StringAttr attr = this->getIteratorTypes()[dim].cast<StringAttr>();
      return attr.getValue() == getParallelIteratorTypeName();
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

def kgen_YieldOp : kgen_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
    HasParent<"::mlir::kgen::LoopOp, ::mlir::kgen::SetYieldOp">]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Yield operation";
  let description = [{
    `kgen.yield` is a special terminator operation for `kgen.loop` body or
    for accumulator regions of `kgen.set_yield`.

    Example:

    ```mlir
    kgen.yield %f0, %f1 : tensor<f32>, tensor<?xf32>
    ```
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

// Base class of all subset types.
class kgen_Set<string name> : TypeDef<kgen_Dialect, name> { }

def kgen_TileType : kgen_Set<"Tile"> {
  let mnemonic = "tile";
  let summary = "Type that represents a tile of an index space.";
  let parameters = (ins ArrayRefParameter<"int64_t">:$shape);
  let assemblyFormat = "`<` custom<ShapeTypeDimensionsList>($shape) `>`";
  let extraClassDeclaration = [{
    unsigned getRank() const { return getShape().size(); }
    bool hasStaticShape() const {
      return llvm::none_of(getShape(), ShapedType::isDynamic);
    }
    int64_t getNumElements() const {
      return ShapedType::getNumElements(getShape());
    }
  }];
}

// Whether a type is a subset type.
def IsSetTypePred : Or<[kgen_TileType.predicate]>;

def AnySet : Type<IsSetTypePred, "subset type">;

def kgen_SpaceOp : kgen_Op<"space", [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins Variadic<Index>:$dynamic_sizes,
                       I64ArrayAttr:$static_sizes);
  let results = (outs kgen_TileType:$result);
  let assemblyFormat = [{
    custom<DynamicIndexList>($dynamic_sizes, $static_sizes,
                               "ShapedType::kDynamicSize")
    attr-dict `:` qualified(type($result))
  }];
  let builders = [
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$sizes,
                    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];
  let extraClassDeclaration = [{
    unsigned getNumDynamicEntriesUpToIdx(unsigned idx);
    mlir::Value getDynamicSize(unsigned idx);
  }];
  let hasVerifier = 1;
}


def kgen_TileOp : kgen_Op<"tile", [
    NoSideEffect,
    AttrSizedOperandSegments,
    OffsetSizeAndStrideOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ComposeSetInterface>]> {
  let arguments = (ins kgen_TileType:$superset,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       I64ArrayAttr:$static_offsets,
                       I64ArrayAttr:$static_sizes,
                       I64ArrayAttr:$static_strides);
  let results = (outs kgen_TileType:$result);
  let assemblyFormat = [{
    $superset
    custom<DynamicIndexList>($offsets, $static_offsets,
                               "ShapedType::kDynamicStrideOrOffset")
    custom<DynamicIndexList>($sizes, $static_sizes,
                               "ShapedType::kDynamicSize")
    custom<DynamicIndexList>($strides, $static_strides,
                               "ShapedType::kDynamicStrideOrOffset")
    attr-dict `:` qualified(type($superset)) `to` qualified(type($result))
  }];
  let builders = [
   OpBuilder<(ins "Value":$superset, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];
  let extraClassDeclaration = [{
    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSuperset().getType().cast<TileType>().getRank();
      return {rank, rank, rank};
    }
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];
  let hasVerifier = 1;
}

def kgen_MaterializeOp : kgen_Op<"materialize", [NoSideEffect]> {
  let arguments = (ins RankedTensorOrVector:$source, AnySet:$set);
  let results = (outs RankedTensorOrVectorOrScalar:$result);

  let builders = [OpBuilder<(ins "Value":$source, "Value":$set)>];

  let assemblyFormat = [{
    $source`[` $set `]` attr-dict `:` type($source) `[` type($set) `]`
      `to` type($result)
  }];
}

class kgen_LoopLikeOp<string mnemonic, list<Trait> traits = []>
    : kgen_Op<mnemonic, !listconcat(traits, [
      AttrSizedOperandSegments,
      DeclareOpInterfaceMethods<LoopLikeOpInterface>,
      RecursiveSideEffects,
      SingleBlockImplicitTerminator<"gml_st::SetYieldOp">
    ])> {
  let results = (outs Variadic<RankedTensorOrVector>:$results);
  let regions = (region SizedRegion<1>:$region);

  code extraBaseClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return getStep().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }

    /// Return whether the op has no output tensors.
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0;
    }

    /// Return terminator of the loop body.
    SetYieldOp getTerminator();
  }];

  let hasCustomAssemblyFormat = 1;
}

def kgen_ParallelOp : kgen_LoopLikeOp<"parallel", []> {
  let summary = "Loop-like operation for parallel loops";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the output tensors or memrefs.

    Tensor-based version:

    The body region of the loop contains set operations applied to
    every output tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `kgen.set_yield` which yields a tensor into a subset of outs.

    Example:

    ```mlir
    %space = kgen.space [8, 16] : !kgen.tile<8x16>

    %result = kgen.parallel (%i) = (%c0, %c0) to (%c8, %c16) step (%c4, %c4) {
      %tile = kgen.tile %space [%i, %j] [4, 4] [1, 1]
        : ! kgen.tile<8x16> to !kgen.tile<4x4>

      %lhs_sub = kgen.materialize %lhs_[%tile]
        : tensor<8x16xf32>[!kgen.tile<4x4>]
      %rhs_sub = kgen.materialize %rhs_[%tile]
        : tensor<8x16xf32>[!kgen.tile<4x4>]
      %out_sub = kgen.materialize %out_[%tile]
        : tensor<8x16xf32>[!kgen.tile<4x4>]

      %result_sub = linalg.generic (%lhs_sub, %rhs_sub, %out_sub) ...

      kgen.set_yield %result_sub into %out[%tile]
        : tensor<4x4xf32> into tensor<16x64xf32>[!kgen.tile<4x4>]
    }
    ```
  }];
  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       OptionalAttr<StrAttr>:$distributionType);
  // The default builder does not generate the block with induction variables
  // as arguments, and conflicts with the custom one. Prevent tablegen from
  // generating it.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$lowerBounds,
      "ValueRange":$upperBounds, "ValueRange":$steps,
      CArg<"Optional<StringAttr>", "llvm::None">:$distributionType,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange)>",
      "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = extraBaseClassDeclaration;
}

def kgen_SetYieldOp : kgen_Op<"set_yield", [NoSideEffect, ReturnLike,
      Terminator, SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Set yield operation";
  let description = [{
    `kgen.set_yield` is a special terminator operation for
    `kgen.parallel` or `kgen.for` body.

    Example:

    ```mlir
    kgen.set_yield %result_sub at %tile into %dst
      : tensor<4x4xf32> into tensor<16x64xf32>[!kgen.tile<4x4>]
    ```
  }];
  let arguments = (ins Variadic<RankedTensorOrVectorOrScalar>:$srcs,
                       Variadic<RankedTensorOrVector>:$dsts,
                       Variadic<AnySet>:$sets,
                       BoolArrayAttr:$accumulatorFlags);
  let regions = (region VariadicRegion<SizedRegion<1>>:$accumulators);
  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins)>,

    // Builder with default update behaviour, i.e. overriding output.
    OpBuilder<(ins "ValueRange":$srcs, "ValueRange":$dsts, "ValueRange":$sets)>,

    // Builder with custom update behaviour.
    OpBuilder<(ins "ValueRange":$srcs, "ValueRange":$dsts, "ValueRange":$sets,
      "ArrayAttr":$accumulatorFlags,
      "ArrayRef<function_ref<void(OpBuilder &, Location, Value, Value)>>"
      :$combiners)>
  ];

  let extraClassDeclaration = [{

    unsigned getNumUpdates() { return getSrcs().size(); }

    // Methods for `dst` arguments.
    OpOperand* getDstOperand(unsigned i) {
      return &getOperation()->getOpOperand(getNumUpdates() + i);
    }

    FailureOr<OpResult> getTiedOpResult(OpOperand &opOperand) {
      if (!isDstOperand(opOperand)) return failure();

      auto parent = getOperation()->getBlock()->getParentOp();
      if (isa<ParallelOp>(parent) || isa<ForOp>(parent)) {
        return parent->getResult(opOperand.getOperandNumber() -
                                 getNumUpdates());
      }
      return failure();
    }

    bool isDstOperand(OpOperand& operand) {
      return operand.getOperandNumber() >= getNumUpdates() &&
             operand.getOperandNumber() < getNumUpdates() * 2;
    }
  }];
}

#endif // _kgen_OPS_TD_INCLUDED_
