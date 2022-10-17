set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name kgen)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix kgen)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix kgen)
add_public_tablegen_target(IMEXkgenPassIncGen)

add_mlir_doc(Passes kgenPasses ./ -gen-pass-doc)
