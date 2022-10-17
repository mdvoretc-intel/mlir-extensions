add_mlir_dialect_library(IMEXkgenTransforms
  kgen_to_gpu_2d.cpp
  tiling.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/kgen

  DEPENDS
  IMEXkgenPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  IMEXkgenDialect
)
