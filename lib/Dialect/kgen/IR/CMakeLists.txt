add_mlir_dialect_library(IMEXkgenDialect
  kgenOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/imex/Dialect/kgen

  DEPENDS
  IMEXkgenOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
