add_triton_ut(
  NAME TestOptimizeLDS
  SRCS OptimizeLDSTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonHCUGPUToLLVM
    MLIRUBToLLVM
    TritonHCUUtils
    TritonHCUAnalysis
    TritonHCUGPUTransforms
    TritonHCUGPUDialectToLLVM
)

if(FLAGTREE_TLE)
  add_triton_ut(
    NAME TestHCUStaticPipe
    SRCS StaticPipeTest.cpp
    LIBS
      TritonHCUGPUToLLVM
      MLIRArithDialect
  )
endif()
