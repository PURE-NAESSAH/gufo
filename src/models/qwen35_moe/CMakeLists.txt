# Qwen3.5-MoE (qwen35moe) Ornith-1.5-122B model package with ROCmFP4 quants.
add_library(gufo_qwen35_moe STATIC
  config.cpp
  weights.cpp
  forward.cpp
  engine.cpp)

target_include_directories(gufo_qwen35_moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen35_moe PUBLIC gufo_core Threads::Threads)

if (ENGINE_ENABLE_HIP)
  target_sources(gufo_qwen35_moe PRIVATE
    hip/kernels.hip
    hip/gpu_runner.cpp)
  target_compile_definitions(gufo_qwen35_moe PRIVATE ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_qwen35_moe PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:/O2;/fp:fast;/clang:-O3;/clang:-ffast-math;/clang:-fno-finite-math-only;/clang:-Wno-unused-value;/clang:-Wno-unused-command-line-argument>)
  target_link_libraries(gufo_qwen35_moe PUBLIC
    hip::host
    roc::hipblas)
endif()
