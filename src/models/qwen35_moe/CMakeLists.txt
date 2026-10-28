# Qwen3.5-MoE (qwen35moe) Ornith-1.5-122B model package with ROCmFP4 quants.
add_library(gufo_qwen35_moe STATIC
  config.cpp
  weights.cpp
  forward.cpp
  engine.cpp)

target_include_directories(gufo_qwen35_moe PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(gufo_qwen35_moe PUBLIC gufo_core Threads::Threads)
