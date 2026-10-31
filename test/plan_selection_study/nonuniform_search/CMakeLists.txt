cmake_minimum_required(VERSION 3.18)
project(vkfft_nonuniform_search LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(NONUNIFORM_CPU_ONLY "Build only isolated CPU tests" OFF)

enable_testing()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_executable(test_nonuniform_candidate test_candidate.cpp)
target_include_directories(test_nonuniform_candidate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
# Assertions in this executable are the test oracle, including Release builds.
if(MSVC)
  target_compile_options(test_nonuniform_candidate PRIVATE /UNDEBUG)
else()
  target_compile_options(test_nonuniform_candidate PRIVATE -UNDEBUG)
endif()
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_link_options(test_nonuniform_candidate PRIVATE -static)
endif()
add_test(NAME nonuniform_candidate_cpu COMMAND test_nonuniform_candidate)
add_test(NAME nonuniform_python_cpu COMMAND "${Python3_EXECUTABLE}" -m unittest discover
  -s "${CMAKE_CURRENT_SOURCE_DIR}" -p "test_*.py")

if(NOT NONUNIFORM_CPU_ONLY)
  set(NONUNIFORM_GENERATED_ROOT "" CACHE PATH
      "Parent of generated include and expanded_benchmark.cpp")
  get_filename_component(NONUNIFORM_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
  set(NONUNIFORM_ORIGINAL_INCLUDE_ROOT "${NONUNIFORM_REPOSITORY_ROOT}/VkFFT/vkFFT" CACHE PATH
      "Untouched VkFFT include root for the original control")
  if(NOT EXISTS "${NONUNIFORM_GENERATED_ROOT}/include/candidate.hpp" OR
     NOT EXISTS "${NONUNIFORM_GENERATED_ROOT}/expanded_benchmark.cpp")
    message(FATAL_ERROR "Generate the isolated headers and set NONUNIFORM_GENERATED_ROOT")
  endif()
  if(NOT EXISTS "${NONUNIFORM_ORIGINAL_INCLUDE_ROOT}/vkFFT.h")
    message(FATAL_ERROR "Untouched VkFFT include root not found")
  endif()
  find_package(CUDAToolkit REQUIRED)
  add_executable(vkfft_nonuniform_candidate "${NONUNIFORM_GENERATED_ROOT}/expanded_benchmark.cpp")
  target_include_directories(vkfft_nonuniform_candidate PRIVATE
    "${NONUNIFORM_GENERATED_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../instrumentation")
  target_compile_definitions(vkfft_nonuniform_candidate PRIVATE
    VKFFT_BACKEND=1 CUDA_TOOLKIT_ROOT_DIR="${CUDAToolkit_LIBRARY_ROOT}")
  target_link_libraries(vkfft_nonuniform_candidate PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)

  # The control uses the same harness, but resolves VkFFT headers from the
  # untouched source before the generated trace and benchmark support headers.
  add_executable(vkfft_nonuniform_original "${NONUNIFORM_GENERATED_ROOT}/expanded_benchmark.cpp")
  target_include_directories(vkfft_nonuniform_original PRIVATE
    "${NONUNIFORM_ORIGINAL_INCLUDE_ROOT}" "${NONUNIFORM_GENERATED_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../instrumentation")
  target_compile_definitions(vkfft_nonuniform_original PRIVATE
    VKFFT_BACKEND=1 VKFFT_STUDY_UNINSTRUMENTED=1 CUDA_TOOLKIT_ROOT_DIR="${CUDAToolkit_LIBRARY_ROOT}")
  target_link_libraries(vkfft_nonuniform_original PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  if(NOT MSVC)
    target_compile_options(vkfft_nonuniform_candidate PRIVATE -O3)
    target_compile_options(vkfft_nonuniform_original PRIVATE -O3)
  endif()
endif()
