cmake_minimum_required(VERSION 3.10)
project(GpuMesherTest)

# ==========================================
# 1. SETUP ENV & CUDA
# ==========================================
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
find_package(CUDA REQUIRED)

# Need Python & Pybind11 found EARLY so we can use their paths in manual commands
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Standard Settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 2. DEFINE SOURCES
# ==========================================
set(GDEL3D_CPU_SOURCES
    gDel3D_modern/GDelFlipping/src/gDel3D/CPU/predicates.cpp
    gDel3D_modern/GDelFlipping/src/gDel3D/CPU/PredWrapper.cpp
    gDel3D_modern/GDelFlipping/src/gDel3D/CPU/Splaying.cpp
    gDel3D_modern/GDelFlipping/src/gDel3D/CPU/Star.cpp
)

set(GDEL3D_CUDA_SOURCES
    gDel3D_modern/GDelFlipping/src/gDel3D/GpuDelaunay.cu
    gDel3D_modern/GDelFlipping/src/gDel3D/GPU/KerDivision.cu
    gDel3D_modern/GDelFlipping/src/gDel3D/GPU/KerPredicates.cu
    gDel3D_modern/GDelFlipping/src/gDel3D/GPU/ThrustWrapper.cu
)

# ==========================================
# 3. MANUAL COMPILATION (CORE LIBRARY)
# ==========================================
set(CORE_OBJECTS "")

# Compile Core Files
foreach(source_file ${GDEL3D_CPU_SOURCES} ${GDEL3D_CUDA_SOURCES})
    get_filename_component(file_name ${source_file} NAME_WE)
    set(obj_file "${CMAKE_CURRENT_BINARY_DIR}/${file_name}.obj")
    
    add_custom_command(
        OUTPUT ${obj_file}
        COMMAND ${CUDA_NVCC_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} 
                -o ${obj_file}
                -I${CMAKE_CURRENT_SOURCE_DIR}/gDel3D_modern/GDelFlipping/src
                -I${CUDA_INCLUDE_DIRS}
                -I"${CUDA_TOOLKIT_ROOT_DIR}/include/cccl"
                --std=c++17
                -arch=sm_75
                -allow-unsupported-compiler
                -Xcompiler "/MD"
                -x cu
                -rdc=true
        DEPENDS ${source_file}
        COMMENT "Compiling ${source_file}"
    )
    list(APPEND CORE_OBJECTS ${obj_file})
endforeach()

# ==========================================
# 4. BUILD TEST EXECUTABLE (Optional)
# ==========================================
# Compile main.cpp separately
set(MAIN_OBJ "${CMAKE_CURRENT_BINARY_DIR}/main.obj")
add_custom_command(
    OUTPUT ${MAIN_OBJ}
    COMMAND ${CUDA_NVCC_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
            -o ${MAIN_OBJ}
            -I${CMAKE_CURRENT_SOURCE_DIR}/gDel3D_modern/GDelFlipping/src
            -I${CUDA_INCLUDE_DIRS}
            -I"${CUDA_TOOLKIT_ROOT_DIR}/include/cccl"
            --std=c++17
            -arch=sm_75
            -allow-unsupported-compiler
            -Xcompiler "/MD"
            -x cu
            -rdc=true
    DEPENDS main.cpp
)

# Link Executable
set(EXE_DLINK_OBJ "${CMAKE_CURRENT_BINARY_DIR}/exe_dlink.obj")
add_custom_command(
    OUTPUT ${EXE_DLINK_OBJ}
    COMMAND ${CUDA_NVCC_EXECUTABLE} -dlink ${CORE_OBJECTS} ${MAIN_OBJ}
            -o ${EXE_DLINK_OBJ}
            -arch=sm_75 -allow-unsupported-compiler -Xcompiler "/MD"
    DEPENDS ${CORE_OBJECTS} ${MAIN_OBJ}
)

add_executable(gpu_mesher_test ${CORE_OBJECTS} ${MAIN_OBJ} ${EXE_DLINK_OBJ})
target_link_libraries(gpu_mesher_test PRIVATE 
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib"
)
set_target_properties(gpu_mesher_test PROPERTIES LINKER_LANGUAGE CXX LINK_FLAGS "/NODEFAULTLIB:LIBCMT")

# ==========================================
# 5. BUILD PYTHON MODULE (_gpumesher)
# ==========================================

# A. Compile bindings.cpp
set(BINDINGS_OBJ "${CMAKE_CURRENT_BINARY_DIR}/bindings.obj")
add_custom_command(
    OUTPUT ${BINDINGS_OBJ}
    COMMAND ${CUDA_NVCC_EXECUTABLE} -c ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp
            -o ${BINDINGS_OBJ}
            -I${CMAKE_CURRENT_SOURCE_DIR}/gDel3D_modern/GDelFlipping/src
            -I${CUDA_INCLUDE_DIRS}
            -I"${CUDA_TOOLKIT_ROOT_DIR}/include/cccl"
            -I"${pybind11_SOURCE_DIR}/include"
            -I"${Python3_INCLUDE_DIRS}"
            --std=c++17
            -arch=sm_75
            -allow-unsupported-compiler
            -Xcompiler "/MD"
            -x cu
            -rdc=true
    DEPENDS bindings.cpp pybind11::module
    COMMENT "Compiling bindings.cpp"
)

# B. Device Link for Python (Must be separate from Exe link)
set(PY_DLINK_OBJ "${CMAKE_CURRENT_BINARY_DIR}/py_dlink.obj")
add_custom_command(
    OUTPUT ${PY_DLINK_OBJ}
    COMMAND ${CUDA_NVCC_EXECUTABLE} -dlink ${CORE_OBJECTS} ${BINDINGS_OBJ}
            -o ${PY_DLINK_OBJ}
            -arch=sm_75 -allow-unsupported-compiler -Xcompiler "/MD"
    DEPENDS ${CORE_OBJECTS} ${BINDINGS_OBJ}
)

# C. Create the Module
add_library(_gpumesher MODULE
    ${CORE_OBJECTS}
    ${BINDINGS_OBJ}
    ${PY_DLINK_OBJ}
)

# D. Link Dependencies (STATIC LINKING FIX)
target_link_libraries(_gpumesher PRIVATE 
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart_static.lib"
    "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudadevrt.lib"
    pybind11::module
)

# E. Properties (No Lib Prefix, .pyd Suffix)
set_target_properties(_gpumesher PROPERTIES 
    PREFIX "" 
    SUFFIX ".pyd"
    LINKER_LANGUAGE CXX
    LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/Release"
)