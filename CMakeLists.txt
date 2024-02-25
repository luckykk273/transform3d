cmake_minimum_required(VERSION 3.0)
project(transform3d VERSION 1.0.0 DESCRIPTION "3D transformations library")
message("Project = ${PROJECT_NAME}, path = ${CMAKE_CURRENT_LIST_DIR}")

# Set default build type:
if(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
endif()
message("Build type = ${CMAKE_BUILD_TYPE}")

# Set compilers:
set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++.exe")
message("CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}, CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")

# Set standards:
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
message("C_STANDARD = ${CMAKE_C_STANDARD}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
message("CXX_STANDARD = ${CMAKE_CXX_STANDARD}")

# Set compiler flag options
set(CMAKE_C_FLAGS "-Wall -Werror")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wmissing-braces -Wno-format -Wmissing-declarations -Waddress")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpointer-arith -Wshadow -fno-strict-aliasing")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-misleading-indentation -Wunused-variable -Wno-unused-function -Wfloat-equal -g")

set(CMAKE_CXX_FLAGS "-MMD -MP")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fdata-sections -ffunction-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-field-initializers -fno-inline -fno-common -g")

# Add your source files here
file(GLOB TRANS_SRCS ${transform3d_SOURCE_DIR}/src/*.c)

# Default we only build the dynamic library for Python wrapper.
# Build the dynamic library
option(TRANS_BUILD_DYNAMIC "option for building transform3d dynamic library" ON)
if(TRANS_BUILD_DYNAMIC)
    message("Build transform3d dynamic library.")
    set(TRANS_DYNAMIC_LIB "${PROJECT_NAME}_dynamic")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${transform3d_BINARY_DIR}/dll)
    add_library(${TRANS_DYNAMIC_LIB} SHARED ${TRANS_SRCS})
    # Include directory for header files
    include_directories(${TRANS_DYNAMIC_LIB} PUBLIC ${transform3d_SOURCE_DIR}/inc)
endif()

# Build the static library
option(TRANS_BUILD_STATIC "option for building transform3d static library" OFF)
if(TRANS_BUILD_STATIC)
    message("Build transform3d static library.")
    set(TRANS_STATIC_LIB "${PROJECT_NAME}_static")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${transform3d_BINARY_DIR}/lib)
    add_library(${TRANS_STATIC_LIB} STATIC ${TRANS_SRCS})
    include_directories(${TRANS_STATIC_LIB} PUBLIC ${transform3d_SOURCE_DIR}/inc)
endif()

# Build the executable file
option(TRANS_BUILD_EXE "option for building transform3d executable file" OFF)
if(TRANS_BUILD_EXE)
    message("Build transform3d executable file.")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${transform3d_BINARY_DIR}/bin)
    add_executable(${PROJECT_NAME} ${transform3d_SOURCE_DIR}/main.c ${TRANS_SRCS})
    include_directories(${PROJECT_NAME} PRIVATE ${transform3d_SOURCE_DIR}/inc)
endif()