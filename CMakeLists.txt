cmake_minimum_required(VERSION 3.0)

# under Windows:
set(CMAKE_CXX_COMPILER "C:/msys64/mingw64/bin/g++.exe")
set(CMAKE_C_COMPILER "C:/msys64/mingw64/bin//gcc.exe")

project(transformation VERSION 1.0.0 DESCRIPTION "3D transformations library")

# Include directory for header files
include_directories(inc)

# Add your source files here
file(GLOB SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.c)

message(${CMAKE_CURRENT_LIST_DIR})

# Build the dynamic library
add_library(transformation SHARED ${SOURCES})

# Not necessary; for testing
add_executable(main
    main.c
    ${SOURCES}
)