cmake_minimum_required(VERSION 3.14)
project(ShepardInterpolation)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Where to find CMake modules and OpenCV
# SET(OpenCV_DIR "C:/opencv-4.1.1/mingw-build/install_release")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
include(MacroOpenCV)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Setup path to other libs
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib/")

add_subdirectory(src)
