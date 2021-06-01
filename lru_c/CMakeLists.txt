cmake_minimum_required(VERSION 2.8)
project (lru_utils)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
find_package(Python COMPONENTS Development)
# message(STATUS ${Python_INCLUDE_DIRS})

include_directories(${Python_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

file(GLOB src_files "${PROJECT_SOURCE_DIR}/*.cc")

add_library(lru_utils SHARED ${src_files})
set_target_properties(lru_utils PROPERTIES PREFIX "")
