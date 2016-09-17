cmake_minimum_required(VERSION 3.5)
project(KNN_MPI)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(KNN_MPI ${SOURCE_FILES})