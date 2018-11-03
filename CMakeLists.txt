cmake_minimum_required(VERSION 2.6)
project(extract_table_node)

set( CMAKE_BUILD_TYPE "Debug" )

#c++11标准
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(THREADS_PREFER_PTHREAD_FLAG ON)

set( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )

find_package( OpenCV REQUIRED )
find_package(Threads REQUIRED)

add_subdirectory( src )
