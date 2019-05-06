
# Set up the project.
cmake_minimum_required( VERSION 3.13 )
project( CUDAMtIssue VERSION 1.0.0 LANGUAGES CXX CUDA )

# Use C++11.
set( CMAKE_CXX_STANDARD_REQUIRED 11 CACHE STRING
   "The C++ version to use" )
set( CMAKE_CXX_EXTENSIONS OFF CACHE BOOL
   "(Dis)Allow C++ compiler extensions" )

# Make sure that in a debug build we would get debug symbols in the device
# code.
if( NOT "${CMAKE_CUDA_FLAGS_DEBUG}" MATCHES "-G" )
   set( CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G" )
endif()

# Set up the local CMake module(s).
list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )

# External(s) needed by the project.
find_package( Boost COMPONENTS program_options REQUIRED )
find_package( TBB REQUIRED )
find_package( Threads )

# Use the GNU Install directory layout.
include( GNUInstallDirs )

# Use CTest for testing the code.
include( CTest )

# Build all subdirectories.
add_subdirectory( core )
add_subdirectory( container )
add_subdirectory( kernel )
add_subdirectory( tests )
