
#
# Test(s) using the core library:
#

# Set up the Info test.
add_executable( test_Info test_Info.cxx )
target_link_libraries( test_Info PRIVATE CoreLib )
add_test( NAME ctest_Info COMMAND test_Info )

# Set up the Info test.
add_executable( test_Memory test_Memory.cu )
target_link_libraries( test_Memory PRIVATE CoreLib )
add_test( NAME ctest_Memory COMMAND test_Memory )

# Set up the StatusCode test.
add_executable( test_StatusCode test_StatusCode.cxx )
target_link_libraries( test_StatusCode PRIVATE CoreLib )
add_test( NAME ctest_StatusCode COMMAND test_StatusCode )

#
# Test(s) using the container library:
#

# Set up the AuxContainer test(s).
add_executable( test_AuxContainer1 test_AuxContainer1.cu )
target_link_libraries( test_AuxContainer1 PRIVATE ContainerLib )
set_target_properties( test_AuxContainer1 PROPERTIES
   CUDA_SEPARABLE_COMPILATION TRUE )
add_test( NAME ctest_AuxContainer1 COMMAND test_AuxContainer1 )

add_executable( test_AuxContainer2 test_AuxContainer2.cu )
target_link_libraries( test_AuxContainer2 PRIVATE ContainerLib )
set_target_properties( test_AuxContainer2 PROPERTIES
   CUDA_SEPARABLE_COMPILATION TRUE )
add_test( NAME ctest_AuxContainer2 COMMAND test_AuxContainer2 )

# Set up the AuxTypeRegistry test.
add_executable( test_AuxTypeRegistry test_AuxTypeRegistry.cu )
target_link_libraries( test_AuxTypeRegistry PRIVATE ContainerLib )
set_target_properties( test_AuxTypeRegistry PROPERTIES
   CUDA_SEPARABLE_COMPILATION TRUE )
add_test( NAME ctest_AuxTypeRegistry COMMAND test_AuxTypeRegistry )

# Set up the AuxTypeVector test.
add_executable( test_AuxTypeVector test_AuxTypeVector.cu )
target_link_libraries( test_AuxTypeVector PRIVATE ContainerLib )
set_target_properties( test_AuxTypeVector PROPERTIES
   CUDA_SEPARABLE_COMPILATION TRUE )
add_test( NAME ctest_AuxTypeVector COMMAND test_AuxTypeVector )

# Set up the AuxTypeVectorFactory test.
add_executable( test_AuxTypeVectorFactory test_AuxTypeVectorFactory.cu )
target_link_libraries( test_AuxTypeVectorFactory PRIVATE ContainerLib )
set_target_properties( test_AuxTypeVectorFactory PROPERTIES
   CUDA_SEPARABLE_COMPILATION TRUE )
add_test( NAME ctest_AuxTypeVectorFactory COMMAND test_AuxTypeVectorFactory )

#
# Test(s) using the kernel library:
#

# Build a shared library with all of the test kernel code.
add_library( KernelTestsLib OBJECT
   kernels/AuxPtMultiply.cuh kernels/AuxPtMultiply.cu )
target_include_directories( KernelTestsLib
   PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> )
target_link_libraries( KernelTestsLib
   PUBLIC CoreLib ContainerLib
   PRIVATE KernelLib )
set_target_properties( KernelTestsLib PROPERTIES
   CUDA_SEPARABLE_COMPILATION TRUE
   POSITION_INDEPENDENT_CODE  TRUE )

# Set up the AuxPtMultiply test.
add_executable( test_AuxPtMultiply test_AuxPtMultiply.cu )
target_link_libraries( test_AuxPtMultiply
   PRIVATE CoreLib ContainerLib KernelLib KernelTestsLib )
add_test( NAME ctest_AuxPtMultiply COMMAND test_AuxPtMultiply )
