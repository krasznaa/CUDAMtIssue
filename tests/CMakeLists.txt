
#
# Test(s) using the core library:
#

# Set up the Info test.
add_executable( test_Info test_Info.cxx )
target_link_libraries( test_Info CoreLib )
add_test( NAME ctest_Info COMMAND test_Info )

# Set up the Info test.
add_executable( test_Memory test_Memory.cu )
target_link_libraries( test_Memory CoreLib )
add_test( NAME ctest_Memory COMMAND test_Memory )

# Set up the StatusCode test.
add_executable( test_StatusCode test_StatusCode.cxx )
target_link_libraries( test_StatusCode CoreLib )
add_test( NAME ctest_StatusCode COMMAND test_StatusCode )
