
# The headers and sources of this library.
set( _headers KernelTask.cuh KernelStatus.cuh KernelRunnerSvc.cuh
   KernelRun.cuh KernelRun.icc )
set( _sources KernelTask.cu KernelStatus.cu KernelRunnerSvc.cu )

# Set up the library.
add_library( KernelLib OBJECT ${_headers} ${_sources} )
target_include_directories( KernelLib
   PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kernel>
          ${TBB_INCLUDE_DIRS} )
target_link_libraries( KernelLib PUBLIC ${TBB_LIBRARIES} CoreLib ContainerLib )
set_target_properties( KernelLib PROPERTIES
   PUBLIC_HEADER "${_headers}"
   CUDA_SEPARABLE_COMPILATION TRUE
   POSITION_INDEPENDENT_CODE  TRUE )

# Install the library.
install( TARGETS KernelLib
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   OBJECTS DESTINATION ${CMAKE_INSTALL_LIBDIR}
   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/kernel )
