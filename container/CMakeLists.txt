
# The headers and sources of this library.
set( _headers IAuxTypeVector.cuh IAuxTypeVectorFactory.cuh AuxTypeVector.cuh
   AuxTypeVectorFactory.cuh AuxTypeRegistry.h AuxContainer.cuh
   AuxContainer.icc )
set( _sources AuxTypeVector.cu AuxTypeVectorFactory.cu AuxTypeRegistry.cu
   AuxContainer.cu )

# Set up the library.
add_library( ContainerLib OBJECT ${_headers} ${_sources} )
target_include_directories( ContainerLib
   PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/container>
          ${TBB_INCLUDE_DIRS} )
target_link_libraries( ContainerLib PRIVATE ${TBB_LIBRARIES} CoreLib )
set_target_properties( ContainerLib PROPERTIES
   PUBLIC_HEADER "${_headers}"
   CUDA_SEPARABLE_COMPILATION TRUE )

# Install the library.
install( TARGETS ContainerLib
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   OBJECTS DESTINATION ${CMAKE_INSTALL_LIBDIR}
   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/container )
