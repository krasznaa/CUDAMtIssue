
# The headers and sources of this library.
set( _headers IAuxTypeVector.cuh IAuxTypeVectorFactory.cuh AuxTypeVector.cuh )
set( _sources AuxTypeVector.cu )

# Set up the library.
add_library( ContainerLib OBJECT ${_headers} ${_sources} )
target_include_directories( ContainerLib
   PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/container> )
target_link_libraries( ContainerLib PRIVATE CoreLib )
set_target_properties( ContainerLib PROPERTIES
   PUBLIC_HEADER "${_headers}" )

# Install the library.
install( TARGETS ContainerLib
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/container )
