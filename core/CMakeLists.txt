
# Choose whether to use a threads library.
set( _threadLib )
if( TARGET Threads::Threads )
   set( _threadLib Threads::Threads )
endif()

# The headers and sources of this library.
set( _headers Info.h Macros.cuh Memory.cuh Memory.icc StatusCode.h
   StreamPool.cuh )
set( _sources Info.cu Memory.cu StatusCode.cxx StreamPool.cu )

# Set up the library.
add_library( CoreLib SHARED ${_headers} ${_sources} )
target_include_directories( CoreLib
   PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/core>
   PRIVATE ${TBB_INCLUDE_DIRS} )
target_link_libraries( CoreLib PRIVATE ${TBB_LIBRARIES} ${_threadLib} )
set_target_properties( CoreLib PROPERTIES
   PUBLIC_HEADER "${_headers}" )

# Install the library.
install( TARGETS CoreLib
   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
   PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/core )
