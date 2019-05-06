
# Choose whether to use a threads library.
set( _threadLib )
if( TARGET Threads::Threads )
   set( _threadLib Threads::Threads )
endif()

# The headers and sources of this library.
set( _headers Info.h Macros.h Memory.h Memory.icc StatusCode.h StreamPool.h )
set( _sources Info.cu Memory.cu StatusCode.cxx StreamPool.cu )

# Set up the library.
add_library( CoreLib SHARED ${_headers} ${_sources} )
target_include_directories( CoreLib PRIVATE ${TBB_INCLUDE_DIRS} )
target_link_libraries( CoreLib PRIVATE ${TBB_LIBRARIES} ${_threadLib} )
set_target_properties( CoreLib PROPERTIES
   PUBLIC_HEADER "${_headers}" )
