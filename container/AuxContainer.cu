// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "AuxContainer.cuh"
#include "AuxTypeRegistry.h"
#include "AuxTypeVector.cuh"

// System include(s).
#include <cassert>
#include <cstring>

__host__
AuxContainer::AuxContainer()
: m_size( 0 ), m_vecs(), m_deviceBuffer( nullptr ) {

}

__host__ __device__
AuxContainer::AuxContainer( std::size_t size, std::size_t nVars,
                            void** vars )
   : m_size( size ), m_vecs( nVars ), m_deviceBuffer( nullptr ) {

   // Set up the internal variable with simple objects wrapping around the
   // received memory pointers.
   for( std::size_t i = 0; i < nVars; ++i ) {
      if( ! vars[ i ] ) {
         m_vecs[ i ] = nullptr;
      } else {
         m_vecs[ i ] = new AuxTypeVector< char, char* >( vars[ i ], size );
      }
   }
}

__host__ __device__
AuxContainer::~AuxContainer() {

   const std::size_t s = m_vecs.size();
   for( std::size_t i = 0; i < s; ++i ) {
      if( m_vecs[ i ] ) {
         delete m_vecs[ i ];
      }
   }
#ifndef __CUDA_ARCH__
   if( m_deviceBuffer ) {
      delete m_deviceBuffer;
   }
#endif // not __CUDA_ARCH__
}

/// This function behaves exactly the same as @c size(). The only reason
/// that it exists is that @c SG::IConstAuxStore defines @c size() as a
/// host-only function.
///
/// @return The size of the arrays held by the store
///
__host__ __device__
std::size_t AuxContainer::arraySize() const {

   return m_size;
}

/// This function can be used to transfer variables to a CUDA kernel call.
/// The return values of the function should be possible to use as kernel
/// function arguments.
///
/// @param stream The stream to perform the memory copy with
/// @return Variables to pass to a CUDA kernel
///
__host__
std::pair< std::size_t, void** >
AuxContainer::variables( cudaStream_t stream ) {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // If there are no variables that can be exposed to the GPU, return
   // quickly.
   if( m_vecs.size() == 0 ) {
      return std::pair< std::size_t, void** >( 0, nullptr );
   }

   // Prepare the host array with the content that we need.
   const std::size_t s = m_vecs.size();
   m_hostBuffer.resize( s );
   for( std::size_t i = 0; i < s; ++i ) {
      if( m_vecs[ i ] ) {
         m_hostBuffer[ i ] = m_vecs[ i ]->attachTo( stream );
      } else {
         m_hostBuffer[ i ] = nullptr;
      }
   }

   // Do different things based on whether we actually send the data to
   // a GPU or not.
   if( stream ) {

      // Create the on-device buffer.
      if( m_deviceBuffer == nullptr ) {
         m_deviceBuffer = new cuda::device_array< void* >();
      }
      *m_deviceBuffer = cuda::make_device_array< void* >( s );
      CUDA_CHECK( cudaMemcpyAsync( m_deviceBuffer->get(), m_hostBuffer.get(),
                                   s * sizeof( void* ),
                                   cudaMemcpyHostToDevice,
                                   stream ) );
      // Return the buffer.
      return std::pair< std::size_t, void** >( s, m_deviceBuffer->get() );
   } else {
      // Return the buffer on the host.
      return std::pair< std::size_t, void** >( s, m_hostBuffer.get() );
   }
}

/// This function needs to be called after a kernel has finished its
/// execution, to ensure that all variables are copied back from the device
/// into the host's memory.
///
/// @param stream The stream to perform the memory copy with
///
__host__
void AuxContainer::retrieveFrom( cudaStream_t stream ) {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // Ask all vectors to pull their data back.
   for( std::size_t i = 0; i < m_vecs.size(); ++i ) {
      if( m_vecs[ i ] ) {
         m_vecs[ i ]->retrieveFrom( stream );
      }
   }
   return;
}

__host__
const void* AuxContainer::getData( std::size_t auxid ) const {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // Check whether we have this variable.
   if( ( auxid >= m_vecs.size() ) || ( m_vecs[ auxid ] == nullptr ) ) {
      return nullptr;
   }

   // Return it if we do.
   return m_vecs[ auxid ]->toPtr();
}

__host__
void* AuxContainer::getData( std::size_t auxid, std::size_t size,
                             std::size_t capacity ) {

   // Try to allocate the variable in CUDA Unified Memory.
   void* ptr = getDataCUDA( auxid, size, capacity );
   if( ptr ) {
      return ptr;
   }

   // This should not really happen, but apparently something went very
   // wrong...
   return 0;
}

__host__
bool AuxContainer::resize( std::size_t size ) {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // Helper variable.
   bool nomove = true;
   // Resize all the vectors.
   const std::size_t v_size = m_vecs.size();
   for( std::size_t i = 0; i < v_size; ++i ) {
      IAuxTypeVector* v = m_vecs[ i ];
      if( ! v ) {
         continue;
      }
      if( ! v->resize( size ) ) {
         nomove = false;
      }
   }
   // Remember the new size.
   m_size = size;
   // Return whether the iterators *not* have been invalidated.
   return nomove;
}

__host__
void AuxContainer::reserve( std::size_t capacity ) {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // Reserve the memory in all vectors.
   const std::size_t v_size = m_vecs.size();
   for( std::size_t i = 0; i < v_size; ++i ) {
      IAuxTypeVector* v = m_vecs[ i ];
      if( ! v ) {
         continue;
      }
      v->reserve( capacity );
   }
   return;
}

__host__
void AuxContainer::shift( std::size_t pos, std::ptrdiff_t offs ) {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // Perform the shift on all vectors.
   const std::size_t v_size = m_vecs.size();
   for( std::size_t i = 0; i < v_size; ++i ) {
      IAuxTypeVector* v = m_vecs[ i ];
      if( ! v ) {
         continue;
      }
      v->shift( pos, offs );
   }
   return;
}

__host__
void* AuxContainer::getDataCUDA( std::size_t auxid, std::size_t size,
                                 std::size_t capacity ) {

   // Protect the function from parallel execution.
   lock_t guard( m_mutex );

   // Check if we already have this variable.
   if( ( m_vecs.size() > auxid ) && ( m_vecs[ auxid ] != nullptr ) ) {
      return m_vecs[ auxid ]->toPtr();
   }

   // Check if we can instantiate such a variable in CUDA Unified Memory.
   IAuxTypeVector* v = AuxTypeRegistry::instance().makeVector( auxid, size,
                                                               capacity );
   if( ! v ) {
      return 0;
   }

   // Get hold of the variable.
   if( m_vecs.size() <= auxid ) {
      m_vecs.resize( auxid + 1 );
   }
   m_vecs[ auxid ] = v;
   return v->toPtr();
}
