// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "AuxTypeVector.cuh"

// System include(s).
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

   template< typename T >
   class ArrayTraits {};

   template< typename T >
   class ArrayTraits< T* > {
   public:
      /// Make a new array of a given size
      __host__
      static T* make_array( std::size_t ) {
         return nullptr;
      }
      /// Make a new array from an existing pointer
      __host__ __device__
      static T* make_array( void* ptr ) {
         return static_cast< T* >( ptr );
      }
      /// Get a pointer to the (beginning of the) array
      __host__ __device__
      static T* get( T* obj ) {
         return obj;
      }
      /// Swap two objects
      __host__ __device__
      static void swap( T*& obj1, T*& obj2 ) {
         T* temp = obj1;
         obj1 = obj2;
         obj2 = temp;
         return;
      }
      /// Attach the memory to a stream
      __host__ __device__
      static void* attach( T* obj, cudaStream_t ) {
         return obj;
      }
      /// Retrieve the memory from a device
      __host__ __device__
      static void retrieve( T*, cudaStream_t ) {
         return;
      }
   }; // class ArrayTraits

   template< typename T >
   class ArrayTraits< cuda::managed_array< T > > {
   public:
      /// Make a new array of a given size
      __host__
      static cuda::managed_array< T > make_array( std::size_t size ) {
         return cuda::make_managed_array< T >( size );
      }
      /// Get a pointer to the (beginning of the) array
#ifdef __CUDA_ARCH__
      __device__
      static T* get( const cuda::managed_array< T >& ) {
         return nullptr;
      }
#else
      __host__
      static T* get( const cuda::managed_array< T >& obj ) {
         return obj.get();
      }
#endif // __CUDA_ARCH__
      /// Swap two objects
#ifdef __CUDA_ARCH__
      __device__
      static void swap( cuda::managed_array< T >&,
                        cuda::managed_array< T >& ) {
         return;
      }
#else
      __host__
      static void swap( cuda::managed_array< T >& obj1,
                        cuda::managed_array< T >& obj2 ) {
         obj1.swap( obj2 );
         return;
      }
#endif // __CUDA_ARCH__
      /// Attach the memory to a stream
#ifdef __CUDA_ARCH__
      __device__
      static void* attach( cuda::managed_array< T >&, cudaStream_t ) {
         return nullptr;
      }
#else
      __host__
      static void* attach( cuda::managed_array< T >& obj,
                           cudaStream_t stream ) {
         if( obj && stream ) {
            CUDA_CHECK( cudaStreamAttachMemAsync( stream, obj.get() ) );
         }
         return obj.get();
      }
#endif // __CUDA_ARCH__
      /// Retrieve the memory from a device
      __host__ __device__
      static void retrieve( cuda::managed_array< T >&, cudaStream_t ) {
         return;
      }
   }; // class ArrayTraits

   template< typename T >
   class ArrayTraits< AuxTypeVectorMemory< T > > {
   public:
      /// Make a new array of a given size
      __host__
      static AuxTypeVectorMemory< T >
      make_array( std::size_t size ) {
         return AuxTypeVectorMemory< T >{ size,
                                          cuda::make_host_array< T >( size ),
                                          cuda::make_device_array< T >( 0 ) };
      }
      /// Get a pointer to the (beginning of the) array
#ifdef __CUDA_ARCH__
      __device__
      static T* get( const AuxTypeVectorMemory< T >& ) {
         return nullptr;
      }
#else
      __host__
      static T* get( const AuxTypeVectorMemory< T >& obj ) {
         return obj.m_host.get();
      }
#endif // __CUDA_ARCH__
      /// Swap two objects
#ifdef __CUDA_ARCH__
      __device__
      static void swap( AuxTypeVectorMemory< T >&,
                        AuxTypeVectorMemory< T >& ) {
         return;
      }
#else
      __host__
      static void swap( AuxTypeVectorMemory< T >& obj1,
                        AuxTypeVectorMemory< T >& obj2 ) {
         std::size_t tempSize = obj1.m_size;
         obj1.m_size = obj2.m_size;
         obj2.m_size = tempSize;
         obj1.m_host.swap( obj2.m_host );
         obj1.m_device.swap( obj2.m_device );
         return;
      }
#endif // __CUDA_ARCH__
      /// Attach the memory to a stream
#ifdef __CUDA_ARCH__
      __device__
      static void* attach( AuxTypeVectorMemory< T >&, cudaStream_t ) {
         return nullptr;
      }
#else
      __host__
      static void* attach( AuxTypeVectorMemory< T >& obj,
                           cudaStream_t stream ) {
         if( stream ) {
            obj.m_device = cuda::make_device_array< T >( obj.m_size );
            CUDA_CHECK( cudaMemcpyAsync( obj.m_device.get(),
                                         obj.m_host.get(),
                                         obj.m_size * sizeof( T ),
                                         cudaMemcpyHostToDevice, stream ) );
            return obj.m_device.get();
         } else {
            return obj.m_host.get();
         }
      }
#endif // __CUDA_ARCH__
      /// Retrieve the memory from a device
#ifdef __CUDA_ARCH__
      __device__
      static void retrieve( AuxTypeVectorMemory< T >&,
                            cudaStream_t ) {
         return;
      }
#else
      __host__
      static void retrieve( AuxTypeVectorMemory< T >& obj,
                            cudaStream_t stream ) {
         if( stream ) {
            CUDA_CHECK( cudaMemcpyAsync( obj.m_host.get(),
                                         obj.m_device.get(),
                                         obj.m_size * sizeof( T ),
                                         cudaMemcpyDeviceToHost, stream ) );
         }
         return;
      }
#endif // __CUDA_ARCH__
   }; // class ArrayTraits

} // private namespace

template< typename T, typename CONT >
__host__
AuxTypeVector< T, CONT >::AuxTypeVector( std::size_t size,
                                         std::size_t capacity )
: m_variable( ::ArrayTraits< CONT >::make_array( capacity ) ),
  m_size( size ), m_capacity( capacity ) {

}

template< typename T, typename CONT >
template< typename DUMMY >
__host__ __device__
AuxTypeVector< T, CONT >::AuxTypeVector( void* addr, std::size_t size,
                                         std::enable_if_t<
                                            std::is_pointer< CONT >::value,
                                                             DUMMY > )
: m_variable( ::ArrayTraits< CONT >::make_array( addr ) ),
  m_size( size ), m_capacity( size ) {

}

template< typename T, typename CONT >
__host__ __device__
void* AuxTypeVector< T, CONT >::toPtr() {

   return ::ArrayTraits< CONT >::get( m_variable );
}

template< typename T, typename CONT >
__host__ __device__
size_t AuxTypeVector< T, CONT >::size() const {

   return m_size;
}

template< typename T, typename CONT >
__host__
bool AuxTypeVector< T, CONT >::resize( size_t sz ) {

   // Check if anything needs to be done.
   if( m_size == sz ) {
      return true;
   }

   // Check if we need to allocate a larger array or not.
   if( sz <= m_capacity ) {
      if( sz > m_size ) {
         bzero( ::ArrayTraits< CONT >::get( m_variable ) + m_size,
                ( sz - m_size ) * sizeof( T ) );
      }
      m_size = sz;
      return true;
   }

   // Decide what the size of the extended array should be.
   std::size_t newSize = m_size;
   do {
      newSize *= 2;
   } while( newSize < sz );

   // Allocate the new array.
   auto newVariable = ::ArrayTraits< CONT >::make_array( newSize );
   // Fill it with the old array's contents.
   auto newPtr = ::ArrayTraits< CONT >::get( newVariable );
   auto oldPtr = ::ArrayTraits< CONT >::get( m_variable );
   if( newPtr && oldPtr ) {
      memcpy( newPtr, oldPtr, m_size * sizeof( T ) );
   }
   bzero( ::ArrayTraits< CONT >::get( newVariable ) + m_size,
          ( sz - m_size ) * sizeof( T ) );
   // And now replace the old array with the new one.
   ::ArrayTraits< CONT >::swap( m_variable, newVariable );
   m_size = sz;
   m_capacity = newSize;

   // Iterators are now invalid.
   return false;
}

template< typename T, typename CONT >
__host__
void AuxTypeVector< T, CONT >::reserve( size_t sz ) {

   // Check if anything needs to be done.
   if( m_capacity >= sz ) {
      return;
   }

   // Allocate the new array.
   auto newVariable = ::ArrayTraits< CONT >::make_array( sz );
   // Fill it with the old array's contents.
   auto newPtr = ::ArrayTraits< CONT >::get( newVariable );
   auto oldPtr = ::ArrayTraits< CONT >::get( m_variable );
   if( newPtr && oldPtr ) {
      memcpy( newPtr, oldPtr, m_size * sizeof( T ) );
   }
   bzero( ::ArrayTraits< CONT >::get( newVariable ) + m_size,
          ( sz - m_size ) * sizeof( T ) );
   // And now replace the old array with the new one.
   ::ArrayTraits< CONT >::swap( m_variable, newVariable );
   m_capacity = sz;
   return;
}

template< typename T, typename CONT >
__host__
void AuxTypeVector< T, CONT >::shift( size_t pos, ptrdiff_t offs ) {

   if( offs < 0 ) {
      // Make sure that the position and offset values make sense.
      if( std::abs( offs ) > static_cast< ptrdiff_t >( pos ) ) {
         offs = -pos;
      }
      // Do the deed.
      memcpy( ::ArrayTraits< CONT >::get( m_variable ) + pos + offs,
              ::ArrayTraits< CONT >::get( m_variable ) + pos,
              std::abs( offs ) * sizeof( T ) );
      // Reduce the size of the vector.
      resize( m_size + offs );
      // Zero out the last bits.
      bzero( ::ArrayTraits< CONT >::get( m_variable ) + m_size + offs,
             std::abs( offs ) * sizeof( T ) );
   } else if( offs > 0 ) {
      // Increase the size of the vector.
      resize( m_size + offs );
      // Do the deed.
      memcpy( ::ArrayTraits< CONT >::get( m_variable ) + pos + offs,
              ::ArrayTraits< CONT >::get( m_variable ) + pos,
              offs * sizeof( T ) );
      bzero( ::ArrayTraits< CONT >::get( m_variable ) + pos,
             offs * sizeof( T ) );
   }

   return;
}

template< typename T, typename CONT >
__host__
bool AuxTypeVector< T, CONT >::insertMove( size_t pos, void* beg,
                                           void* end ) {

   // The size of the inserted range.
   const std::size_t insSize = ( static_cast< T* >( end ) -
                                 static_cast< T* >( beg ) );

   // Make sure that the position makes sense.
   if( pos > m_size ) {
      pos = m_size;
   }

   // Increase the size of the vector.
   const bool result = resize( m_size + insSize );

   // Move the existing data to the right place.
   assert( ( static_cast< int >( m_size ) - static_cast< int >( pos ) -
             static_cast< int >( insSize ) ) >= 0 );
   auto tempArray = ::ArrayTraits< CONT >::make_array( m_size - pos -
                                                       insSize );
   auto tempPtr = ::ArrayTraits< CONT >::get( tempArray );
   if( tempPtr ) {
      memcpy( tempPtr, ::ArrayTraits< CONT >::get( m_variable ) + pos,
              ( m_size - pos - insSize ) * sizeof( T ) );
      memcpy( ::ArrayTraits< CONT >::get( m_variable ) + pos + insSize,
              tempPtr, ( m_size - pos - insSize ) * sizeof( T ) );
   }

   // Insert the received memory content.
   memcpy( ::ArrayTraits< CONT >::get( m_variable ) + pos,
           beg, insSize * sizeof( T ) );

   // Tell the caller whether iterators got invalidated.
   return result;
}

template< typename T, typename CONT >
__host__
void* AuxTypeVector< T, CONT >::attachTo( cudaStream_t stream ) {

   return ::ArrayTraits< CONT >::attach( m_variable, stream );
}

template< typename T, typename CONT >
__host__
void AuxTypeVector< T, CONT >::retrieveFrom( cudaStream_t stream ) {

   ::ArrayTraits< CONT >::retrieve( m_variable, stream );
   return;
}

/// Macro helping to instantiate the class for all POD types.
#define INST_AUXVEC( TYPE )                                          \
   template class AuxTypeVector< TYPE,                               \
                                 AuxTypeVectorMemory< TYPE > >;      \
   template class AuxTypeVector< TYPE,                               \
                                 cuda::managed_array< TYPE > >;      \
   template class AuxTypeVector< TYPE, TYPE* >;                      \
   template AuxTypeVector< TYPE,                                     \
                           TYPE* >::AuxTypeVector( void*,            \
                                                   std::size_t,      \
                                                   char )

// Instantiate the type for all POD types.
INST_AUXVEC( char );
INST_AUXVEC( unsigned char );
INST_AUXVEC( short );
INST_AUXVEC( unsigned short );
INST_AUXVEC( int );
INST_AUXVEC( unsigned int );
INST_AUXVEC( long );
INST_AUXVEC( unsigned long );
INST_AUXVEC( long long );
INST_AUXVEC( unsigned long long );
INST_AUXVEC( float );
INST_AUXVEC( double );

// Clean up.
#undef INST_AUXVEC
