// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "Memory.cuh"
#include "Info.h"
#include "Macros.cuh"

// TBB include(s).
#include <tbb/concurrent_queue.h>

// System include(s).
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace cuda {

   namespace details {

      /// Service thread taking care of deleting GPU memory
      ///
      /// It's here to try avoid the locks incurred by cudaFree(...).
      /// Though memory allocation is still done in multiple threads
      /// concurrently, so some conention will still happen.
      ///
      class MemoryDeleterSvc {

      public:
         /// Destructor
         ~MemoryDeleterSvc() {
            // Stop the service thread.
            m_queue.push( DeleteInfo{ FinishThread, nullptr } );
            m_thread.join();
         }

         /// Access the singleton instance
         static MemoryDeleterSvc& instance() {

            static MemoryDeleterSvc svc;
            return svc;
         }

         /// Types of memory to delete
         enum MemType {
            Host   = 0, ///< Memory pinned on the host
            Device = 1, ///< Memory allocated on a/the device
            FinishThread = 999 ///< Exit the service thread
         };

         /// Schedule an auxiliary vector for deletion.
         void scheduleDelete( MemType type, void* mem ) {
            m_queue.push( DeleteInfo{ type, mem } );
            return;
         }

      private:
         /// Constructor launching the thread taking care of memory cleanups
         MemoryDeleterSvc()
         : m_queue(), m_thread( [ this ]() {

            // Run until a @c FinishThread element is received.
            while( true ) {

               // Get a new object from the queue.
               DeleteInfo info;
               m_queue.pop( info );

               // Delete the object in the correct way.
               if( info.type == Host ) {
                  CUDA_CHECK( cudaFreeHost( info.ptr ) );
               } else if( info.type == Device ) {
                  CUDA_CHECK( cudaFree( info.ptr ) );
               } else if( info.type == FinishThread ) {
                  break;
               } else {
                  throw std::runtime_error( "Unknown memory type received" );
               }
            }
         } ) {

         }

         /// Struct used in the internal queue
         struct DeleteInfo {
            MemType type;
            void* ptr;
         };
         /// The queue of memory areas to delete
         tbb::concurrent_bounded_queue< DeleteInfo > m_queue;

         /// The thread taking care of the memory cleanups
         std::thread m_thread;

      }; // class MemoryDeleterSvc

      void ManagedArrayDeleter::operator()( void* ptr ) {

         // Don't do anything for a null pointer.
         if( ptr == nullptr ) {
            return;
         }

         // If a device is available, then free up the memory using CUDA.
         if( Info::instance().nDevices() != 0 ) {
            MemoryDeleterSvc::instance().
               scheduleDelete( MemoryDeleterSvc::Device, ptr );
            return;
         }

         // If not, then the memory was simply allocated with malloc...
         ::free( ptr );
         return;
      }

      void DeviceArrayDeleter::operator()( void* ptr ) {

         // Don't do anything for a null pointer.
         if( ptr == nullptr ) {
            return;
         }

         // If a device is available, then free up the memory using CUDA.
         if( Info::instance().nDevices() != 0 ) {
            MemoryDeleterSvc::instance().
               scheduleDelete( MemoryDeleterSvc::Device, ptr );
            return;
         }

         // If not, then the memory was simply allocated with malloc...
         ::free( ptr );
         return;
      }

      void HostArrayDeleter::operator()( void* ptr ) {

         // Don't do anything for a null pointer.
         if( ptr == nullptr ) {
            return;
         }

         // If a device is available, then free up the memory using CUDA.
         if( Info::instance().nDevices() != 0 ) {
            MemoryDeleterSvc::instance().
               scheduleDelete( MemoryDeleterSvc::Host, ptr );
            return;
         }

         // If not, then the memory was simply allocated with malloc...
         ::free( ptr );
         return;
      }

      /// Use CUDA managed memory if CUDA is available during the build and
      /// a CUDA device is available during runtime. Otherwise do the deed
      /// simply with standard C memory allocation.
      ///
      /// @param size The size of the array to create
      /// @return A pointer to the allocated array
      ///
      template< typename T >
      T* managedMallocHelper( std::size_t size ) {

         // For a zero sized array return a null pointer.
         if( size == 0 ) {
            return nullptr;
         }

         // The result pointer.
         T* result = 0;

         // Try to allocate the array in CUDA managed memory first.
         if( Info::instance().nDevices() != 0 ) {
            CUDA_CHECK( cudaMallocManaged( &result, size * sizeof( T ) ) );
            return result;
         }

         // If that didn't work, fall back on simple malloc.
         result = static_cast< T* >( ::malloc( size * sizeof( T ) ) );
         return result;
      }

      /// Use CUDA device memory if CUDA is available during the build and
      /// a CUDA device is available during runtime. Otherwise do the deed
      /// simply with standard C memory allocation.
      ///
      /// @param size The size of the array to create
      /// @return A pointer to the allocated array
      ///
      template< typename T >
      T* deviceMallocHelper( std::size_t size ) {

         // For a zero sized array return a null pointer.
         if( size == 0 ) {
            return nullptr;
         }

         // The result pointer.
         T* result = 0;

         // Try to allocate the array in CUDA managed memory first.
         if( Info::instance().nDevices() != 0 ) {
            CUDA_CHECK( cudaMalloc( &result, size * sizeof( T ) ) );
            return result;
         }

         // If that didn't work, fall back on simple malloc.
         result = static_cast< T* >( ::malloc( size * sizeof( T ) ) );
         return result;
      }

      /// Use CUDA to allocate page-locked memory on the host if CUDA is
      /// available, otherwise just allocate plain old memory.
      ///
      /// @param size The size of the array to create
      /// @return A pointer to the allocated array
      ///
      template< typename T >
      T* hostMallocHelper( std::size_t size ) {

         // For a zero sized array return a null pointer.
         if( size == 0 ) {
            return nullptr;
         }

         // The result pointer.
         T* result = 0;

         // Try to allocate the array in CUDA managed memory first.
         if( Info::instance().nDevices() != 0 ) {
            CUDA_CHECK( cudaHostAlloc( &result, size * sizeof( T ),
                                       cudaHostAllocDefault |
                                       cudaHostAllocWriteCombined ) );
            return result;
         }

         // If that didn't work, fall back on simple malloc.
         result = static_cast< T* >( ::malloc( size * sizeof( T ) ) );
         return result;
      }

   } // namespace details

} // namespace cuda

/// Helper macro for instantiating the allocator functions for different types
#define INST_MALLOC( TYPE )                                                  \
   template TYPE* cuda::details::managedMallocHelper< TYPE >( std::size_t ); \
   template TYPE* cuda::details::hostMallocHelper< TYPE >( std::size_t );    \
   template TYPE* cuda::details::deviceMallocHelper< TYPE >( std::size_t )

// Instantiate the array allocators for all "reasonable" primitive types.
INST_MALLOC( void* );
INST_MALLOC( char );
INST_MALLOC( unsigned char );
INST_MALLOC( short );
INST_MALLOC( unsigned short );
INST_MALLOC( int );
INST_MALLOC( unsigned int );
INST_MALLOC( long );
INST_MALLOC( unsigned long );
INST_MALLOC( long long );
INST_MALLOC( unsigned long long );
INST_MALLOC( float );
INST_MALLOC( double );

// Clean up.
#undef INST_MALLOC
