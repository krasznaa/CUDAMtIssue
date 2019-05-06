// Dear emacs, this is -*- c++ -*-
#ifndef CONTAINER_AUXTYPEVECTOR_CUH
#define CONTAINER_AUXTYPEVECTOR_CUH

// Local include(s).
#include "IAuxTypeVector.cuh"

// Project include(s).
#include "core/Memory.cuh"

// System include(s).
#include <type_traits>
#include <vector>

/// Helper type used with "explicit memory management"
template< typename T >
struct AuxTypeVectorMemory {
   std::size_t m_size; ///< Size of the managed array
   cuda::host_array< T > m_host; ///< The managed array on the host
   cuda::device_array< T > m_device; ///< The managed array on the device
};

/// Class managing one auxiliary variable in CUDA memory
///
/// This class should do much of the work in @c AuxContainer.
/// Managing auxiliary variables in CUDA memory, and allowing
/// @c AuxContainer to expose those variables to a GPU kernel
/// if/when needed.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
template< typename T, typename CONT =
#ifdef __CUDA_ARCH__
          T*
#else
          AuxTypeVectorMemory< T >
#endif // __CUDA_ARCH__
         >
class AuxTypeVector : public IAuxTypeVector {

public:
   /// Make sure that we're only trying to use this class on simple types
   static_assert( std::is_pod< T >::value == true,
                  "AuxTypeVector is only available for POD types" );

   /// Constructor with the size to use for a new variable
   __host__
   AuxTypeVector( std::size_t size, std::size_t capacity );
   /// Constructor pointing at an allocated memory address
   template< typename DUMMY = char >
   __host__ __device__
   AuxTypeVector( void* addr, std::size_t size,
                  std::enable_if_t< std::is_pointer< CONT >::value,
                                    DUMMY > = 0 );

   /// @name Interface inherited from @c IAuxTypeVector
   /// @{

   /// Return a pointer to the start of the vector's data.
   __host__ __device__
   virtual void* toPtr() override;

   /// Return the size of the vector.
   __host__ __device__
   virtual size_t size() const override;

   /// Change the size of the vector.
   ///
   /// @param sz The new vector size.
   /// @return @c true if it is known that iterators have not been
   ///         invalidated, @c false otherwise
   ///
   __host__
   virtual bool resize( size_t sz ) override;

   /// Change the allocated size of the vector.
   ///
   /// @param sz The new vector allocated size.
   /// @return @c true if it is known that iterators have not been
   ///         invalidated, @c false otherwise
   ///
   __host__
   virtual void reserve( size_t sz ) override;

   /// Shift the elements of the vector.
   ///
   /// @param pos The starting index for the shift.
   /// @param offs The (signed) amount of the shift.
   ///
   __host__
   virtual void shift( size_t pos, ptrdiff_t offs ) override;

   /// Insert elements into the vector via move semantics.
   ///
   /// @param pos The starting index of the insertion.
   /// @param beg Start of the range of elements to insert.
   /// @param end End of the range of elements to insert.
   /// @return @c true if it is known that the vector's memory did not move,
   ///         @c false otherwise.
   ///
   __host__
   virtual bool insertMove( size_t pos, void* beg, void* end ) override;

   /// Attach the memory managed by the vector to a given CUDA stream
   ///
   /// @param stream The CUDA stream to attach the managed memory to
   /// @return The memory address to be used for the variable in device code
   ///
   __host__
   virtual void* attachTo( cudaStream_t stream ) override;

   /// Retrieve the memory from the device using a CUDA stream
   ///
   /// @param stream The CUDA stream to retrieve the memory with
   ///
   __host__
   virtual void retrieveFrom( cudaStream_t stream ) override;

   /// @}

private:
   /// Pointer to the managed variable
   CONT m_variable;
   /// The current size of the managed variable array
   std::size_t m_size;
   /// The current capacity of the allocated array
   std::size_t m_capacity;

}; // class AuxTypeVector

#endif // CONTAINER_AUXTYPEVECTOR_CUH
