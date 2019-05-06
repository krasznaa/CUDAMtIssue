// Dear emacs, this is -*- c++ -*-
#ifndef CONTAINER_IAUXTYPEVECTOR_H
#define CONTAINER_IAUXTYPEVECTOR_H

// System include(s).
#include <cstddef>

/// Interface to the array handling objects.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class IAuxTypeVector {

public:
   /// Destructor.
   __host__ __device__
#ifndef __CUDA_ARCH__
   virtual
#endif // __CUDA_ARCH__
   ~IAuxTypeVector() {}

   /// Return a pointer to the start of the vector's data.
   __host__ __device__
   virtual void* toPtr() = 0;

   /// Return the size of the vector.
   __host__ __device__
   virtual size_t size() const = 0;

   /// Change the size of the vector.
   ///
   /// @param sz The new vector size.
   /// @return @c true if it is known that iterators have not been
   ///         invalidated, @c false otherwise
   ///
   __host__
   virtual bool resize( size_t sz ) = 0;

   /// Change the allocated size of the vector.
   ///
   /// @param sz The new vector allocated size.
   /// @return @c true if it is known that iterators have not been
   ///         invalidated, @c false otherwise
   ///
   __host__
   virtual void reserve( size_t sz ) = 0;

   /// Shift the elements of the vector.
   ///
   /// @param pos The starting index for the shift.
   /// @param offs The (signed) amount of the shift.
   ///
   __host__
   virtual void shift( size_t pos, ptrdiff_t offs ) = 0;

   /// Insert elements into the vector via move semantics.
   ///
   /// @param pos The starting index of the insertion.
   /// @param beg Start of the range of elements to insert.
   /// @param end End of the range of elements to insert.
   /// @return @c true if it is known that the vector's memory did not move,
   ///         @c false otherwise.
   ///
   __host__
   virtual bool insertMove( size_t pos, void* beg, void* end ) = 0;

   /// Attach the memory managed by the vector to a given CUDA stream
   ///
   /// @param stream The CUDA stream to attach the managed memory to
   /// @return The memory address to be used for the variable in device code
   ///
   __host__
   virtual void* attachTo( cudaStream_t stream ) = 0;

   /// Retrieve the memory from the device using a CUDA stream
   ///
   /// @param stream The CUDA stream to retrieve the memory with
   ///
   __host__
   virtual void retrieveFrom( cudaStream_t stream ) = 0;

}; // class IAuxTypeVector

#endif // CONTAINER_IAUXTYPEVECTOR_H
