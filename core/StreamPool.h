// Dear emacs, this is -*- c++ -*-
#ifndef CORE_STREAMPOOL_H
#define CORE_STREAMPOOL_H

// Local include(s).
#include "Macros.h"

// System include(s).
#include <atomic>
#include <cstddef>

/// Forward declaration of the internal data object
struct StreamPoolData;

/// Class managing a fixed number of CUDA streams during an application
///
/// The idea here is to allow us to write applications that would only ever
/// push a limited number of kernels to the GPU concurrently. Allowing the
/// calculation to fall back on the CPU when no GPU is available.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class StreamPool {

public:
   /// Destructor
   ~StreamPool();

   /// Get the only instance of the object
   static StreamPool& instance();

   /// Set the total number of streams to use
   ///
   /// This is not a thread-safe function. One should only call it during
   /// initialisation, before any clients would start using the stream pool.
   ///
   /// @param value The number of streams to use
   ///
   void setNStreams( std::size_t value );

   /// Get the number of available streams
   ///
   /// This is not a thread-safe function. It should only be used to check
   /// during the initialisation of an application whether any streams could
   /// be allocated or not.
   ///
   std::size_t nStreams() const;

   /// Check whether any streams are available for the job
   ///
   /// This is a thread-safe way to check whether any CUDA streams are
   /// available at runtime.
   ///
   bool isEmpty() const;

   /// Helper class to return a stream to the caller with
   class StreamHolder {

   public:
      /// Constructor with the stream and the pool
      StreamHolder( cudaStream_t stream, StreamPool& pool );
      /// Move constructor
      StreamHolder( StreamHolder&& parent );
      /// Destructor
      ~StreamHolder();

      /// Move assignment operator
      StreamHolder& operator=( StreamHolder&& rhs );

      /// Convenience operator for checking if the holder has a valid object
      operator bool() const;

      /// Get the managed stream
      cudaStream_t stream() const;

   private:
      /// The managed stream
      cudaStream_t m_stream;
      /// The pool that this holder is attached to
      StreamPool* m_pool;

   }; // class StreamHolder

   /// Make @c StreamHolder a friend of this class
   friend class StreamHolder;

   /// Get a stream from the pool
   ///
   /// Note that the returned holder may point to a null memory address,
   /// in case no streams are available at the moment of the call.
   ///
   StreamHolder getAvailableStream();

private:
   /// Private constructor
   StreamPool();

   /// Re-take posession of a stream that was loned out for a calculation
   void yieldStream( cudaStream_t stream );

   /// The data managed by the stream pool
   StreamPoolData* m_data;
   /// Flag showing whether streams are available
   std::atomic_bool m_isEmpty;

}; // class StreamPool

#endif // CORE_STREAMPOOL_H
