// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "StreamPool.cuh"
#include "Macros.cuh"
#include "Info.h"

// TBB include(s).
#include <tbb/concurrent_queue.h>

struct StreamPoolData {

   /// The concurrent pool of streams that we manage
   tbb::concurrent_queue< cudaStream_t > m_streams;

}; // struct StreamPoolData

StreamPool::~StreamPool() {

   // Delete all the streams.
   cudaStream_t stream = nullptr;
   while( m_data->m_streams.try_pop( stream ) ) {
      CUDA_IGNORE( cudaStreamDestroy( stream ) );
   }

   // Delete the data holder object.
   delete m_data;
}

StreamPool& StreamPool::instance() {

   static StreamPool sp;
   return sp;
}

void StreamPool::setNStreams( std::size_t value ) {

   const std::size_t currentSize = m_data->m_streams.unsafe_size();
   if( currentSize == value ) {
      // Do nothing.
   } else if( value > currentSize ) {
      // Add more streams to get to the desired value.
      for( std::size_t i = 0; i < value - currentSize; ++i ) {
         cudaStream_t stream = nullptr;
         CUDA_CHECK( cudaStreamCreate( &stream ) );
         m_data->m_streams.push( stream );
      }
   } else {
      // Remove streams to get to the desired value.
      cudaStream_t stream = nullptr;
      std::size_t toRemove = currentSize - value;
      while( m_data->m_streams.try_pop( stream ) && ( toRemove-- > 0 ) ) {
         CUDA_CHECK( cudaStreamDestroy( stream ) );
      }
   }

   // Set the "empty flag".
   if( value == 0 ) {
      m_isEmpty = true;
   } else {
      m_isEmpty = false;
   }

   return;
}

std::size_t StreamPool::nStreams() const {

   return m_data->m_streams.unsafe_size();
}

bool StreamPool::isEmpty() const {

   return m_isEmpty;
}

StreamPool::StreamHolder::StreamHolder( cudaStream_t stream,
                                        StreamPool& pool )
: m_stream( stream ), m_pool( &pool ) {

}

StreamPool::StreamHolder::StreamHolder( StreamHolder&& parent )
: m_stream( parent.m_stream ), m_pool( parent.m_pool ) {

   parent.m_stream = nullptr;
}

StreamPool::StreamHolder::~StreamHolder() {

   if( m_stream && m_pool ) {
      m_pool->yieldStream( m_stream );
   }
}

StreamPool::StreamHolder&
StreamPool::StreamHolder::operator=( StreamHolder&& rhs ) {

   if( &rhs == this ) {
      return *this;
   }

   m_stream = rhs.m_stream;
   m_pool = rhs.m_pool;
   rhs.m_stream = nullptr;
   rhs.m_pool = nullptr;

   return *this;
}

StreamPool::StreamHolder::operator bool() const {

   return ( m_stream != nullptr );
}

cudaStream_t StreamPool::StreamHolder::stream() const {

   return m_stream;
}

StreamPool::StreamHolder StreamPool::getAvailableStream() {

   cudaStream_t stream;
   if( m_data->m_streams.try_pop( stream ) ) {
      return StreamHolder( stream, *this );
   } else {
      return StreamHolder( nullptr, *this );
   }
}

StreamPool::StreamPool()
: m_data( new StreamPoolData() ), m_isEmpty( true ) {

   // Allocate 2 streams by default. If a GPU is available.
   if( Info::instance().nDevices() > 0 ) {
      setNStreams( 2 );
   }
}

void StreamPool::yieldStream( cudaStream_t stream ) {

   m_data->m_streams.push( stream );
   return;
}
