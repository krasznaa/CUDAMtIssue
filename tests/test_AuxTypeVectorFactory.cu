// Dear emacs, this is -*- c++ -*-

// Project include(s).
#include "container/AuxTypeVectorFactory.cuh"
#include "container/IAuxTypeVector.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <memory>

int main() {

   // Smart pointer to receive vectors into.
   typedef std::unique_ptr< IAuxTypeVector > vec_t;

   // Try to instantiate a few simple vectors, and do some very simple tests
   // with them.
   vec_t vec1( AuxTypeVectorFactory< int >().create( 20, 50 ) );
   assert( vec1->size() == 20 );
   vec_t vec2( AuxTypeVectorFactory< float >().create( 30, 50 ) );
   assert( vec2->size() == 30 );

   // Return gracefully.
   return 0;
}
