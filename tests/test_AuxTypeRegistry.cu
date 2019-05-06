// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "container/AuxTypeRegistry.h"
#include "container/IAuxTypeVector.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <memory>

int main() {

   // Access the singleton.
   AuxTypeRegistry& reg = AuxTypeRegistry::instance();

   // Test that vectors for primitive types can be made.
   const std::size_t auxid1 = reg.getAuxID( "var1" );
   std::unique_ptr< IAuxTypeVector > vec1(
      reg.makeVector( auxid1, 10, 20 ) );
   assert( vec1.get() != nullptr );
   assert( vec1->size() == 10 );

   const std::size_t auxid2 = reg.getAuxID( "var2" );
   std::unique_ptr< IAuxTypeVector > vec2(
      reg.makeVector( auxid2, 15, 20 ) );
   assert( vec2.get() != nullptr );
   assert( vec2->size() == 15 );

   assert( auxid1 != auxid2 );

   // Return gracefully.
   return 0;
}
