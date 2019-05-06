// Dear emacs, this is -*- c++ -*-

// Project include(s).
#include "container/AuxContainer.cuh"
#include "container/AuxTypeRegistry.h"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <iostream>

int main() {

   // Instantiate a container.
   AuxContainer aux;

   // Some helper variables for the test.
   AuxTypeRegistry& reg = AuxTypeRegistry::instance();
   const std::size_t auxid1 = reg.getAuxID( "var1" );
   const std::size_t auxid2 = reg.getAuxID( "var2" );

   // Fill the container with some dummy information.
   static const std::size_t AUXSIZE = 10;
   aux.resize( AUXSIZE );
   float* var1 = reinterpret_cast< float* >( aux.getData( auxid1, AUXSIZE,
                                                          AUXSIZE ) );
   float* var2 = reinterpret_cast< float* >( aux.getData( auxid2, AUXSIZE,
                                                          AUXSIZE ) );
   for( std::size_t i = 0; i < AUXSIZE; ++i ) {
      var1[ i ] = 1.5f * i;
      var2[ i ] = 3.0f * i;
   }

   // Instantiate another auxiliary container from the payload of the first
   // one. Just like we would do in device/GPU code.
   auto variables = aux.variables( nullptr );
   AuxContainer auxCopy( aux.arraySize(), variables.first, variables.second );

   // Check that the payload in the copied container.
   assert( auxCopy.arraySize() == aux.arraySize() );
   for( std::size_t i = 0; i < AUXSIZE; ++i ) {
      assert( std::abs( aux.array< float >( auxid1 )[ i ] -
                        auxCopy.array< float >( auxid1 )[ i ] ) < 0.0001 );
      assert( std::abs( aux.array< float >( auxid2 )[ i ] -
                        auxCopy.array< float >( auxid2 )[ i ] ) < 0.0001 );
   }

   // Return gracefully.
   return 0;
}
