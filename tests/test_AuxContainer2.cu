// Dear emacs, this is -*- c++ -*-

// Project include(s).
#include "container/AuxContainer.cuh"
#include "container/AuxTypeRegistry.h"
#include "core/StreamPool.cuh"
#include "core/Macros.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>

/// Kernel incrementing the values in the "var1" auxiliary variable
__global__
void incrementVar1( std::size_t csize, std::size_t vsize, void** vars,
                    std::size_t var1Id ) {

   // Find the current index that we need to process.
   const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   if( i >= csize ) {
      return;
   }

   // Construct the auxiliary container.
   AuxContainer aux( csize, vsize, vars );

   // Perform the variable increment.
   aux.array< float >( var1Id )[ i ] += 0.5f;
   return;
}

int main() {

   // Some helper variables for the test.
   AuxTypeRegistry& reg = AuxTypeRegistry::instance();
   const std::size_t auxid1 = reg.getAuxID( "var1" );
   const std::size_t auxid2 = reg.getAuxID( "var2" );

   // Set up a container.
   static const std::size_t AUXSIZE = 10;
   AuxContainer aux;
   aux.resize( AUXSIZE );
   float* var1 = reinterpret_cast< float* >( aux.getData( auxid1, AUXSIZE,
                                                          AUXSIZE ) );
   float* var2 = reinterpret_cast< float* >( aux.getData( auxid2, AUXSIZE,
                                                          AUXSIZE ) );
   for( std::size_t i = 0; i < AUXSIZE; ++i ) {
      var1[ i ] = 1.5f * i;
      var2[ i ] = 3.0f * i;
   }

   // Run a simple kernel on this container.
   auto stream = StreamPool::instance().getAvailableStream();
   auto variables = aux.variables( stream.stream() );
   incrementVar1<<< 1, AUXSIZE, 0, stream.stream() >>>(
      aux.arraySize(), variables.first, variables.second, auxid1 );
   aux.retrieveFrom( stream.stream() );
   CUDA_CHECK( cudaGetLastError() );
   CUDA_CHECK( cudaStreamSynchronize( stream.stream() ) );

   // Check the updated arrays:
   for( std::size_t i = 0; i < AUXSIZE; ++i ) {
      assert( std::abs( var1[ i ] - ( 1.5f * i + 0.5f ) ) < 0.0001 );
      assert( std::abs( var2[ i ] - 3.0f * i ) < 0.0001 );
   }

   // Return gracefully.
   return 0;
}
