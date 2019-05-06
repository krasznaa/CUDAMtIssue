// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "kernels/AuxPtMultiply.cuh"

// Project include(s).
#include "core/StreamPool.cuh"
#include "container/AuxContainer.cuh"
#include "container/AuxTypeRegistry.h"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>

int main() {

   // 4-momentum variable IDs.
   AuxTypeRegistry& reg = AuxTypeRegistry::instance();
   const std::size_t ptId  = reg.getAuxID( "pt" );
   const std::size_t etaId = reg.getAuxID( "eta" );
   const std::size_t phiId = reg.getAuxID( "phi" );

   // Set up a "particle array".
   static const std::size_t NPARTICLES = 10;
   AuxContainer aux;
   aux.resize( NPARTICLES );
   float* ptArray  = reinterpret_cast< float* >( aux.getData( ptId, NPARTICLES,
                                                              NPARTICLES ) );
   float* etaArray = reinterpret_cast< float* >( aux.getData( etaId, NPARTICLES,
                                                              NPARTICLES ) );
   float* phiArray = reinterpret_cast< float* >( aux.getData( phiId, NPARTICLES,
                                                              NPARTICLES ) );
   for( std::size_t i = 0; i < NPARTICLES; ++i ) {
      ptArray[ i ]  = 10000.0f * i + 5000.0f;
      etaArray[ i ] = 0.1f * i;
      phiArray[ i ] = 0.2f * i;
   }

   // Run the transformation.
   SC_CHECK( auxPtMultiply( aux ) );

   // Verify the output.
   for( std::size_t i = 0; i < NPARTICLES; ++i ) {
      assert( std::abs( ptArray[ i ] -
                        ( 10000.0f * i + 5000.0f ) * 1.4f ) < 0.001 );
      assert( std::abs( etaArray[ i ] - 0.1f * i ) < 0.001 );
      assert( std::abs( phiArray[ i ] - 0.2f * i ) < 0.001 );
   }

   // Return gracefully.
   return 0;
}
