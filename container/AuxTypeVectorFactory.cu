// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "AuxTypeVectorFactory.cuh"
#include "AuxTypeVector.cuh"

template< typename T >
__host__
IAuxTypeVector* AuxTypeVectorFactory< T >::create( std::size_t size,
                                                   std::size_t capacity ) {

   return new AuxTypeVector< T, AuxTypeVectorMemory< T > >( size, capacity );
}

/// Macro helping to instantiate the class for all POD types.
#define INST_FACT( TYPE )                               \
   template class AuxTypeVectorFactory< TYPE >

// Instantiate the type for all POD types.
INST_FACT( char );
INST_FACT( unsigned char );
INST_FACT( short );
INST_FACT( unsigned short );
INST_FACT( int );
INST_FACT( unsigned int );
INST_FACT( long );
INST_FACT( unsigned long );
INST_FACT( long long );
INST_FACT( unsigned long long );
INST_FACT( float );
INST_FACT( double );

// Clean up.
#undef INST_FACT
