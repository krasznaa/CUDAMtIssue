// Dear emacs, this is -*- c++ -*-
#ifndef CONTAINER_AUXTYPEVECTORFACTORY_CUH
#define CONTAINER_AUXTYPEVECTORFACTORY_CUH

// Local include(s).
#include "IAuxTypeVectorFactory.cuh"

/// Implementation for @c IAuxTypeVectorFactory
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
template< typename T >
class AuxTypeVectorFactory : public IAuxTypeVectorFactory {

public:
   /// @name Interface inherited from @c IAuxTypeVectorFactory
   /// @{

   /// Function creating a new vector (handling) object
   __host__
   virtual IAuxTypeVector* create( std::size_t size,
                                   std::size_t capacity ) override;

   /// @}

}; // class AuxTypeVectorFactory

#endif // CONTAINER_AUXTYPEVECTORFACTORY_CUH
