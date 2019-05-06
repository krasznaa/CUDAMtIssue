// Dear emacs, this is -*- c++ -*-
#ifndef CONTAINER_IAUXTYPEVECTORFACTORY_CUH
#define CONTAINER_IAUXTYPEVECTORFACTORY_CUH

// Forward declaration(s).
class IAuxTypeVector;

/// A much simplified version of @c SG::IAuxTypeVectorFactory
///
/// Just like @c SG::IAuxTypeVectorFactory, this interface is used to
/// conveniently create @c AthCUDA::IAuxTypeVector type objects at runtime.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class IAuxTypeVectorFactory {

public:
   /// Virtual destructor, to make vtable happy...
   __host__
   virtual ~IAuxTypeVectorFactory() {}

   /// Function creating a new vector (handling) object
   __host__
   virtual IAuxTypeVector* create( std::size_t size,
                                   std::size_t capacity ) = 0;

}; // class IAuxTypeVectorFactory

#endif // CONTAINER_IAUXTYPEVECTORFACTORY_CUH
