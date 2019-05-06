
// Local include(s).
#include "AuxTypeRegistry.h"
#include "AuxTypeVectorFactory.cuh"

AuxTypeRegistry& AuxTypeRegistry::instance() {

   static AuxTypeRegistry reg;
   return reg;
}

std::size_t AuxTypeRegistry::getAuxID( const std::string& name ) {

   // Protect this function.
   lock_t lock( m_mutex );

   // Check if we already assigned an ID to this name.
   auto itr = m_auxids.find( name );
   if( itr != m_auxids.end() ) {
      return itr->second;
   }

   // If not, set it up as a new variable.
   std::size_t auxid = m_types.size();
   m_types.grow_by( 1 );
   m_types[ auxid ] = name;
   m_auxids[ name ] = auxid;

   return auxid;
}

IAuxTypeVector* AuxTypeRegistry::makeVector( std::size_t /*auxid*/,
                                             std::size_t size,
                                             std::size_t capacity ) {

   // Simply just use a float array for all variables.
   static AuxTypeVectorFactory< float > factory;
   return factory.create( size, capacity );
}
