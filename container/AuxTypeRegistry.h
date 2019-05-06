// Dear emacs, this is -*- c++ -*-
#ifndef CONTAINER_AUXTYPEREGISTRY_H
#define CONTAINER_AUXTYPEREGISTRY_H

// TBB include(s).
#include <tbb/concurrent_vector.h>

// System include(s).
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>

// Forward declaration(s).
class IAuxTypeVector;

/// Class used for managing CUDA auxiliary vector objects
///
/// Just like @c AuxTypeVector, this class is modeled after a
/// class (@c SG::AuxTypeRegistry) sitting in the core ATLAS EDM code.
/// It takes care of managing @c IAuxTypeVector type objects
/// in type independent code.
///
/// Note that this code was simplified to *only* deal with float arrays.
/// To simplify the example code just a little bit.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class AuxTypeRegistry {

public:
   /// Function implementing the singleton pattern for the class
   static AuxTypeRegistry& instance();

   /// Get the auxiliary ID for a named variable
   std::size_t getAuxID( const std::string& name );

   /// Construct a new vector to hold an auxiliary variable
   IAuxTypeVector* makeVector( std::size_t auxid,
                               std::size_t size,
                               std::size_t capacity );

private:
   /// Private constructor, to implement the singleton pattern
   AuxTypeRegistry() {}

   /// Mutex type used by the class
   typedef std::mutex mutex_t;
   /// Lock type used by the class
   typedef std::lock_guard< mutex_t > lock_t;

   /// Mutex used for global operations
   mutable mutex_t m_mutex;

   /// Map from name -> auxid
   std::unordered_map< std::string, std::size_t > m_auxids;
   /// Table of aux data items, indexed by auxid
   tbb::concurrent_vector< std::string > m_types;

}; // class AuxTypeRegistry

#endif // CONTAINER_AUXTYPEREGISTRY_H
