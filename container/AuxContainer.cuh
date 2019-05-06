// Dear emacs, this is -*- c++ -*-
#ifndef CONTAINER_AUXCONTAINER_CUH
#define CONTAINER_AUXCONTAINER_CUH

// Local include(s).
#include "IAuxTypeVector.cuh"

// Project include(s).
#include "core/Memory.cuh"

// CUDA include(s).
#include <cuda.h>

// System include(s).
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

/// Container helping with exposing xAOD data to CUDA kernels
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class AuxContainer {

public:
   /// Default constructor
   __host__
   AuxContainer();
   /// Constructor from existing data
   __host__ __device__
   AuxContainer( std::size_t size, std::size_t nVars, void** vars );
   /// Destructor
   __host__ __device__
   ~AuxContainer();

   /// @name Interface for using the data on a (GPU) device
   /// @{

   /// Get the size of the variable arrays
   __host__ __device__
   std::size_t arraySize() const;

   /// Function for accessing a variable array (non-const)
   template< typename T >
   __host__ __device__
   T* array( std::size_t auxid );

   /// Function for accessing a variable array (const)
   template< typename T >
   __host__ __device__
   const T* array( std::size_t auxid ) const;

   /// Get the array of variables to be used on a CUDA device
   __host__
   std::pair< std::size_t, void** > variables( cudaStream_t stream );

   /// Retrieve the variables from a CUDA device
   __host__
   void retrieveFrom( cudaStream_t stream );

   /// @}

   /// @name Implementation of @c SG::IConstAuxStore
   /// @{

   /// Get a pointer to one auxiliary variable
   __host__
   const void* getData( std::size_t auxid ) const;

   /// @}

   /// @name Implementation of @c SG::IAuxStore
   /// @{

   /// Get a (possibly new) pointer to one auxiliary variable
   __host__
   void* getData( std::size_t auxid, std::size_t size, std::size_t capacity );

   /// Change the size of the container
   bool resize( std::size_t size );

   /// Change the capacity of the auxiliary vectors
   void reserve( std::size_t capacity );

   /// Shift elements of the container
   void shift( std::size_t pos, std::ptrdiff_t offs );

   /// @}

private:
   /// Try to reserve a new variable in CUDA Unified memory
   void* getDataCUDA( std::size_t auxid, std::size_t size,
                      std::size_t capacity );

   /// @name Variables used in both the host and device code
   /// @{

   /// The size of the container
   std::size_t m_size;

   /// Array of helper objects managing the variables in CUDA memory
   cuda::array< IAuxTypeVector* > m_vecs;

   /// @}

   /// @name Variables used only in the host code
   ///
   /// I use bare pointers for everything to avoid having to hide the
   /// variables during device code compilation. (They can't be created
   /// in device code...) Not pretty, but it's good enough for now.
   ///
   /// @{

   /// Buffer in CUDA device memory holding the allocated variables
   cuda::device_array< void* >* m_buffer;

   /// Mutex used to synchronise the modifications to the cache vector
   typedef std::mutex mutex_t;
   typedef std::lock_guard< mutex_t > lock_t;
   mutable mutex_t m_mutex;

   /// @}

}; // class AuxContainer

// Include the template implementation.
#include "AuxContainer.icc"

#endif // CONTAINER_AUXCONTAINER_CUH
