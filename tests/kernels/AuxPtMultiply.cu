// Dear emacs, this is -*- c++ -*-

// Project include(s).
#include "kernel/KernelRun.cuh"

namespace {

   /// Functor performing the pT multiplication
   class AuxPtMultiply {
   public:
      __host__ __device__
      void operator()( std::size_t index, AuxContainer& aux,
                       std::size_t ptId ) {

         aux.array< float >( ptId )[ index ] *= 1.4f;
         return;
      }
   }; // class AuxPtMultiply

} // private namespace

StatusCode auxPtMultiply( AuxContainer& aux ) {

   auto result = Kernel::run< ::AuxPtMultiply >( aux, "pt" );
   return result.wait();
}
