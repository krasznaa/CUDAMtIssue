// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "CalibrateParticles.cuh"

// Project include(s).
#include "kernel/KernelRun.cuh"

// System include(s).
#include <cmath>

namespace {
   /// Functor "calibrating" Particles (possibly) on a GPU
   class ParticleCalibrator {
   public:
      __host__ __device__
      void operator()( std::size_t index, AuxContainer& aux,
                       std::size_t etaId, std::size_t phiId,
                       std::size_t ptId, std::size_t iterations ) {

         // Make a local copy of the values of interest.
         const double eta = aux.array< float >( etaId )[ index ];
         const double phi = aux.array< float >( phiId )[ index ];
         double pt = aux.array< float >( ptId )[ index ];

         /// Perform the specified number of calculation iterations.
         for( std::size_t i = 0; i < iterations; ++i ) {
            if( eta < 0.0 ) {
               pt += 1.05 * std::cos( std::pow( std::abs( std::sin( phi +
                                                                    pt ) ),
                                                eta ) );
            } else {
               pt += 0.95 * std::sin( std::pow( std::abs( std::cos( phi +
                                                                    pt ) ),
                                                eta ) );
            }
         }

         // Update the "pt" auxiliary variable.
         aux.array< float >( ptId )[ index ] = pt;
         return;
      }
   }; // class ParticleCalibrator
} // private namespace

StatusCode calibrateParticles( std::size_t iterations,
                               AuxContainer& particles ) {

   auto result = Kernel::runWithArg< ::ParticleCalibrator >(
                    particles, iterations, "eta", "phi", "pt" );
   return result.wait();
}
