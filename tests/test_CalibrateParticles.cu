// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "kernels/CalibrateParticles.cuh"

// Project include(s).
#include "core/Info.h"
#include "core/StreamPool.cuh"
#include "container/AuxContainer.cuh"
#include "container/AuxTypeRegistry.h"
#include "kernel/KernelRunnerSvc.cuh"

// TBB include(s).
#include <tbb/tbb.h>

// Boost include(s).
#include <boost/program_options.hpp>

// System include(s).
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

/// Functor performing a "calibration" on a particle container
class ParticleCalibratorTbbFtr {

public:
   /// Constructor with all needed parameters
   ///
   /// @param maxParticles The maximum number of particles to generate
   /// @param calcIterations The calculation iterations in the "calibration"
   ///
   ParticleCalibratorTbbFtr( std::size_t maxParticles,
                             std::size_t calcIterations )
      : m_maxParticles( maxParticles ),
        m_calcIterations( calcIterations ) {

   }

   /// Perform a "calibration" on a random number of particles
   std::size_t operator()( std::size_t randomSeed ) const {

      // Set up an "electron array".
      AuxContainer aux;

      // Variable IDs.
      AuxTypeRegistry& reg = AuxTypeRegistry::instance();
      static const std::size_t ptId  = reg.getAuxID( "pt" );
      static const std::size_t etaId = reg.getAuxID( "eta" );
      static const std::size_t phiId = reg.getAuxID( "phi" );

      // The random number generators.
      std::mt19937 gen( randomSeed );
      std::uniform_int_distribution< std::size_t >
         nPartGen( 0, m_maxParticles );
      std::normal_distribution< float > ptGen( 20000.0, 5000.0 );
      std::normal_distribution< float > etaGen( 0.0, 1.5 );
      std::uniform_real_distribution< float > phiGen( -M_PI, M_PI );

      // Generate some random "particles".
      const std::size_t nParticles = nPartGen( gen );
      aux.resize( nParticles );
      float* ptArray =
         reinterpret_cast< float* >( aux.getData( ptId, nParticles,
                                                  nParticles ) );
      float* etaArray =
         reinterpret_cast< float* >( aux.getData( etaId, nParticles,
                                                  nParticles ) );
      float* phiArray =
         reinterpret_cast< float* >( aux.getData( phiId, nParticles,
                                                  nParticles ) );
      for( std::size_t i = 0; i < nParticles; ++i ) {
         ptArray[ i ]  = ptGen( gen );
         etaArray[ i ] = etaGen( gen );
         phiArray[ i ] = phiGen( gen );
      }

      // Run the calculaion.
      if( calibrateParticles( m_calcIterations, aux ).isFailure() ) {
         throw std::runtime_error( "Failed to execute electron calibration" );
      }

      // Return the random seed.
      return randomSeed;
   }

private:
   /// The maximum number of particles to generate
   std::size_t m_maxParticles;
   /// The number of calculation iterations to run on the particles
   std::size_t m_calcIterations;

}; // class ParticleCalibratorTbbFtr

int main( int argc, char* argv[] ) {

   // Set up the command line arguments.
   namespace po = boost::program_options;
   po::options_description desc( "TBB + CUDA Test Application" );
   desc.add_options()
      ( "help,h", "Give some help about the command line arguments" )
      ( "max-particles,p", po::value< std::size_t >()->default_value( 50 ),
        "Maximum number of particles to generate" )
      ( "events,v", po::value< std::size_t >()->default_value( 10000 ),
        "Number of events to process" )
      ( "calc-iterations,i", po::value< std::size_t >()->default_value( 10 ),
        "Number of calculation iterations to execute" )
      ( "cpu-threads,t", po::value< std::size_t >()->default_value( 4 ),
        "Number of CPU threads to use with TBB" )
      ( "gpu-tasks,g", po::value< std::size_t >()->default_value( 2 ),
        "The number of GPU calculations to run in parallel" )
      ( "gpu-streams,s", po::value< std::size_t >()->default_value( 2 ),
        "Number of GPU streams to use" );

   // Now read/interpret them.
   po::variables_map vm;
   try {
      po::store( po::parse_command_line( argc, argv, desc ), vm );
      po::notify( vm );
   } catch( const std::exception& ex ) {
      std::cerr << "Command line parsing error: " << ex.what() << std::endl;
      std::cerr << desc << std::endl;
      return 1;
   }

   // If the user asked for help...
   if( vm.count( "help" ) ) {
      std::cout << desc << std::endl;
      return 0;
   }

   // Extract the configuration parameters.
   const std::size_t maxParticles = vm[ "max-particles" ].as< std::size_t >();
   const std::size_t calcIterations =
      vm[ "calc-iterations" ].as< std::size_t >();
   const std::size_t events = vm[ "events" ].as< std::size_t >();
   const std::size_t threads = vm[ "cpu-threads" ].as< std::size_t >();
   const std::size_t gpuTasks = vm[ "gpu-tasks" ].as< std::size_t >();
   const std::size_t streams = vm[ "gpu-streams" ].as< std::size_t >();

   // Print the configuration parameters.
   std::cout << "Using:" << std::endl;
   std::cout << "  - Max. Particles : " << maxParticles << std::endl;
   std::cout << "  - Calc-iterations: " << calcIterations << std::endl;
   std::cout << "  - Events         : " << events << std::endl;
   std::cout << "  - CPU Threads    : " << threads << std::endl;
   std::cout << "  - GPU Tasks      : " << gpuTasks << std::endl;
   std::cout << "  - GPU Streams    : " << streams << std::endl;

   // Set up the GPU usage details.
   if( Info::instance().nDevices() > 0 ) {
      KernelRunnerSvc::instance().setParallelKernels( gpuTasks );
      StreamPool::instance().setNStreams( streams );
   }

   // Set the number of CPU threads to use.
   tbb::task_scheduler_init threadSetter( threads );

   // Create the execution graph.
   tbb::flow::graph graph;

   // Set up the node launching the processing of an event.
   tbb::flow::broadcast_node< std::size_t > launcher( graph );

   // Set up the node that "calibrates" particles.
   tbb::flow::function_node< std::size_t, std::size_t >
      calibrator( graph, tbb::flow::unlimited,
                  ParticleCalibratorTbbFtr( maxParticles,
                                            calcIterations ) );
   tbb::flow::make_edge( launcher, calibrator );

   // Add a node for printing the processing progress.
   tbb::flow::function_node< std::size_t, std::size_t >
      eventPrinter( graph, tbb::flow::unlimited,
                    [ events ]( std::size_t event ) {
                       if( ! ( event % 1000 ) ) {
                          std::cout << "Processed " << event << " / " << events
                                    << " events" << std::endl;
                       }
                       return event;
                    } );
   tbb::flow::make_edge( calibrator, eventPrinter );

   // Perform the calculation.
   for( std::size_t i = 0; i < events; ++i ) {
      launcher.try_put( i );
   }
   graph.wait_for_all();

   // Return gracefully.
   return 0;
}
