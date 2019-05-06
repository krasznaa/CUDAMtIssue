// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "KernelRunnerSvc.cuh"

// Project include(s).
#include "core/Info.h"
#include "core/Macros.cuh"
#include "core/StreamPool.cuh"

// System include(s).
#include <cassert>
#include <thread>

namespace {

   /// Function called by CUDA when a kernel task is finished
   ///
   /// It then updates the status of the associated @c AthCUDA::KernelStatus
   /// object to report the task as finished.
   ///
   void finishKernel( void* userData ) {

      // Cast the user data to the right type.
      KernelTask* task =
         reinterpret_cast< KernelTask* >( userData );

      // Check if there was an error in the execution.
      if( cudaGetLastError() != cudaSuccess ) {
         task->setCode( StatusCode::FAILURE );
      }

      // Set the task as finished.
      task->setFinished();
      return;
   }

} // private namespace

KernelRunnerSvc::~KernelRunnerSvc() {

   // Stop the service thread.
   m_queue.push( nullptr );
   m_thread.join();
}

KernelRunnerSvc& KernelRunnerSvc::instance() {

   static KernelRunnerSvc svc;
   return svc;
}

std::size_t KernelRunnerSvc::parallelKernels() const {

   return m_parallelKernels;
}

void KernelRunnerSvc::setParallelKernels( std::size_t value ) {

   m_parallelKernels = value;
   return;
}

KernelStatus
KernelRunnerSvc::execute( std::unique_ptr< KernelTask > task ) {

   // Make sure that we received a valid task.
   assert( task.get() != nullptr );

   // Check if a GPU is available, or if there are already "many tasks in
   // flight".
   if( StreamPool::instance().isEmpty() ||
       ( KernelStatus::count() > m_parallelKernels ) ) {

      // If so, let's just execute the task in the current thread.
      task->execute( nullptr );
      // Set the task to be finished.
      task->code().isSuccess();
      task->setFinished();

      // Create the result that does not refer to the task anymore.
      return KernelStatus();
   }

   // If we got here, we need to schedule the task for execution on the/a
   // GPU.

   // Construct the return object already.
   KernelStatus result( std::shared_ptr< KernelTask >( task.get() ) );
   // Add the task to our queue.
   m_queue.push( task.release() );
   // Now return the result/future object.
   return result;
}

KernelRunnerSvc::KernelRunnerSvc()
: m_queue(), m_thread( [ this ]() {

   // Run until we find a null pointer in the queue.
   while( true ) {

      // Try to get a new task from the queue.
      KernelTask* task = nullptr;
      m_queue.pop( task );

      // Check if we should stop...
      if( ! task ) {
         break;
      }

      // Get an available stream for the job.
      auto stream = StreamPool::instance().getAvailableStream();
      assert( stream );

      // First off, let the task schedule all of its own operations.
      task->execute( stream.stream() );

      // Now add a step after those to the stream, one that signals to
      // the user that the task is done.
      CUDA_CHECK( cudaLaunchHostFunc( stream.stream(), ::finishKernel,
                                      task ) );
   }
} ) {

}
