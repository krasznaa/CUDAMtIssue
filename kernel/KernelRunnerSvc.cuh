// Dear emacs, this is -*- c++ -*-
#ifndef KERNEL_KERNELRUNNERSVC_CUH
#define KERNEL_KERNELRUNNERSVC_CUH

// Local include(s).
#include "KernelStatus.cuh"
#include "KernelTask.cuh"

// TBB include(s).
#include <tbb/concurrent_queue.h>

// System include(s).
#include <atomic>
#include <cstddef>
#include <memory>
#include <thread>

/// Service used for executing @c KernelTask tasks
///
/// This service can be used to run calculations in a balanced way
/// between the CPU and the GPU.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class KernelRunnerSvc {
   
public:
   /// Destructor
   ~KernelRunnerSvc();

   /// Get the only instance of the object
   static KernelRunnerSvc& instance();

   /// The number of kernels to execute in parallel
   std::size_t parallelKernels() const;
   /// Set the number of kernels to execute in parallel
   void setParallelKernels( std::size_t value );

   /// Execute a user specified kernel task
   ///
   /// If a GPU is available at runtime, and it is not doing other things
   /// at the moment, this function offloads the calculation to the GPU,
   /// and returns right away. Expecting the user to wait for the execution
   /// of the task using the returned @c AthCUDA::KernelStatus object.
   ///
   /// If a GPU is not available for any reason, the function just executes
   /// the task on the CPU in the caller thread, and returns only once the
   /// task is finished. (The returned object in this case reports the task
   /// to be finished, right away.)
   ///
   /// @param task The task to be executed on the CPU or GPU
   /// @return A smart object describing the success/failure to run the task
   ///
   KernelStatus execute( std::unique_ptr< KernelTask > task );

private:
   /// Private constructor
   KernelRunnerSvc();

   /// The number of kernels to execute in parallel
   std::atomic< std::size_t > m_parallelKernels;

   /// Queue of tasks to launch on the GPU
   tbb::concurrent_bounded_queue< KernelTask* > m_queue;

   /// Thread responsible for launching GPU kernels
   std::thread m_thread;

}; // class KernelRunnerSvc

#endif // KERNEL_KERNELRUNNERSVC_CUH
