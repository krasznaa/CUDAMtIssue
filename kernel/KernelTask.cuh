// Dear emacs, this is -*- c++ -*-
#ifndef KERNEL_KERNELTASK_CUH
#define KERNEL_KERNELTASK_CUH

// Project include(s).
#include "core/Macros.cuh"
#include "core/StatusCode.h"

// CUDA include(s).
#include <cuda.h>

// System include(s).
#include <mutex>

/// Base class for GPU tasks scheduled using @c KernelRunnerSvc
class KernelTask {

public:
   /// Default constructor
   KernelTask();
   /// Disallow copying the task
   KernelTask( const KernelTask& ) = delete;
   /// Disallow moving the task
   KernelTask( KernelTask&& ) = delete;
   /// Virtual destructor
   virtual ~KernelTask() {}

   /// Disallow copy assigning the task
   KernelTask& operator=( const KernelTask& ) = delete;
   /// Disallow move assigning the task
   KernelTask& operator=( KernelTask&& ) = delete;

   /// Function executing the kernel on a specific stream
   ///
   /// If the stream is set to @c nullptr, the function is expected to
   /// execute the task in the calling thread on the CPU.
   ///
   virtual void execute( cudaStream_t stream ) = 0;

   /// The type of mutex used by the task
   typedef std::mutex mutex_t;
   /// The lock to use on the mutex
   typedef std::lock_guard< mutex_t > lock_t;

   /// Get the mutex that describes the execution state of the kernel
   mutex_t& mutex();
   /// Set the task to be finished (release the lock on the mutex)
   void setFinished();

   /// Get the final status code of the task (only valid once the kernel
   /// finished)
   StatusCode code() const;
   /// Set the final status code of the task
   void setCode( StatusCode code );

private:
   /// Mutex locked until the completion of the task
   mutex_t m_mutex;
   /// The final status code of the task
   StatusCode m_code = StatusCode::SUCCESS;

}; // class KernelTask

#endif // KERNEL_KERNELTASK_CUH
