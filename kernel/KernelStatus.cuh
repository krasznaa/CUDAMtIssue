// Dear emacs, this is -*- c++ -*-
#ifndef KERNEL_KERNELSTATUS_CUH
#define KERNEL_KERNELSTATUS_CUH

// Local include(s).
#include "KernelTask.cuh"

// Project include(s).
#include "core/StatusCode.h"

// System include(s).
#include <memory>

/// Helper object used for synchronising GPU kernel tasks
///
/// It is meant to be used as a mixture of @c std::future and
/// @c StatusCode.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class KernelStatus {

public:
   /// Default constructor
   KernelStatus();
   /// Constructor with the task whose result this object describes
   KernelStatus( std::shared_ptr< KernelTask > task );
   /// Copy constructor
   KernelStatus( const KernelStatus& parent );
   /// Move constructor
   KernelStatus( KernelStatus&& parent );
   /// Destructor
   ~KernelStatus();

   /// Assignment operator
   KernelStatus& operator=( const KernelStatus& rhs );
   /// Move assignment operator
   KernelStatus& operator=( KernelStatus&& rhs );

   /// Wait for the execution of the kernel to finish
   StatusCode wait();

   /// The total number of kernel status objects in flight at the moment
   static std::size_t count();

private:
   /// Pointer to the task whose result this object describes
   std::shared_ptr< KernelTask > m_task;

}; // class KernelStatus

#endif // KERNEL_KERNELSTATUS_CUH