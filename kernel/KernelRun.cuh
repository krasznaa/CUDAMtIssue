// Dear emacs, this is -*- c++ -*-
#ifndef KERNEL_KERNELRUN_CUH
#define KERNEL_KERNELRUN_CUH

// Local include(s).
#include "KernelStatus.cuh"

// Project include(s).
#include "container/AuxContainer.cuh"

namespace Kernel {

   /// Function used to execute user defined functors on an auxiliary store
   template< class FUNCTOR, class... VARNAMES >
   KernelStatus run( AuxContainer& aux, VARNAMES... varNames );

   /// Function used to execute user defined functors on an auxiliary store
   template< class FUNCTOR, class USERVAR, class... VARNAMES >
   KernelStatus runWithArg( AuxContainer& aux, USERVAR userVariable,
                            VARNAMES... varNames );

} // namespace Kernel

// Include the template implementation.
#include "KernelRun.icc"

#endif // KERNEL_KERNELRUN_CUH
