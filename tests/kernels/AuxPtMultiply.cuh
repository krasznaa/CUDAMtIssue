// Dear emacs, this is -*- c++ -*-
#ifndef TESTS_KERNELS_AUXPTMULTIPLY_CUH
#define TESTS_KERNELS_AUXPTMULTIPLY_CUH

// Project include(s).
#include "core/StatusCode.h"
#include "container/AuxContainer.cuh"

/// Function multiplying the pT values on all elements by 1.4
StatusCode auxPtMultiply( AuxContainer& aux );

#endif // TESTS_KERNELS_AUXPTMULTIPLY_CUH
