// Dear emacs, this is -*- c++ -*-
#ifndef TESTS_KERNEL_CALIBRATEPARTICLES_CUH
#define TESTS_KERNEL_CALIBRATEPARTICLES_CUH

// Project include(s).
#include "core/StatusCode.h"
#include "container/AuxContainer.cuh"

/// Function "calibrating" particles (possibly) on a GPU
StatusCode calibrateParticles( std::size_t iterations,
                               AuxContainer& particles );

#endif // TESTS_KERNEL_CALIBRATEPARTICLES_CUH
