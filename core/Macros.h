// Dear emacs, this is -*- c++ -*-
#ifndef CORE_MACROS_H
#define CORE_MACROS_H

// Local include(s).
#include "StatusCode.h"

// System include(s).
#include <iostream>
#include <stdexcept>
#include <string>

// CUDA include(s).
#include <cuda.h>

/// Helper macro for checking the return types of CUDA function calls
#define CUDA_CHECK( EXP )                                            \
   do {                                                              \
      const cudaError_t ce = EXP;                                    \
      if( ce != cudaSuccess ) {                                      \
         throw std::runtime_error( std::string( __FILE__ ) + ":" +   \
                                   std::to_string( __LINE__ ) +      \
                                   " Failed to execute: " #EXP " ("  \
                                   + cudaGetErrorString( ce ) +      \
                                   ")" );                            \
      }                                                              \
   } while( false )

/// Helper macro for running a CUDA function, but not caring about its return
#define CUDA_IGNORE( EXP )                                           \
   do {                                                              \
      EXP;                                                           \
   } while( false )

/// Helper macro for checking the return types of CUDA function calls
#define CUDA_SC_CHECK( EXP )                                         \
   do {                                                              \
      const cudaError_t ce = EXP;                                    \
      if( ce != cudaSuccess ) {                                      \
         std::cerr << __FILE__ << ":" << __LINE__                    \
                   << " Failed to execute: " << #EXP << " ("         \
                   << cudaGetErrorString( ce ) << ")"                \
                   << std::endl;                                     \
         return StatusCode::FAILURE;                                 \
      }                                                              \
   } while( false )

/// Helper macro for executing @c StatusCode returning functions
#define SC_CHECK( EXP )                                              \
   do {                                                              \
      auto sc = EXP;                                                 \
      if( ! sc.isSuccess() ) {                                       \
         std::cerr << __FILE__ << ":" << __LINE__                    \
                   << " Failed to execute: " << #EXP << std::endl;   \
         return sc;                                                  \
      }                                                              \
   } while( false )

#endif // CORE_MACROS_H
