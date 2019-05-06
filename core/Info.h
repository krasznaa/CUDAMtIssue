// Dear emacs, this is -*- c++ -*-
#ifndef CORE_INFO_H
#define CORE_INFO_H

// System include(s).
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

/// Class providing information about the CUDA devices at runtime
///
/// This is meant to be most useful when we build the code with CUDA
/// available, but then try to execute the code on a machine that has no
/// GPUs.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class Info {

public:
   /// Singleton accessor function
   static Info& instance();

   /// Get the number of available devices
   int nDevices() const;

   /// Get the names of all of the devices
   const std::vector< std::string >& names() const;
   /// Get the maximum number of threads per execution block on all devices
   const std::vector< int >& maxThreadsPerBlock() const;
   /// Get the multi-kernel capabilities of all devices
   const std::vector< bool >& concurrentKernels() const;
   /// Get the total amount of memory for all devices
   const std::vector< std::size_t >& totalMemory() const;

   /// Print the properties of all detected devices
   void print() const;

private:
   /// The constructor is private to implement the singleton behaviour
   Info();

   /// The number of CUDA devices available at runtime
   int m_nDevices = 0;
   /// The names of all available devices
   std::vector< std::string > m_names;
   /// The maximum number of threads per block for each available device
   std::vector< int > m_maxThreadsPerBlock;
   /// The multi-kernel capabilities of all devices
   std::vector< bool > m_concurrentKernels;
   /// The total amount of memory for all devices
   std::vector< std::size_t > m_totalMemory;

}; // class Info

namespace std {

   /// Print operator for @c AthCUDA::Info
   std::ostream& operator<< ( std::ostream& out, const Info& info );

} // namespace std

#endif // CORE_INFO_H
