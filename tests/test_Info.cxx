
// Project include(s).
#include "core/Info.h"

// System include(s).
#undef NDEBUG
#include <cassert>

int main() {

   // Access the singleton object.
   Info& info = Info::instance();

   // Make sure that we can check whether a GPU is available.
   const std::size_t nDevices = static_cast< std::size_t >( info.nDevices() );

   // Now check for some internal consistency in the info object.
   assert( nDevices == info.names().size() );
   assert( nDevices == info.maxThreadsPerBlock().size() );
   assert( nDevices == info.concurrentKernels().size() );
   assert( nDevices == info.totalMemory().size() );

   // Print the collected attributes, just as a "visual test".
   info.print();

   // Return gracefully.
   return 0;
}
