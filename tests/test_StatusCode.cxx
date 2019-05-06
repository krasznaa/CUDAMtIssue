
// Project include(s).
#include "core/StatusCode.h"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <utility>

int main() {

   // Test a successful code.
   StatusCode code1( StatusCode::SUCCESS );
   assert( code1.isSuccess() == true );
   assert( code1.isFailure() == false );

   // Test a failure code.
   StatusCode code2( StatusCode::FAILURE );
   assert( code2.isSuccess() == false );
   assert( code2.isFailure() == true );

   // Test the move constructor.
   StatusCode code3( StatusCode::SUCCESS );
   StatusCode code4( std::move( code3 ) );
   assert( code4.isSuccess() == true );
   assert( code4.isFailure() == false );

   // Test the move assignment.
   StatusCode code5( StatusCode::FAILURE );
   StatusCode code6( StatusCode::SUCCESS );
   code6 = std::move( code5 );
   assert( code6.isSuccess() == false );
   assert( code6.isFailure() == true );

   // Return gracefully.
   return 0;
}
