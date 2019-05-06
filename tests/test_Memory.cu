
// Local include(s).
#include "core/Memory.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <cmath>

int main() {

   // Allocate some test arrays.
   auto array1 = cuda::make_managed_array< float >( 500 );
   auto array2 = cuda::make_managed_array< int >( 100 );
   auto array3 = cuda::make_managed_array< double >( 1000 );

   // Make sure that they can be filled.
   for( int i = 0; i < 10; ++i ) {
      array1.get()[ i ] = 1.2f;
      array2.get()[ i ] = 123;
      array3.get()[ i ] = 3.141592f;
   }

   // Allocate some (device) arrays.
   cuda::array< int > array4( 100 );
   cuda::array< float > array5;
   array5.resize( 200 );

   // Make sure that we can write to these.
   for( int i = 0; i < 10; ++i ) {
      array4[ i ] = i;
      array5[ i ] = i * 3.141592f;
   }

   // Check that resizing works as intended.
   array5.resize( 100 );
   array5.resize( 500 );
   for( int i = 0; i < 10; ++i ) {
      assert( std::abs( array5[ i ] - i * 3.141592f ) < 0.0001 );
   }

   // Return gracefully.
   return 0;
}
