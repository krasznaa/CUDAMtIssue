// Dear emacs, this is -*- c++ -*-

// Project include(s).
#include "container/AuxTypeVector.cuh"

// System include(s).
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <vector>

/// Function comparing the payload of a reference vector vs. a test one
template< typename T, typename CONT >
bool isIdentical( AuxTypeVector< T, CONT >& refVec,
                  AuxTypeVector< T, CONT >& testVec ) {

   // Test that the two vectors have the same size.
   if( refVec.size() != testVec.size() ) {
      std::cerr << "refVec.size() = " << refVec.size()
                << ", testVec.size() = " << testVec.size() << std::endl;
      return false;
   }
   // Test that the two vectors have the same payload. If not, print all the
   // differences before returning with a failure. (For better debugging...)
   bool result = true;
   const std::size_t size = testVec.size();
   for( std::size_t i = 0; i < size; ++i ) {
      if( *( static_cast< T* >( refVec.toPtr() ) + i ) !=
          *( static_cast< T* >( testVec.toPtr() ) + i ) ) {
         std::cerr << "refVec[ " << i << " ] = "
                   << *( static_cast< T* >( refVec.toPtr() ) + i )
                   << ", testVec[ " << i << " ] = "
                   << *( static_cast< T* >( testVec.toPtr() ) + i )
                   << std::endl;
         result = false;
      }
   }
   return result;
}

/// Helper function setting the contents of a vector to a hardcoded value
template< typename T, typename CONT >
void setVector( AuxTypeVector< T, CONT >& vec,
                const std::vector< T >& content ) {

   // A security check.
   assert( vec.size() == content.size() );

   // Do the deed.
   for( std::size_t i = 0; i < content.size(); ++i ) {
      *( static_cast< T* >( vec.toPtr() ) + i ) = content[ i ];
   }
   return;
}

template< typename T, typename CONT >
void test() {

   // Fill an object with a starter payload.
   AuxTypeVector< T, CONT > testVec( 10, 10 );
   for( int i = 0; i < 10; ++i ) {
      *( static_cast< T* >( testVec.toPtr() ) + i ) = i;
   }

   // Test the resizing of the vectors.
   assert( testVec.resize( 50 ) == false );
   testVec.resize( 15 );

   // Test shifting the payload of the vectors.
   testVec.shift( 5, 5 );
   testVec.shift( 10, -5 );

   // Prepare a small vector whose content can be inserted into the vector.
   std::vector< T > insVec;
   for( int i = 0; i < 5; ++i ) {
      insVec.push_back( i );
   }

   // Test inserting some content into the vectors.
   testVec.insertMove( 7, &( insVec.front() ), &( insVec.back() ) );
   testVec.insertMove( 10, &( insVec.front() ), &( insVec.back() ) );
   assert( testVec.size() == 23 );

   // Create a vector with what *should* be in the test vector at this point.
   AuxTypeVector< T, CONT > refVec( 23, 23 );
   setVector( refVec, { 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 0, 1, 2, 3, 3, 7, 8, 9,
                        0, 0, 0, 0, 0 } );
   assert( isIdentical( refVec, testVec ) );

   // State that the test was successful.
   std::cout << "test successful" << std::endl;
   return;
}

int main() {

   // Run the test for a few primitive types.
   test< int, cuda::managed_array< int > >();
   test< int, AuxTypeVectorMemory< int > >();

   // Return gracefully.
   return 0;
}
