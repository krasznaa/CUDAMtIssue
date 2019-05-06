
// Local include(s).
#include "StatusCode.h"

// System include(s).
#include <cstdlib>
#include <iostream>

StatusCode::StatusCode( Code code )
: m_code( code ), m_checked( false ) {

}

StatusCode::StatusCode( const StatusCode& parent )
: m_code( parent.m_code ),
  m_checked( static_cast< bool >( parent.m_checked ) ) {

   parent.m_checked = true;
}

StatusCode::~StatusCode() {
   if( ! m_checked ) {
      std::cerr << "AthCUDA::StatusCode was not checked!" << std::endl;
      std::abort();
   }
}

StatusCode& StatusCode::operator=( const StatusCode& rhs ) {

   m_code = rhs.m_code;
   m_checked = static_cast< bool >( rhs.m_checked );
   rhs.m_checked = true;

   return *this;
}

bool StatusCode::isSuccess() {

   m_checked = true;
   return ( m_code == SUCCESS );
}

bool StatusCode::isFailure() {

   m_checked = true;
   return ( m_code == FAILURE );
}

StatusCode::operator int() {

   m_checked = true;
   return static_cast< int >( m_code );
}
