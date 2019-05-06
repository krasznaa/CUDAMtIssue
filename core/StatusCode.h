// Dear emacs, this is -*- c++ -*-
#ifndef CORE_STATUSCODE_H
#define CORE_STATUSCODE_H

// System include(s).
#include <atomic>

/// Lightweight StatusCode class to use in the CUDA tests
///
/// Copying the basic setup of Gaudi's @c StatusCode, but in a much simpler
/// implementation.
///
/// @author Attila Krasznahorkay <Attila.Krasznahorkay@cern.ch>
///
class StatusCode {

public:
   /// StatusCode types
   enum Code {
      SUCCESS = 0, ///< The function call was successful
      FAILURE = 1  ///< The function call failed
   }; // enum Code

   /// Constructor from an enum value
   StatusCode( Code code );
   /// Copy constructor
   StatusCode( const StatusCode& parent );
   /// Destructor
   ~StatusCode();

   /// Copy assignment
   StatusCode& operator=( const StatusCode& rhs );

   /// Check if the code is a success
   bool isSuccess();
   /// Check if the code is a failure
   bool isFailure();

   /// Conversion operator.
   operator int();

private:
   /// The code of the function call
   Code m_code;
   /// Flag showing whether the user already checked this code or not
   mutable std::atomic< bool > m_checked;

}; // class StatusCode

#endif // CORE_STATUSCODE_H
