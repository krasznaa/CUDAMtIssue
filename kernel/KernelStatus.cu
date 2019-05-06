// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "KernelStatus.cuh"

// System include(s).
#include <atomic>
#include <mutex>
#include <utility>

/// Counter keeping track of all status objects
static std::atomic< std::size_t > s_count( 0 );

KernelStatus::KernelStatus()
: m_task() {

}

KernelStatus::KernelStatus( std::shared_ptr< KernelTask > task )
: m_task( task ) {

   if( m_task ) {
      s_count += 1;
   }
}

KernelStatus::KernelStatus( const KernelStatus& parent )
: m_task( parent.m_task ) {

   if( m_task ) {
      s_count += 1;
   }
}

KernelStatus::KernelStatus( KernelStatus&& parent )
: m_task( std::move( parent.m_task ) ) {

}

KernelStatus::~KernelStatus() {

   if( m_task ) {
      s_count -= 1;
   }
}

KernelStatus& KernelStatus::operator=( const KernelStatus& rhs ) {

   if( this == &rhs ) {
      return *this;
   }

   --s_count;
   m_task = rhs.m_task;
   if( m_task ) {
      ++s_count;
   }
   return *this;
}

KernelStatus& KernelStatus::operator=( KernelStatus&& rhs ) {

   if( this == &rhs ) {
      return *this;
   }

   m_task = std::move( rhs.m_task );
   return *this;
}

StatusCode KernelStatus::wait() {

   // Check whether a task was given to the object.
   if( ! m_task ) {
      return StatusCode::SUCCESS;
   }

   // If yes, wait for it to finish.
   KernelTask::lock_t lock( m_task->mutex() );
   return m_task->code();
}

std::size_t KernelStatus::count() {

   return s_count;
}
