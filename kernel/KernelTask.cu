// Dear emacs, this is -*- c++ -*-

// Local include(s).
#include "KernelTask.cuh"

KernelTask::KernelTask()
: m_mutex() {

   m_mutex.lock();
}

KernelTask::mutex_t& KernelTask::mutex() {

   return m_mutex;
}

void KernelTask::setFinished() {

   m_mutex.unlock();
   return;
}

StatusCode KernelTask::code() const {

   return m_code;
}

void KernelTask::setCode( StatusCode code ) {

   m_code = code;
   return;
}
