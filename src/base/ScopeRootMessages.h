//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// RootErrorWrapper class.
//---------------------------------------------------------------------------//

#ifndef celeritas_ScopeRootMessages_h
#define celeritas_ScopeRootMessages_h

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * \def CELERITAS_SCOPE_ROOT_MESSAGES
 *
 * Set up a log message handler for root local to this scope.
 *
 * \code
     void myfunc() {
       CELERITAS_SCOPE_ROOT_MESSAGES;
       noisy_root_call();
     }
   \endcode
 */
#define CELERITAS_SCOPE_ROOT_MESSAGES \
  celeritas::ScopeRootMessages scope_messages_##__LINE__

//---------------------------------------------------------------------------//
/*!
 * \class ScopeRootMessages
 *
 * Control ROOT logging and error handling.
 */
class ScopeRootMessages {
 public:
  ScopeRootMessages();
  ~ScopeRootMessages();

 private:
  using ErrHandlerT = void (*)(int, bool, const char *, const char *);
  ErrHandlerT original_handler_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif
