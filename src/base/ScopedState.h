//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file base/ScopedState.h
//---------------------------------------------------------------------------//
#ifndef base_ScopedState_h
#define base_ScopedState_h

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * RAII class for managing a state through its associated model.
 *
 * This should only be used with local scope.
 * \code
   Geometry my_geo(filename);

   {
     ScopedState sstate(my_geo);
     my_code.transport(sstate.Get());
   }
   \endcode
 */
template <class Invariant>
class ScopedState {
 public:
  //@{
  //! Type aliases
  using InitialState = typename Invariant::InitialState;
  using State = typename Invariant::State;
  //@}

 public:
  // Construct with invariant and initial state
  inline ScopedState(const Invariant& inv, const InitialState& init);

  // Destroy using invariant
  inline ~ScopedState();

  //@{
  //! Access the temporary state
  const State& Get() const { return state_; }
  State& Get() { return state_; }
  //@}

 private:
  const Invariant& invariant_;
  State state_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "ScopedState.i.h"

#endif  // base_ScopedState_h
