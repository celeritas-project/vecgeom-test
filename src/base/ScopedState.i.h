//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ScopedState.i.h
//---------------------------------------------------------------------------//

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Construct with the invariant and initial state.
 */
template <class Invariant>
ScopedState<Invariant>::ScopedState(const Invariant& inv,
                                    const InitialState& init)
    : invariant_(inv), state_{} {
  inv.Construct(&state_, init);
}

//---------------------------------------------------------------------------//
/*!
 * Destroy using invariant.
 */
template <class Invariant>
ScopedState<Invariant>::~ScopedState() {
  invariant_.Destroy(&state_);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
