//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file base/PrimaryTrack.h
//---------------------------------------------------------------------------//
#ifndef base_PrimaryTrack_h
#define base_PrimaryTrack_h

#include <array>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Simple state used to initialize tracks.
 */
struct PrimaryTrack {
  //@{
  //! Type aliases
  using Vec3 = std::array<double, 3>;
  //@}

  Vec3 pos;
  Vec3 dir;
  double energy;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_PrimaryTrack_h
