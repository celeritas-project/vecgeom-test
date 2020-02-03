//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geom/VecGeom.h
//---------------------------------------------------------------------------//
#ifndef geom_VecGeom_h
#define geom_VecGeom_h

#include <iosfwd>

#include "base/PrimaryTrack.h"
#include "geom/RootModel.h"

namespace celeritas {

class VecGeomState;

//---------------------------------------------------------------------------//
/*!
 * Manage a VecGeom geometry.
 */
class VecGeom {
 public:
  //@{
  //! Type aliases
  using InitialState = PrimaryTrack;
  using State = VecGeomState;
  //@}

 public:
  // Construct from a ROOT model
  explicit VecGeom(const RootModel& model);

  // Construct a state
  void Construct(VecGeomState* state, const InitialState& primary) const;

  // Destroy a state
  void Destroy(VecGeomState* state) const;

  //--- TRACKING ---//

  bool IsInside(const VecGeomState& state) const;
  void FindNextStep(VecGeomState* state) const;
  double NextStep(const VecGeomState& state) const;
  void MoveNextStep(VecGeomState* state) const;

  static constexpr double StepFudge() { return 1e-6; }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VecGeom_h
