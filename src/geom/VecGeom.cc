//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecGeom.cc
//---------------------------------------------------------------------------//
#include "VecGeom.h"

#include <cmath>
#include <iostream>
#include <limits>

#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/RootGeoManager.h>
#include <VecGeom/navigation/GlobalLocator.h>
#include <VecGeom/navigation/VNavigator.h>

#include "VecGeomState.h"
#include "base/TypeToString.h"
#include "detail/Compatibility.h"

namespace celeritas {
//---------------------------------------------------------------------------//
// MANAGEMENT
//---------------------------------------------------------------------------//
/*!
 * Construct from a ROOT model.
 */
VecGeom::VecGeom(const RootModel& model) {
  using std::cout;
  using std::endl;

  cout << "::: Converting ROOT to VecGeom" << endl;
  vecgeom::RootGeoManager::Instance().LoadRootGeometry();
  cout << "::: Initializing tracking information" << endl;
  vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
}

//---------------------------------------------------------------------------//
/*!
 * Construct a state.
 */
void VecGeom::Construct(VecGeomState* state,
                        const InitialState& primary) const {
  using Vec3 = vecgeom::Vector3D<double>;
  using VgState = vecgeom::NavigationState;

  const auto& geo_manager = vecgeom::GeoManager::Instance();
  const int max_depth = geo_manager.getMaxDepth();

  // Initialize position/direction
  state->pos = ToVector(primary.pos);
  state->dir = ToVector(primary.dir);

  // Set up current state and locate daughter volume.
  state->vgstate = VgState::MakeInstance(max_depth);
  state->vgstate->Clear();
  state->volume = geo_manager.GetWorld();
  const bool contains_point = true;
  // Note that LocateGlobalPoint sets state->vgstate.
  state->volume = vecgeom::GlobalLocator::LocateGlobalPoint(
      state->volume, state->pos, *state->vgstate, contains_point);
  assert(state->volume);

  // Set up next state
  state->vgnext = VgState::MakeInstance(max_depth);
  state->vgnext->Clear();
  state->next_step = std::numeric_limits<double>::quiet_NaN();
}

//---------------------------------------------------------------------------//
/*!
 * Destroy a state.
 */
void VecGeom::Destroy(VecGeomState* state) const {
  using VgState = vecgeom::NavigationState;
  VgState::ReleaseInstance(state->vgstate);
  VgState::ReleaseInstance(state->vgnext);
}

//---------------------------------------------------------------------------//
// TRACKING
//---------------------------------------------------------------------------//
/*!
 * Return whether the point is inside the valid geometry region
 */
bool VecGeom::IsInside(const VecGeomState& state) const {
  return !state.vgstate->IsOutside();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
void VecGeom::FindNextStep(VecGeomState* state) const {
  vecgeom::VNavigator const* navigator =
      state->volume->GetLogicalVolume()->GetNavigator();
  state->next_step = navigator->ComputeStepAndPropagatedState(
      state->pos, state->dir, vecgeom::kInfLength, *state->vgstate,
      *state->vgnext);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return the next straight-line distance to the boundary.
 */
double VecGeom::NextStep(const VecGeomState& state) const {
  return state.next_step;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Move to the next boundary.
 */
void VecGeom::MoveNextStep(VecGeomState* state) const {
  std::swap(state->vgstate, state->vgnext);
  state->volume = state->vgstate->Top();
  state->pos += state->dir * (state->next_step + this->StepFudge());
  state->vgnext->Clear();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
