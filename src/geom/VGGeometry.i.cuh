//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.i.cuh
//---------------------------------------------------------------------------//

#include "base/Utility.cuh"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Construct states.
 */
__device__ void VGGeometry::Construct(StateRef& state,
                                      const InitialStateRef& primary) const {
  // Initialize position/direction
  state.pos(primary.pos());
  state.dir(primary.dir());

  // Set up current state and locate daughter volume.
  // Note that LocateGlobalPoint sets state->vgstate.
  state.vgstate().Clear();
  auto volume = data_.world_volume;
  const bool contains_point = true;
  volume = VGGeometry::GlobalLocator::LocateGlobalPoint(
      volume, ToVector(state.pos()), state.vgstate(), contains_point);

  // assert(volume);
  state.volume(volume);

  // Set up next state
  state.vgstate().Clear();
  state.next_step(std::numeric_limits<real_type>::quiet_NaN());
}

//---------------------------------------------------------------------------//
/*!
 * Return whether the point is inside the valid geometry region
 */
__device__ bool VGGeometry::IsInside(const StateRef& state) const {
  return !state.vgstate().IsOutside();
}

//---------------------------------------------------------------------------//
/*!
 * Find the distance to the next geometric boundary.
 */
__device__ void VGGeometry::FindNextStep(StateRef& state) const {
  VGGeometry::VNavigator const* navigator =
      state.volume()->GetLogicalVolume()->GetNavigator();
  real_type next_step = navigator->ComputeStepAndPropagatedState(
      ToVector(state.pos()), ToVector(state.dir()), VGGeometry::kInfLength,
      state.vgstate(), state.vgnext());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return the next straight-line distance to the boundary.
 */
__device__ real_type VGGeometry::NextStep(const StateRef& state) const {
  return state.next_step;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Move to the next boundary.
 */
__device__ void VGGeometry::MoveNextStep(StateRef& state) const {
  state.vgstate() = state.vgnext();
  state.volume(state.vgstate().Top());
  state.vgnext().Clear();

  // Move the next step plus an extra fudge distance
  Real3 newpos(state.pos());
  axpy(state.next_step() + VGGeometry::StepFudge(), state.dir(), &newpos);
  state.pos(newpos);
}

//---------------------------------------------------------------------------//
/*!
 * Destroy and invalidate a state
 */
__device__ void VGGeometry::Destroy(StateRef& state,
                                    const InitialStateRef& primary) const {
  state.vgstate().Clear();
  state.vgnext().Clear();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
