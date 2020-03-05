//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometry.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometry_cuh
#define geom_VGGeometry_cuh

#include <VecGeom/volumes/PlacedVolume.h>

#include "VGStateRef.cuh"
#include "base/Types.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Track on a VGGeometry geometry.
 */
class VGGeometry {
 public:
  //@{
  //! Type aliases
  using Volume = VecGeom::cuda::VPlacedVolume;
  using ConstPtrVolume = const Volume*;
  using StateRef = VGStateRef;
  //@}

  //! Construction parameters
  struct Params {
    ConstPtrVolume world_volume = nullptr;
  };

 public:
  explicit __host__ VGGeometry(const Params& params) : data_(params) {}

  // Construct a state
  __device__ inline void Construct(StateRef& state,
                                   const InitialStateRef& primary) const;

  // Whether the state is inside the geometry
  __device__ inline bool IsInside(const StateRef& state) const;

  // Calculate the distance to the next boundary crossing
  __device__ inline void FindNextStep(StateRef& state) const;

  // Get the distance to the next straight-line boundary crossing
  __device__ inline real_type NextStep(const StateRef& state) const;

  // Move to the next boundary
  __device__ inline void MoveNextStep(StateRef& state) const;

  // Destroy a state
  __device__ inline void Destroy(StateRef&) const;

  // Fudge factor for movement (absolute distance)
  __device__ static constexpr double StepFudge() { return 1e-6; }

 private:
  Params data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "VGGeometry.i.cuh"

#endif  // geom_VGGeometry_cuh
