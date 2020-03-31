//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateRef.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGStateRef_cuh
#define geom_VGStateRef_cuh

#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>

#include "base/Types.h"
#include "base/array.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * On-device reference to a thread-local particle state.
 *
 * This simple class should *only* provide accessors for obtaining and setting
 * data that resides in global device memory.
 */
class VGStateRef {
 public:
  //@{
  //! Type aliases
  using NavState = vecgeom::cuda::NavigationState;
  using Volume = vecgeom::cuda::VPlacedVolume;
  using ConstPtrVolume = const Volume*;
  //@}

  //! Construction parameters
  struct Params {
    ConstPtrVolume* volume_handle;
    NavState* vgstate;
    NavState* vgnext;

    Real3* pos;
    Real3* dir;
    real_type* next_step;
  };

 public:
  explicit __device__ VGStateRef(const Params& params) : data_(params) {}

  //@{
  //! Access the state's current VecGeom navigation state
  __device__ const NavState& vgstate() const { return *data_.vgstate; }
  __device__ NavState& vgstate() { return *data_.vgstate; }
  //@}

  //@{
  //! Access the state's current volume pointer
  __device__ const Volume* volume() const { return this->vgstate().Top(); }
  //@}

  //@{
  //! Access the state's *next* VecGeom navigation state
  __device__ const NavState& vgnext() const { return *data_.vgnext; }
  __device__ NavState& vgnext() { return *data_.vgnext; }
  //@}

  //@{
  //! Access the state's current position
  __device__ const Real3& pos() const { return *data_.pos; }
  __device__ void pos(const Real3& p) { *data_.pos = p; }
  //@}

  //@{
  //! Access the state's current direction
  __device__ const Real3& dir() const { return *data_.dir; }
  __device__ void dir(const Real3& d) { *data_.dir = d; }
  //@}

  //@{
  //! Access the state's current direction
  __device__ real_type next_step() const { return *data_.next_step; }
  __device__ void next_step(real_type ns) { *data_.next_step = ns; }
  //@}

 private:
  const Params data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGStateRef_cuh
