//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateView.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGStateView_cuh
#define geom_VGStateView_cuh

#include <VecGeom/navigation/NavigationState.h>

#include "VGStateRef.cuh"

namespace celeritas {

//---------------------------------------------------------------------------//
/*!
 * View to a vector of VecGeom state information.
 *
 * This "view" is expected to be an argument to a geometry-related kernel
 * launch.
 *
 * The \c vgstate and \c vgnext arguments must be the result of
 * vecgeom::NavStateContainer::GetGPUPointer; and they are only meaningful with
 * the corresponding \c vgmaxdepth, the result of \c GeoManager::getMaxDepth .
 */
class VGStateView {
 public:
  //@{
  //! Type aliases
  using size_type = celeritas::ssize_type;
  using value_type = VGStateRef;
  using NavState = VGStateRef::NavState;
  using ConstPtrVolume = VGStateRef::ConstPtrVolume;
  //@}

  //! Construction parameters
  struct Params {
    ssize_type size = 0;
    ssize_type vgmaxdepth = 0;
    ConstPtrVolume* volume_handle = nullptr;
    void* vgstate = nullptr;
    void* vgnext = nullptr;

    Real3* pos = nullptr;
    Real3* dir = nullptr;
    real_type* next_step = nullptr;
  };

 public:
  //! Construct with invariant parameters
  explicit VGStateView(const Params& params) : data_(params) {}

  //! Number of states
  __device__ size_type size() const { return data_.size; }

  //! Get a reference to the local state for a thread
  __device__ value_type operator[](ssize_type idx) const {
    VGStateRef::Params params;
    params.volume_handle = volume_handle + idx;
    params.vgstate = this->GetNavState(data_.vgstate);
    params.vgnext = this->GetNavState(data_.vgnext);
    params.pos = data_.pos + idx;
    params.dir = data_.dir + idx;
    params.next_step = next_step + idx;
  }

 private:
  __device__ NavState* GetNavState(void* state, ssize_type idx) const {
    char* ptr = reinterpret_cast<char*>(state);
    ptr += vecgeom::cuda::NavigationState::SizeOf(data_.vgmaxdepth) * idx;
    return reinterpret_cast<NavState*>(ptr);
  }

 private:
  Params data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGStateView_cuh
