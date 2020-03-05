//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateContainer.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGStateContainer_cuh
#define geom_VGStateContainer_cuh

#include <VecGeom/navigation/NavStatePool.h>
#include <thrust/device_vector.h>

#include "VGStateView.h"
#include "base/Types.h"

namespace celeritas {
class VGGeometryHost;

//---------------------------------------------------------------------------//
/*!
 * Manage memory for multiple geometry states on the device.
 */
class VGStateContainer {
 public:
  //@{
  //! Type aliases
  using ssize_type = celeritas::ssize_type;
  using Volume = vecgeom::cuda::VPlacedVolume;
  //@}

 public:
  // Allow default constructor for delayed construction
  VGStateContainer();

  // Construct from number of states and the root geometry model
  VGStateContainer(ssize_type size, const VGGeometryHost& host);

  // Emit a view to on-device memory
  inline VGStateView View();

  //! Number of states
  ssize_type Size() const { return size_; }

 private:
  ssize_type size_ = 0;
  ssize_type vgmaxdepth_ = 0;
  thrust::device_vector<const Volume*> volume_;
  vecgeom::cuda::NavStatePool vgstate_;
  vecgeom::cuda::NavStatePool vgnext_;
  thrust::device_vector<Real3> pos_;
  thrust::device_vector<Real3> dir_;
  thrust::device_vector<double> next_step_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "VGStateContainer.i.cuh"

#endif  // geom_VGStateContainer_cuh
