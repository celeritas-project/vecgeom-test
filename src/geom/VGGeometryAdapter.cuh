//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryAdapter.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometryAdapter_cuh
#define geom_VGGeometryAdapter_cuh

#include <thrust/device_vector.h>

#include "VGGeometry.cuh"
#include "VGGeometryHost.h"

namespace celeritas {

//---------------------------------------------------------------------------//
/*!
 * Manage on-device memory used by our VGGeometry object.
 *
 * Since this is basically all done by the VGGeometry project code, we don't
 * need to manage much.
 */
class VGGeometryAdapter {
 public:
  //@{
  //! Type aliases
  using SPHostGeom = std::shared_ptr<celeritas::VGGeometryHost>;
  //@}

 public:
  //! Construct from a host geometry
  explicit VGGeometryAdapter(SPHostGeom host) : host_geom_(std::move(host)) {}

  // Copy data from host to device memory
  void HostToDevice();

  // Copy data from device to host memory
  void DeviceToHost();

  // Get an on-device object
  VGGeometry DeviceView() const {
    using thrust::raw_pointer_cast;

    VGGeometry::Params params;
    params.world_volume = world_volume_;

    return VGGeometry{params};
  }

 private:
  using ConstPtrVolume = VGGeometry::ConstPtrVolume;

  SPHostGeom host_geom_;
  ConstPtrVolume world_volume_ = nullptr;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGGeometryAdapter_cuh
