//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryDeviceStorage.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometryDeviceStorage_cuh
#define geom_VGGeometryDeviceStorage_cuh

#include "VGGeometry.cuh"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Manage on-device memory used by our VGGeometry object.
 *
 * Since this is basically all done by the VGGeometry project code, we don't
 * need to manage much.
 */
struct VGGeometryStorage {
  using ConstPtrVolume = VGGeometry::ConstPtrVolume;

  ConstPtrVolume world_volume = nullptr;

  //! Get an on-device object
  VGGeometry View() const { return VGGeometry({world_volume}); }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGGeometryDeviceStorage_cuh
