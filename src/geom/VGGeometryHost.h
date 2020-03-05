//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geom/VGGeometryHost.h
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometry_h
#define geom_VGGeometry_h

#include <iosfwd>

#include "base/PrimaryTrack.h"
#include "geom/RootModel.h"

namespace celeritas {

class VGGeometryDeviceStorage;
class VGGeometryDevice;

//---------------------------------------------------------------------------//
/*!
 * Manage a CUDA VGGeometry geometry.
 */
class VGGeometryHost {
 public:
  //@{
  //! Type aliases
  using DeviceStorageType = VGGeometryDeviceStorage;
  using DeviceType = VGGeometryDevice;
  //@}

 public:
  // Construct from a ROOT model
  explicit VGGeometryHost(const RootModel& model);

  void HostToDevice(VGGeometryDeviceStorage* device_storage) const;
  void DeviceToHost(const VGGeometryDeviceStorage& device_storage);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGGeometry_h
