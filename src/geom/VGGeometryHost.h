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
#include <memory>

namespace celeritas {
class RootModel;
class VGGeometryDeviceStorage;
class VGGeometryDevice;
template <class T>
class Mirror;

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

  // Copy geometry to device
  void HostToDevice(VGGeometryDeviceStorage* device_storage) const;
  // Copy simulation data back to host
  void DeviceToHost(const VGGeometryDeviceStorage& device_storage);

  // Maximum nested geometry depth
  int MaxDepth() const;
};

// Create a host/device mirror from a root model
std::shared_ptr<Mirror<VGGeometryHost>> MakeVGMirror(const RootModel& model);

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGGeometry_h
