//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryHost.cc
//---------------------------------------------------------------------------//
#include "VGGeometryHost.h"

#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/RootGeoManager.h>

namespace celeritas {
//---------------------------------------------------------------------------//
// MANAGEMENT
//---------------------------------------------------------------------------//
/*!
 * Construct from a ROOT model.
 */
VGGeometryHost::VGGeometryHost(const RootModel& model) {
  using std::cout;
  using std::endl;

  cout << "::: Converting ROOT to VGGeometry" << endl;
  vecgeom::RootGeoManager::Instance().LoadRootGeometry();
  cout << "::: Initializing tracking information" << endl;
  vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device
 */
void VGGeometryHost::HostToDevice(
    VGGeometryDeviceStorage* device_storage) const {
  cout << "::: Transferring geometry to GPU" << endl;
  auto& cuda_manager = vecgeom::CudaManager::Instance();
  cuda_manager.set_verbose(3);
  cuda_manager.LoadGeometry(vecgeom::GeoManager::Instance().GetWorld());
  cuda_manager.Synchronize();
  device_storage->world
}

//---------------------------------------------------------------------------//
/*!
 * Copy data back to host (null-op)
 */
void VGGeometryHost::DeviceToHost(
    const VGGeometryDeviceStorage& device_storage) {}

//---------------------------------------------------------------------------//
}  // namespace celeritas
