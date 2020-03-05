//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryHost.cu
//---------------------------------------------------------------------------//
#include <iostream>

#include <VecGeom/management/CudaManager.h>

#include "VGGeometry.cuh"
#include "VGGeometryHost.h"
#include "VGGeometryStorage.cuh"
#include "base/Mirror.h"

using std::cout;
using std::endl;

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Copy data to device
 */
void VGGeometryHost::HostToDevice(VGGeometryStorage* device_storage) const {
  cout << "::: Transferring geometry to GPU" << endl;
  auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
  cuda_manager.set_verbose(3);
  cuda_manager.LoadGeometry();
  cuda_manager.Synchronize();
  device_storage->world_volume = cuda_manager.world_gpu();
}

//---------------------------------------------------------------------------//
/*!
 * Copy data back to host (null-op)
 */
void VGGeometryHost::DeviceToHost(const VGGeometryStorage& device_storage) {}

//---------------------------------------------------------------------------//
/*!
 * Create a host/device mirror from a root model and transfer memory to device
 */
std::shared_ptr<Mirror<VGGeometryHost>> MakeVGMirror(const RootModel& model) {
  auto result = std::make_shared<Mirror<VGGeometryHost>>(model);
  result->HostToDevice();
  return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
