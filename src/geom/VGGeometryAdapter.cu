//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryAdapter.cu
//---------------------------------------------------------------------------//
#include <iostream>

#include <VecGeom/management/CudaManager.h>

#include "VGGeometryAdapter.cuh"

using std::cout;
using std::endl;

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Copy data to device.
 *
 * Note that since vecgeom's data is all stored in global memory, we don't have
 * to access the host pointer here.
 */
void VGGeometryAdapter::HostToDevice() {
  cout << "::: Transferring geometry to GPU" << endl;
  auto& cuda_manager = vecgeom::cxx::CudaManager::Instance();
  cuda_manager.set_verbose(1);
  cuda_manager.LoadGeometry();
  cudaError_t result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);

  auto world_top = cuda_manager.Synchronize();
  assert(world_top != nullptr);
  result = cudaDeviceSynchronize();
  assert(result == cudaSuccess);
  cout << ">>> Synchronized successfully!" << endl;

  cuda_manager.PrintGeometry();

  world_volume_ = cuda_manager.world_gpu();
}
//---------------------------------------------------------------------------//
}  // namespace celeritas
