//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryHost.cc
//---------------------------------------------------------------------------//
#include "VGGeometryHost.h"

#include <iostream>

#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/management/RootGeoManager.h>

#include "RootModel.h"

using std::cout;
using std::endl;

namespace celeritas {
//---------------------------------------------------------------------------//
// MANAGEMENT
//---------------------------------------------------------------------------//
/*!
 * Construct from a ROOT model.
 */
VGGeometryHost::VGGeometryHost(const RootModel& model) {
  cout << "::: Converting ROOT to VGGeometry" << endl;
  vecgeom::RootGeoManager::Instance().LoadRootGeometry();
  cout << "::: Initializing tracking information" << endl;
  vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();
}

//---------------------------------------------------------------------------//
/*!
 * Maximum nested geometry depth, needed for navigation state allocation
 */
int VGGeometryHost::MaxDepth() const {
  const auto& geo_manager = vecgeom::GeoManager::Instance();
  return geo_manager.getMaxDepth();
}

//---------------------------------------------------------------------------//
/*!
 * Copy data to device
 */
void VGGeometryHost::LoadCudaGeometryManager() const {
  cout << "::: Transferring geometry to GPU" << endl;
  auto& cuda_manager = vecgeom::CudaManager::Instance();
  cuda_manager.set_verbose(3);
  cuda_manager.LoadGeometry();
  cuda_manager.Synchronize();
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
