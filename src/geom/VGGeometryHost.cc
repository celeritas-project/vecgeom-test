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

namespace {
unsigned int FindMaxVolumeId() {
  using CVolPtr = const vecgeom::VPlacedVolume*;
  std::vector<CVolPtr> volumes;
  vecgeom::GeoManager::Instance().getAllPlacedVolumes(volumes);

  auto iter = std::max_element(
      volumes.begin(), volumes.end(),
      [](CVolPtr left, CVolPtr right) { return left->id() < right->id(); });
  return (*iter)->id();
}
}  // namespace

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

  num_visits_.resize(FindMaxVolumeId() + 1);
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID
 */
const std::string& VGGeometryHost::IdToLabel(UniqueCellId vol_id) const {
  const auto* vol =
      vecgeom::GeoManager::Instance().FindPlacedVolume(vol_id.Get());
  assert(vol);
  return vol->GetLabel();
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label
 */
auto VGGeometryHost::LabelToId(const std::string& label) const -> UniqueCellId {
  const auto* vol =
      vecgeom::GeoManager::Instance().FindPlacedVolume(label.c_str());
  assert(vol);
  return UniqueCellId{vol->id()};
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
}  // namespace celeritas
