//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryHost.cc
//---------------------------------------------------------------------------//
#include "VGGeometryHost.h"

#include <iostream>

#include <VecGeom/gdml/Frontend.h>
#include <VecGeom/management/ABBoxManager.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/volumes/PlacedVolume.h>


using std::cout;
using std::endl;

namespace celeritas {
//---------------------------------------------------------------------------//
// MANAGEMENT
//---------------------------------------------------------------------------//
/*!
 * Construct from a GDML input.
 */
VGGeometryHost::VGGeometryHost(const char* gdml_filename) {
  cout << "::: Loading from GDML" << endl;
  // NOTE: the validation check disabling is missing from vecgeom 1.1.6 and
  // earlier; without it, the VGDML loader may crash.
  constexpr bool validate_xml_schema = false;
  vgdml::Frontend::Load(gdml_filename, validate_xml_schema);
  cout << "::: Initializing tracking information" << endl;
  vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();

  max_id_ = vecgeom::VPlacedVolume::GetIdCount();
  max_depth_ = vecgeom::GeoManager::Instance().getMaxDepth();
}

//---------------------------------------------------------------------------//
/*!
 * Get the label for a placed volume ID
 */
const std::string& VGGeometryHost::IdToLabel(IdType vol_id) const {
  assert(vol_id.Get() < max_id_);
  const auto* vol =
      vecgeom::GeoManager::Instance().FindPlacedVolume(vol_id.Get());
  assert(vol);
  return vol->GetLabel();
}

//---------------------------------------------------------------------------//
/*!
 * Get the ID corresponding to a label
 */
auto VGGeometryHost::LabelToId(const std::string& label) const -> IdType {
  const auto* vol =
      vecgeom::GeoManager::Instance().FindPlacedVolume(label.c_str());
  assert(vol);
  assert(vol->id() < max_id_);
  return IdType{vol->id()};
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
