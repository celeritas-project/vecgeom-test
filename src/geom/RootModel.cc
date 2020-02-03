//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootModel.cc
//---------------------------------------------------------------------------//
#include "RootModel.h"

#include <iostream>

#include <TGeoManager.h>

#include "base/Range.h"
#include "base/ScopeRootMessages.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Load ROOT model from a .root or .gdml file.
 */
RootModel::RootModel(const char *root_file) {
  using std::cout;
  using std::endl;

  // Load geometry through ROOT
  CELERITAS_SCOPE_ROOT_MESSAGES;
  cout << ">>> Loading geometry file from " << root_file << endl;
  TGeoManager::Import(root_file);
}

//---------------------------------------------------------------------------//
/*!
 * Print volume properties for debugging.
 */
void RootModel::PrintGeometry(std::ostream &os) const {
  TObjArray *vlist = gGeoManager->GetListOfVolumes();
  for (int i : Range(vlist->GetEntries())) {
    TGeoVolume *vol = dynamic_cast<TGeoVolume *>(vlist->At(i));
    assert(vol);
    std::string volname(vol->GetName());

    std::cout << "Volume " << i << ": " << volname << '\n';
  }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
