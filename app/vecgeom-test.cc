//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// Example app for navigating with VecGeom.
//---------------------------------------------------------------------------//

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <TGeoManager.h>

#include "Range.h"

using celeritas::Range;
using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  assert(args.size() == 1);

  // Load geometry through ROOT
  cout << "Loading geometry file from " << args[0] << endl;
  TGeoManager::Import(args[0].c_str());

  // now try to find shape with logical volume name given on the command line
  TObjArray *vlist = gGeoManager->GetListOfVolumes();
  for (int i : Range(vlist->GetEntries())) {
    TGeoVolume *vol = dynamic_cast<TGeoVolume *>(vlist->At(i));
    assert(vol);
    std::string volname(vol->GetName());

    cout << "Volume " << i << ": " << volname << endl;
  }
}

//---------------------------------------------------------------------------//
