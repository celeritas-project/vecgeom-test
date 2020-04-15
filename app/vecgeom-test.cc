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

#include "geom/RootModel.h"
#include "geom/VGGeometryHost.h"
#include "transport/vgthelper.h"

using celeritas::RootModel;

void LoadAndTrack(const char* input_filename) {
  // Load model
  RootModel model(input_filename);
  // Create host geometry
  auto host_geom = std::make_shared<celeritas::VGGeometryHost>(model);
  // Call transport-and-io
  celeritas::RunTransport(host_geom, 2);
}

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  assert(args.size() == 1);
  LoadAndTrack(args[0].c_str());
}
