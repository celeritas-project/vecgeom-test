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

using celeritas::RootModel;

void LoadAndTrack(const char* input_filename) {
  // Load model
  RootModel model(input_filename);
  model.PrintGeometry(std::cout);

  // Create host-device mirror
  auto geom_mirror_sp = celeritas::MakeVGMirror(model);
}

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  assert(args.size() == 1);
  LoadAndTrack(args[0].c_str());
}
