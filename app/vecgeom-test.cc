//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// Example app for navigating with VecGeom.
//---------------------------------------------------------------------------//

#include "geom/VecGeom.h"

#include <iostream>
#include <string>
#include <vector>

#include "base/ScopedState.h"
#include "base/TypeToString.h"
#include "geom/VecGeomState.h"

using celeritas::PrimaryTrack;
using celeritas::RootModel;
using celeritas::ScopedState;
using celeritas::TypeToString;
using celeritas::VecGeom;
using std::cout;
using std::endl;

void LoadAndTrack(const char* input_filename) {
  // Load model
  RootModel model(input_filename);
  model.PrintGeometry(cout);
  // Transfer to vecgeom
  VecGeom geometry(model);

  // Decide starting parameters
  PrimaryTrack primary;
  primary.pos = {-100, 0, 0};
  primary.dir = {1, 0, 0};

  // Build a state to track on
  ScopedState<VecGeom> state(geometry, primary);

  cout << ">>> Tracking from " << state.Get().pos << " along "
       << state.Get().dir << endl;
  while (geometry.IsInside(state.Get())) {
    geometry.FindNextStep(&state.Get());
    const auto* cur_vol = state.Get().vgstate->Top();
    cout << "Step " << state.Get().next_step << " from " << cur_vol->GetLabel()
         << "(id " << cur_vol->id() << ", type " << TypeToString(*cur_vol)
         << ")\n";

    geometry.MoveNextStep(&state.Get());
  }
  cout << "Completed" << endl;
}

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  assert(args.size() == 1);
  LoadAndTrack(args[0].c_str());
}
