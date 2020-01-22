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

#include <management/ABBoxManager.h>
#include <management/GeoManager.h>
#include <management/RootGeoManager.h>
#include <navigation/GlobalLocator.h>
#include <navigation/VNavigator.h>

#include "base/Range.h"
#include "base/ScopeRootMessages.h"
#include "base/TypeToString.h"

using celeritas::Range;
using celeritas::TypeToString;
using std::cout;
using std::endl;
using vecgeom::ABBoxManager;
using vecgeom::NavigationState;
using vecgeom::RootGeoManager;
using vecgeom::VPlacedVolume;

int main(int argc, char *argv[]) {
  std::vector<std::string> args(argv + 1, argv + argc);
  assert(args.size() == 1);

  // Load geometry through ROOT
  CELERITAS_SCOPE_ROOT_MESSAGES;
  cout << ">>> Loading geometry file from " << args[0] << endl;
  TGeoManager::Import(args[0].c_str());

  // Print all volume names
  TObjArray *vlist = gGeoManager->GetListOfVolumes();
  for (int i : Range(vlist->GetEntries())) {
    TGeoVolume *vol = dynamic_cast<TGeoVolume *>(vlist->At(i));
    assert(vol);
    std::string volname(vol->GetName());

    cout << "Volume " << i << ": " << volname << " (" << TypeToString(*vol)
         << ")\n";
  }

  // Load ROOT geometry into VecGeom
  cout << "::: Converting ROOT to VecGeom" << endl;
  RootGeoManager::Instance().LoadRootGeometry();
  cout << "::: Initializing tracking information" << endl;
  vecgeom::ABBoxManager::Instance().InitABBoxesForCompleteGeometry();

  RootGeoManager::Instance().PrintNodeTable();

  // States at the current and post-step
  const auto &geo_manager = vecgeom::GeoManager::Instance();
  const int max_depth = geo_manager.getMaxDepth();
  NavigationState *cur_state = NavigationState::MakeInstance(max_depth);
  NavigationState *next_state = NavigationState::MakeInstance(max_depth);

  vecgeom::Vector3D<double> pos{-100, 0, 0};
  vecgeom::Vector3D<double> dir{1, 0, 0};
  cur_state->Clear();

  cout << ">>> Tracking from " << pos << " along " << dir << endl;
  // Locate daughter volume, updating cur_state
  const VPlacedVolume *cur_vol = geo_manager.GetWorld();
  const bool contains_point = true;
  cur_vol = vecgeom::GlobalLocator::LocateGlobalPoint(cur_vol, pos, *cur_state,
                                                      contains_point);
  assert(cur_vol);

  while (!cur_state->IsOutside()) {
    next_state->Clear();
    vecgeom::VNavigator const *navigator =
        cur_vol->GetLogicalVolume()->GetNavigator();
    double step = navigator->ComputeStepAndPropagatedState(
        pos, dir, vecgeom::kInfLength, *cur_state, *next_state);

    // Propagate
    cout << "Step " << step << " from " << cur_vol->GetLabel() << "(id "
         << cur_vol->id() << ", type " << TypeToString(*cur_vol) << ")\n";
    pos += dir * (step + 1e-6);
    std::swap(cur_state, next_state);
    cur_vol = cur_state->Top();
    assert(cur_vol || cur_state->IsOutside());
  }
  cout << "Completed: at point " << pos << endl;
}

//---------------------------------------------------------------------------//
