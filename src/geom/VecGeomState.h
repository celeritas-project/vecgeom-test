//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VecGeomState.h
//---------------------------------------------------------------------------//
#ifndef geom_VecGeomState_h
#define geom_VecGeomState_h

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/volumes/PlacedVolume.h>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * State for VecGeom tracking through a geometry.
 */
struct VecGeomState {
  const vecgeom::VPlacedVolume *volume;
  vecgeom::NavigationState *vgstate;
  vecgeom::NavigationState *vgnext;

  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  double next_step;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VecGeomState_h
