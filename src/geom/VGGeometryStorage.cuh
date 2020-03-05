//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGGeometryStorage.cuh
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometryStorage_cuh
#define geom_VGGeometryStorage_cuh

#include "VGGeometry.cuh"

namespace celeritas {

//---------------------------------------------------------------------------//
/*!
 * Manage on-device memory used by our VGGeometry object.
 *
 * Since this is basically all done by the VGGeometry project code, we don't
 * need to manage much.
 */
struct VGGeometryStorage {};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "VGGeometryStorage.i.cuh"

#endif  // geom_VGGeometryStorage_cuh
