//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateContainer.cu
//---------------------------------------------------------------------------//
#include "VGGeometryHost.h"
#include "VGStateContainer.cuh"

namespace celeritas {
//---------------------------------------------------------------------------//
//! Default constructor
VGStateContainer::VGStateContainer() = default;

//---------------------------------------------------------------------------//
/*!
 * Construct from number of states and the root geometry model
 */
VGStateContainer::VGStateContainer(size_type size, const VGGeometryHost& geom)
    : size_(size),
      vgmaxdepth_(geom.MaxDepth()),
      vgstate_(size_, maxdepth_),
      vgnext_(size, maxdepth_),
      pos_(size),
      dir_(size),
      next_step_(size) {}

//---------------------------------------------------------------------------//
}  // namespace celeritas
