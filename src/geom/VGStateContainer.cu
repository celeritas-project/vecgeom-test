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
/*!
 * Construct from number of states and the root geometry model
 */
VGStateContainer::VGStateContainer(ssize_type size, const VGGeometryHost& geom)
    : size_(size),
      vgmaxdepth_(geom.MaxDepth()),
      volume_ptrs_(size_),
      vgstate_(size_, vgmaxdepth_),
      vgnext_(size_, vgmaxdepth_),
      pos_(size_),
      dir_(size_),
      next_step_(size_) {}

//---------------------------------------------------------------------------//
}  // namespace celeritas
