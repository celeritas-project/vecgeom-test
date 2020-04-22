//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStatePool.cc
//---------------------------------------------------------------------------//
#include "VGNavStatePool.h"

#include <VecGeom/navigation/NavStatePool.h>

using vecgeom::cxx::NavStatePool;

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Construct with sizes, allocating on GPU.
 */
VGNavStatePool::VGNavStatePool(int size, int depth) {
  pool_ = new NavStatePool(size, depth);
  static_cast<NavStatePool*>(pool_)->CopyToGpu();
}

//---------------------------------------------------------------------------//
/*!
 * Default destructor in .cc file for PIMPL
 */
VGNavStatePool::~VGNavStatePool() { delete static_cast<NavStatePool*>(pool_); }

//---------------------------------------------------------------------------//
/*!
 * Get allocated GPU state pointer
 */
void* VGNavStatePool::DevicePointer() const {
  assert(pool_);
  void* ptr = static_cast<NavStatePool*>(pool_)->GetGPUPointer();
  assert(ptr);
  return ptr;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
