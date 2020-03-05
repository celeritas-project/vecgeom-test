//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGStateContainer.i.cuh
//---------------------------------------------------------------------------//

#include <thrust/memory.h>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Emit a device view.
 */
VGStateView VGStateContainer::View() {
  using thrust::raw_pointer_cast;

  VGStateView::Params params;
  params.size = size_;
  params.vgmaxdepth = vgmaxdepth_;
  params.volume_handle = raw_pointer_cast(volume_ptrs_.data());
  params.vgstate = vgstate_.GetGPUPointer();
  params.vgnext = vgnext_.GetGPUPointer();
  params.pos = raw_pointer_cast(pos_.data());
  params.dir = raw_pointer_cast(dir_.data());
  params.next_step = raw_pointer_cast(next_step_.data());

  return VGStateView(params);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
