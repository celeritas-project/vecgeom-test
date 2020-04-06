//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file vgthelper.cu
//---------------------------------------------------------------------------//

#include "vgthelper.h"

namespace celeritas {
//---------------------------------------------------------------------------//
void RunTransport(std::shared_ptr<celeritas::VGGeometryHost> geo, int ntracks) {
  return RunTransportCuda(geo, ntracks);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
