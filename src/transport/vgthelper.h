//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file vgthelper.h
//---------------------------------------------------------------------------//
#ifndef app_vgthelper_h
#define app_vgthelper_h

#include <memory>

#include "geom/VGGeometryHost.h"

namespace celeritas {
//---------------------------------------------------------------------------//
void RunTransport(std::shared_ptr<celeritas::VGGeometryHost> geo, int ntracks);

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // app_vgthelper_h
