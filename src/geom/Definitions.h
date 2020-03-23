//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Definitions.h
//---------------------------------------------------------------------------//
#ifndef geom_Definitions_h
#define geom_Definitions_h

#include "base/OpaqueId.h"

namespace celeritas {
//---------------------------------------------------------------------------//
//@{
//! Instantiators for geometry IDs
struct GeometryCell {};
struct GeometryUniqueCell {};
//@}

using CellId = OpaqueId<GeometryCell>;
using UniqueCellId = OpaqueId<GeometryUniqueCell>;

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif // geom_Definitions_h
