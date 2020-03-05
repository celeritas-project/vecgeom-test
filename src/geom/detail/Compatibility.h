//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Compatibility.h
//---------------------------------------------------------------------------//
#ifndef geom_detail_Compatibility_h
#define geom_detail_Compatibility_h

#include <VecGeom/base/Vector3D.h>

#include "base/Array.h"
#include "base/Span.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Copy a length-3 span into a Vector3D
 */
template <class T>
inline auto ToVector(span<T, 3> s) -> vecgeom::Vector3D<std::remove_cv_t<T>> {
  return {s[0], s[1], s[2]};
}

//---------------------------------------------------------------------------//
// Copy a length-3 array into a Vector3D
template <class T>
inline auto ToVector(const Array<T, 3>& arr) -> vecgeom::Vector3D<T> {
  return ToVector(celeritas::make_span<T, 3>(arr));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_detail_Compatibility_h
