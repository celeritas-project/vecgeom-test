//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file array.i.h
//---------------------------------------------------------------------------//

#include "Macros.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Test equality of two arrays.
 */
template <typename T, std::size_t N>
CELERITAS_HOST_DEVICE bool operator==(const array<T, N>& lhs,
                                      const array<T, N>& rhs) {
  for (std::size_t i = 0; i != N; ++i) {
    if (lhs[i] != rhs[i]) return false;
  }
  return true;
}

//---------------------------------------------------------------------------//
/*!
 * Test inequality of two arrays.
 */
template <typename T, std::size_t N>
CELERITAS_HOST_DEVICE bool operator!=(const array<T, N>& lhs,
                                      const array<T, N>& rhs) {
  return !(lhs == rhs);
}

//---------------------------------------------------------------------------//
/*!
 * Increment a vector by another vector multiplied by a scalar.
 */
template <typename T, std::size_t N>
CELERITAS_HOST_DEVICE void axpy(T a, const array<T, N>& x, array<T, N>* y) {
  for (std::size_t i = 0; i != N; ++i) {
    (*y)[i] = a * x[i] + (*y)[i];
  }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
