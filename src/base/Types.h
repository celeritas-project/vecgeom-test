//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Types.h
//---------------------------------------------------------------------------//
#ifndef base_Types_h
#define base_Types_h

namespace celeritas {
//---------------------------------------------------------------------------//
template <typename T, std::size_t N>
class Array;

using ssize_type = int;
using real_type = double;
using RealPointer3 = Array<real_type*, 3>;
using Real3 = Array<real_type, 3>;

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // base_Types_h
