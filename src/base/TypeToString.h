//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// TypeToString utility functions.
//---------------------------------------------------------------------------//

#ifndef celeritas_TypeToString_h
#define celeritas_TypeToString_h

#include <string>
#include <type_traits>

namespace celeritas {
namespace internal {
std::string DemangleTypeidName(const char* typeid_name);
}

//---------------------------------------------------------------------------//
/*!
 * \fn TypeToString
 *
 * Get the pretty name of a type. If given an instance, RTTI will be used to
 * determine the instance's true type.
 *
 * See:
 * http://stackoverflow.com/questions/281818/unmangling-the-result-of-stdtype-infoname
 */
template <class T>
inline std::string TypeToString() {
  return internal::DemangleTypeidName(typeid(T).name());
}

template <class T>
inline std::string TypeToString(const T& t) {
  return internal::DemangleTypeidName(typeid(t).name());
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

//---------------------------------------------------------------------------//
#endif  // celeritas_Range_h
