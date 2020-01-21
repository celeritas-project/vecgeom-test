//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// Example app for navigating with VecGeom.
//---------------------------------------------------------------------------//

#include "TypeToString.h"

#ifdef __GNUG__
#include <cstdlib>
#include <memory>

#include <cxxabi.h>
#endif  // __GNUG__

namespace celeritas {
namespace internal {
//---------------------------------------------------------------------------//
std::string DemangleTypeidName(const char* typeid_name) {
#ifdef __GNUG__
  int status = -1;
  // Return a null-terminated string allocated with malloc
  char* demangled = abi::__cxa_demangle(typeid_name, NULL, NULL, &status);

  // Copy result, falling back to mangled name if unsuccessful
  std::string result(status == 0 ? demangled : typeid_name);

  // Free the returned memory
  free(demangled);
#else   // __GNUG__
  std::string result(typeid_name);
#endif  // __GNUG__

  return result;
}
}  // namespace internal

//---------------------------------------------------------------------------//
}  // namespace celeritas
