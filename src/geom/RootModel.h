//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file RootModel.h
//---------------------------------------------------------------------------//
#ifndef geom_RootModel_h
#define geom_RootModel_h

#include <iosfwd>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Access user-defined geometry, materials, etc through ROOT.
 */
class RootModel {
 public:
  //@{
  //! Type aliases
  //@}

 public:
  // Construct from filename
  explicit RootModel(const char* root_file);

  // Print volume names
  void PrintGeometry(std::ostream& os) const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_RootModel_h
