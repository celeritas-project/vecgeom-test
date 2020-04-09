//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geom/VGGeometryHost.h
//---------------------------------------------------------------------------//
#ifndef geom_VGGeometryHost_h
#define geom_VGGeometryHost_h

#include <string>

#include "base/OpaqueId.h"

namespace celeritas {

//---------------------------------------------------------------------------//
/*!
 * Wrap a VecGeom geometry definition with convenience functions.
 */
class VGGeometryHost {
 public:
  //@{
  //! Type aliases
  using IdType = OpaqueId<VGGeometryHost, unsigned int>;
  //@}

 public:
  // Construct from a GDML filename
  explicit VGGeometryHost(const char* gdml_filename);

  // >>> ACCESSORS

  // Get the label for a placed volume ID
  const std::string& IdToLabel(IdType vol_id) const;
  // Get the ID corresponding to a label
  IdType LabelToId(const std::string& label) const;

  //! Maximum nested geometry depth
  int MaxDepth() const { return max_depth_; }

  //! Off-the-end ID
  IdType EndId() const { return IdType{max_id_}; }

 private:
  int max_depth_;
  unsigned int max_id_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGGeometryHost_h
