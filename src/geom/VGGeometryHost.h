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
#include "Definitions.h"

namespace celeritas {
class RootModel;

//---------------------------------------------------------------------------//
/*!
 * Wrap a VecGeom geometry definition with convenience functions.
 */
class VGGeometryHost {
 public:
  //@{
  //! Type aliases
  using SpanInt = span<int>;
  //@}

 public:
  // Construct from a ROOT model
  explicit VGGeometryHost(const RootModel& model);

  // >>> ACCESSORS

  // Get the label for a placed volume ID
  const std::string& IdToLabel(UniqueCellId vol_id) const;
  // Get the ID corresponding to a label
  UniqueCellId LabelToId(const std::string& label) const;

  // Maximum nested geometry depth
  int MaxDepth() const;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGGeometryHost_h
