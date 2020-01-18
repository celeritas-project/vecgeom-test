//---------------------------------*-C++-*-----------------------------------//
// Copyright 2020 UT-Battelle, LLC and other VecGeomTest Developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
/// \file
/// Example app for navigating with VecGeom.
//---------------------------------------------------------------------------//

#include "ScopeRootMessages.h"

#include <iostream>

#include <TError.h>

namespace {
//---------------------------------------------------------------------------//
void HandleRootError(int level, bool abort, const char *location,
                     const char *msg) {
  if (level < kInfo) return;

  std::cerr << "\e[2;37m[" << location << "]\e[0m " << msg << '\n';
}

//---------------------------------------------------------------------------//
}  // namespace

namespace celeritas {
//---------------------------------------------------------------------------//
ScopeRootMessages::ScopeRootMessages() {
  original_handler_ = SetErrorHandler(&HandleRootError);
}

//---------------------------------------------------------------------------//
/*!
 */
ScopeRootMessages::~ScopeRootMessages() { SetErrorHandler(original_handler_); }

//---------------------------------------------------------------------------//
}  // namespace celeritas
