//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file InitialStateRef.cuh
//---------------------------------------------------------------------------//
#ifndef geom_InitialStateRef_cuh
#define geom_InitialStateRef_cuh

#include "base/Types.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * Optional detailed class description, and possibly example usage:
 * \code
  InitialStateRef ...;
   \endcode
 */
class InitialStateRef {
 public:
  //! Construction parameters
  struct Params {
    Real3* pos;
    Real3* dir;
  };

 public:
  explicit __device__ InitialStateRef(const Params& params) : data_(params) {}
  //@{
  //! Accessors
  __device__ const Real3& pos() const { return *data_.pos; }
  __device__ const Real3& dir() const { return *data_.dir; }
  //@}

 private:
  const Params data_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_InitialStateRef_cuh
