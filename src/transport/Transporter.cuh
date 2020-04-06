//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Transporter.cuh
//---------------------------------------------------------------------------//
#ifndef transport_Transporter_cuh
#define transport_Transporter_cuh

#include "Tallier.cuh"
#include "geom/VGGeometry.cuh"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * \code
  Transporter ...;
   \endcode
 */
class Transporter {
 public:
  //@{
  //! Type aliases
  //@}

  struct Params {
    VGGeometry geometry;
    Tallier tallier;
  };

 public:
  explicit __device__ Transporter(const Params& params, VGStateRef state)
      : data_(params), state_(state) {}

  // Initialize
  __device__ inline void Construct(const PrimaryRef& primary);

  // Track and tally
  __device__ inline void Transport();

  // Clean up
  __device__ inline void Destroy();

 private:
  const Params& data_;
  VGStateRef state_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "Transporter.i.cuh"

#endif  // transport_Transporter_cuh
