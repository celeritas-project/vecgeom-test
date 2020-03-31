//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelParamCalculator.cu
//---------------------------------------------------------------------------//
#include <cassert>
#include <limits>

#include "KernelParamCalculator.cuh"

namespace {
// Integer division, rounding up, for positive numbers
template <class UInt>
UInt ceil_div(UInt top, UInt bottom) {
  return (top / bottom) + (top % bottom != 0);
}
}  // namespace

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * \brief Initialize with the number of threads per block
 */
__host__ KernelParamCalculator::KernelParamCalculator(dim_type size)
    : block_size_(size) {
  assert(size >= 64);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Calculate launch params given the number of threads
 */
__host__ KernelParamCalculator::LaunchParams KernelParamCalculator::operator()(
    std::size_t min_num_threads) const {
  assert(min_num_threads > 0);
  assert(min_num_threads <=
         static_cast<std::size_t>(std::numeric_limits<dim_type>::max()));

  // Ceiling integer division
  dim_type grid_size = ceil_div<dim_type>(min_num_threads, this->block_size_);

  LaunchParams result;
  result.grid_size.x = grid_size;
  result.block_size.x = this->block_size_;
  assert(result.grid_size.x * result.block_size.x >= min_num_threads);
  return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
