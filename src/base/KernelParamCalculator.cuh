//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file KernelParamCalculator.cuh
//---------------------------------------------------------------------------//
#ifndef base_KernelParamCalculator_cuh
#define base_KernelParamCalculator_cuh

#include <cstdint>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Kernel management helper functions.
 *
 * We assume that all our kernel launches use 1-D thread indexing to make
 * things easy.
 *
 * \code
  KernelParamCalculator ...;
   \endcode
 */
class KernelParamCalculator {
 public:
  //@{
  //! Type aliases
  using dim_type = unsigned int;
  //@}

  struct LaunchParams {
    dim3 grid_size;   //!< Number of blocks for kernel grid
    dim3 block_size;  //!< Number of threads per block
  };

 public:
  // Construct with defaults
  explicit __host__ KernelParamCalculator(dim_type block_size = 256);

  // Get launch parameters
  LaunchParams operator()(std::size_t min_num_threads) const;

  // Get the thread ID
  __device__ inline static dim_type ThreadId();

 private:
  //! Default threads per block
  dim_type block_size_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "KernelParamCalculator.i.cuh"

#endif  // base_KernelParamCalculator_cuh
