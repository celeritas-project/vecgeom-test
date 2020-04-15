//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file vgthelper.cu
//---------------------------------------------------------------------------//
#include <cassert>
#include <iostream>

#include "base/KernelParamCalculator.cuh"
#include "base/Types.h"
#include "geom/VGGeometryAdapter.cuh"
#include "geom/VGStateContainer.cuh"
#include "vgthelper.h"

using std::cout;
using std::endl;

namespace celeritas {
//---------------------------------------------------------------------------//
// CUDA FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Transport a particle on a single thread.
 */
__device__ void transport_one(const VGGeometry& geo, VGStateRef state) {
  Real3 pos{-100, 0, 0};
  Real3 dir{1, 0, 0};
  InitialStateRef primary({&pos, &dir});
  // geo.Construct(state, primary);
  while (geo.IsInside(state)) {
    // geo.FindNextStep(state);
    // tal.CellTally(geo.Id(state), geo.NextStep(state));
    // geo.MoveNextStep(state);
  }
}

//---------------------------------------------------------------------------//
/*!
 * Kernel for transporting a batch of particles.
 */
__global__ void transport_all(VGGeometry geometry, VGStateView states) {
  auto thread_id = KernelParamCalculator::ThreadId();
  if (thread_id >= states.size()) return;

  transport_one(geometry, states[thread_id]);
}

//---------------------------------------------------------------------------//

__host__ void RunTransportCuda(std::shared_ptr<celeritas::VGGeometryHost> geo,
                               int ntracks) {
  VGGeometryAdapter adapter(geo);
  adapter.HostToDevice();

  VGStateContainer states(ntracks, *geo);

  cout << "::: Launching transport kernel" << endl;
  KernelParamCalculator calc_kernel_params;
  auto launch_params = calc_kernel_params(states.size());
  transport_all<<<launch_params.grid_size, launch_params.block_size>>>(
      adapter.DeviceView(), states.DeviceView());
  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    cout << "!!! Error " << result << ": " << cudaGetErrorString(result)
         << endl;
    assert(result == cudaSuccess);
  } else {
    cout << ">>> Ran transport kernel successfully!" << endl;
  }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
