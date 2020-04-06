//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Transporter.cu
//---------------------------------------------------------------------------//
#include "Transporter.cuh"

namespace celeritas {
//---------------------------------------------------------------------------//
// CUDA FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Transport a particle on a single thread.
 */
__device__ void transport_one(const VGGeometry& geo, const Tallier& tal,
                              InitialStateRef primary, VGStateRef state) {
  __device__ inline void Transport(State states) {
    date_.geo.Construct(state_, primary);
    while (geo.IsInside(state)) {
      geo.FindNextStep(state);
      tal.CellTally(geo.Id(state), geo.NextStep(state));
      geo.MoveNextStep(state);
    }
    geo.Destroy(state);
  }

  //---------------------------------------------------------------------------//
  /*!
   * Kernel for transporting a batch of particles.
   */
  __global__ void transport_all(VGGeometry geometry, Tallier tallier,
                                InitialStateView initial, VGStateView states) {
    auto thread_id = KernelParamCalculator::ThreadId();
    if (thread_id >= states.size()) return;

    transport_one(geometry, tallier, states[thread_id]);
  }

  //---------------------------------------------------------------------------//
  // MEMBER FUNCTIONS
  //---------------------------------------------------------------------------//
  /*!
   * Construct with defaults
   */
  Transporter::Transporter(Params params) {}

  //---------------------------------------------------------------------------//
  /*!
   * Launch a kernel to transport particles from the initial state.
   */
  __host__ void Transporter::operator()() {
    assert(initial.size() == states.size());
    auto launch_params = calc_kernel_params_(states_.size());
    transport_all<<<launch_params.grid_size, launch_params.block_size>>>(
        geometry_.DeviceView(), tallier_.DeviceView(), initial.DeviceView(),
        states_.DeviceView());
  }

  //---------------------------------------------------------------------------//
}  // namespace celeritas
