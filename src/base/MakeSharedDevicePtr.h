//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MakeSharedDevicePtr.h
//---------------------------------------------------------------------------//
#ifndef base_MakeSharedDevicePtr_h
#define base_MakeSharedDevicePtr_h

#include <cassert>
#include <memory>

#include <cuda_runtime.h>

#include "DeviceObject.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Construct an on-device pointer wrapped by a shared pointer.
 *
 * The destructor of the shared pointer calls cudaFree when its scope ends.
 */
template <class T, class... Args>
std::shared_ptr<DeviceObject<T>> MakeSharedDevicePtr(Args&&... args) {
  static_assert(CELERITAS_IS_TRIVIALLY_COPYABLE(T),
                "T is not trivially copyable");
  // Construct object on host
  T temp(std::forward<Args>(args)...);

  // Allocate and copy data
  void* devptr = nullptr;
  cudaError_t errcode;
  errcode = cudaMalloc(&devptr, sizeof(T));
  assert(errcode == cudaSuccess);
  errcode = cudaMemcpy(devptr, &temp, sizeof(T), cudaMemcpyHostToDevice);
  assert(errcode == cudaSuccess);

  // Resulting pointer to on-device object
  return std::shared_ptr<DeviceObject<T>>(static_cast<DeviceObject<T>*>(devptr),
                                          cudaFree);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // base_MakeSharedDevicePtr_h
