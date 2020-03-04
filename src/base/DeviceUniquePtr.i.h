//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceUniquePtr.i.h
//---------------------------------------------------------------------------//

#include <cassert>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Allocate uninitialized device memory managed by unique_ptr.
 */
template <class T>
DeviceUniquePtr<T> DeviceUniqueMalloc() {
  void* devptr = nullptr;
  cudaError_t errcode = cudaMalloc(&devptr, sizeof(T));
  assert(errcode == cudaSuccess);
  assert(devptr);
  return DeviceUniquePtr<T>(devptr);
}

//---------------------------------------------------------------------------//
/*!
 * Create a unique pointer by forwarding arguments to the host constructor.
 */
template <class T, class... Args>
DeviceUniquePtr<T> MakeDeviceUnique(Args&&... args) {
  static_assert(CELERITAS_IS_TRIVIALLY_COPYABLE(T),
                "T is not trivially copyable");
  // Construct temporary object on host
  T temp(std::forward<Args>(args)...);

  // Create unique pointer
  DeviceUniquePtr<T> result = DeviceUniqueMalloc<T>();

  // Copy data
  cudaError_t errcode =
      cudaMemcpy(result.get(), &temp, sizeof(T), cudaMemcpyHostToDevice);
  assert(errcode == cudaSuccess);
  return result;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
