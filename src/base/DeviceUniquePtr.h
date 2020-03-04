//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceUniquePtr.h
//
//! Memory management for device pointers.
//---------------------------------------------------------------------------//
#ifndef base_DeviceUniquePtr_h
#define base_DeviceUniquePtr_h

#include <memory>

#include <cuda_runtime_api.h>

namespace celeritas {
//---------------------------------------------------------------------------//
//! Functor to free device memory when called.
template <class T>
struct CudaDeleter {
  void operator()(T* ptr) const { cudaFree(ptr); }
};

//! Typename alias for CUDA-owned memory managed by a std::unique_ptr.
template <class T>
using DeviceUniquePtr = std::unique_ptr<T, CudaDeleter>;

//---------------------------------------------------------------------------//
// Allocate uninitialized device memory managed by unique_ptr
template <class T>
inline DeviceUniquePtr<T> DeviceUniqueMalloc();

//---------------------------------------------------------------------------//
// Create a unique pointer by forwarding arguments to the host constructor
template <class T, class... Args>
inline DeviceUniquePtr<T> MakeDeviceUnique(Args&&... args);
//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "DeviceUniquePtr.i.h"

#endif  // base_DeviceUniquePtr_h
