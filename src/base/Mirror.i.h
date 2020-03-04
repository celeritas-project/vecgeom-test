//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Mirror.i.h
//---------------------------------------------------------------------------//

#include <cassert>

#include <cuda_runtime_api.h>

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Construct by capturing a host object.
 */
template <class T>
Mirror<T>::Mirror() : host_(std::move(host_obj)) {
  static_assert(CELERITAS_IS_TRIVIALLY_COPYABLE(DeviceType),
                "DeviceType is not trivially copyable");
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from host to device memory.
 *
 * If the persistent device pointer has been created, this will update its
 * contents.
 */
template <class T>
void Mirror<T>::HostToDevice() {
  host_.HostToDevice(device_storage_);
  if (persistent_device_ptr_) {
    this->UpdateDevicePtr();
  }
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from device to host memory.
 */
template <class T>
void Mirror<T>::DeviceToHost() {
  host_.DeviceToHost(&device_storage_);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Access a persistent device pointer (it must first be created)
 */
template <class T>
const auto* Mirror<T>::DevicePointer() const -> const DeviceType* {
  assert(persistent_device_ptr_);
  return persistent_device_ptr_.get();
}

//---------------------------------------------------------------------------//
/*!
 * Access a persistent device pointer, creating if needed.
 */
template <class T>
auto Mirror<T>::DevicePointer() -> DeviceType* {
  if (!persistent_device_ptr_) {
    persistent_device_ptr_ = DeviceUniqueMalloc<DeviceType>();
  }
  return persistent_device_ptr_.get();
}

//---------------------------------------------------------------------------//
/*!
 * Update the stored persistent device object.
 *
 * This is essentially:
 * \code
 *    *persistent_device_ptr_ = device_storage_.View();
 * \endcode
 */
template <class T>
void Mirror<T>::UpdateDevicePtr() {
  // Get an updated view (on host)
  DeviceType temp = device_storage_.View();
  // Copy view to the persistent pointer
  cudaError_t errcode = cudaMemcpy(persistent_device_ptr_.get(), &temp,
                                   sizeof(T), cudaMemcpyHostToDevice);
  assert(errcode == cudaSuccess);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
