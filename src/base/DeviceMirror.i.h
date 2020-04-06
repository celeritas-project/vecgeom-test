//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceMirror.i.h
//---------------------------------------------------------------------------//
#ifndef base_DeviceMirror_i_h
#define base_DeviceMirror_i_h

#include <cassert>
#include <utility>

#include <cuda_runtime_api.h>

#include "DeviceUniquePtr.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Capture a shared pointer to host data and allocate device data.
 */
template <class T>
DeviceMirror<T>::DeviceMirror(SPHostType host)
    : host_(std::move(host)), adapter(*host_) {}

//---------------------------------------------------------------------------//
/*!
 * Copy data from host to device memory.
 *
 * If the persistent device pointer has been created, this will update its
 * contents.
 */
template <class T>
void DeviceMirror<T>::HostToDevice() {
  adapter_.HostToDevice(*host_);
  if (persistent_device_ptr_) {
    this->UpdateDevicePtr();
  }
}

//---------------------------------------------------------------------------//
/*!
 * Copy data from device to host memory.
 */
template <class T>
void DeviceMirror<T>::DeviceToHost() {
  adapter_.DeviceToHost(host_.get());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Access a persistent device pointer (it must first be created)
 */
template <class T>
auto DeviceMirror<T>::DevicePointer() const -> const DeviceType* {
  assert(persistent_device_ptr_);
  return persistent_device_ptr_.get();
}

//---------------------------------------------------------------------------//
/*!
 * Access a persistent device pointer, creating if needed.
 */
template <class T>
auto DeviceMirror<T>::DevicePointer() -> DeviceType* {
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
 *    *persistent_device_ptr_ = device_storage_.DeviceView();
 * \endcode
 */
template <class T>
void DeviceMirror<T>::UpdateDevicePtr() {
  // Get an updated view (on host)
  DeviceType temp = this->DeviceView();
  // Copy view to the persistent pointer
  cudaError_t errcode = cudaMemcpy(persistent_device_ptr_.get(), &temp,
                                   sizeof(T), cudaMemcpyHostToDevice);
  assert(errcode == cudaSuccess);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
