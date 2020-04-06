//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceMirror.h
//---------------------------------------------------------------------------//
#ifndef base_DeviceMirror_h
#define base_DeviceMirror_h

#include <memory>

#include "DeviceUniquePtr.h"
#include "Macros.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Store "equivalent" host and device objects and facilitate memory transfer.
 *
 * This class is best used inside a \c shared_ptr for persistent storage of
 * complex datatypes. It guarantees persistence of associated host and device
 * storage and enforces an interface for their transfer.
 *
 * Its design is based on these assumptions:
 * - The host class's design is upstream of the CUDA kernel
 * - The context in which the host class is used cannot depend on any CUDA code
 * - Persistent views of the device class corresponding to the host class are
 *   not always needed
 * - The on-device object should not require knowledge of the host class.
 *
 * The DeviceMirror requires a shared pointer to a host object and allocates
 * (and owns) an AdapterType using the host object. Calling \c HostToDevice on
 * the mirror will dispatch to the \c AdapterType's method of the same name.
 * Likewise with \c DeviceToHost .
 *
 * Additionally, the \c DeviceMirror enables concurrent management of a
 * persistent device object that can be referenced by other device objects as
 * needed. The first call to the non-const \c DevicePointer() method will
 * allocate a \c DeviceType on the device and return the corresponding device
 * pointer. This allows a network of AdapterType objects with \c DeviceMirror
 * member data to build complex objects.
 *
 * The Host type and its associated device storage class must have the
 * following interface: \code
    class AdapterType {
     public:
      using HostType = ...;
      using DeviceType = ...;

      explicit AdapterType(const HostType& host);
      void HostToDevice(const HostType& host_cls);
      void DeviceToHost(HostType* host_cls) const;
      DeviceType DeviceView() const;
     private:
      // Device data, e.g. thrust device vectors
    };
\endcode
 *
 * The final restriction is that \c DeviceType must be trivially copyable.
 */
template <class AdapterT>
class DeviceMirror {
 public:
  //@{
  //! Type aliases
  using AdapterType = AdapterT;
  using HostType = typename AdapterT::HostType;
  using SPHostType = std::shared_ptr<HostType>;
  using DeviceType = typename AdapterT::DeviceType;
  //@}

 public:
  // Capture a shared pointer to host data and allocate device data.
  explicit inline DeviceMirror(SPHostType host);

  // Copy data from host to device memory
  inline void HostToDevice();

  // Copy data from device to host memory
  inline void DeviceToHost();

  //@{
  //! Host accessors
  HostType& Host() { return *host_; }
  const HostType& Host() const { return *host_; }
  AdapterType& Adapter() { return adapter_; }
  const AdapterType& Adapter() const { return adapter_; }
  //@}

  //! Host-callable accessor to device-resident memory
  DeviceType DeviceView() const { return adapter_.DeviceView(); }

  // Access a persistent device pointer if created
  inline const DeviceType* DevicePointer() const;

  // Access a persistent device pointer, creating if needed
  inline DeviceType* DevicePointer();

 private:
  // Replace the allocated device object
  inline void UpdateDevicePtr();

 private:
  SPHostType host_;
  AdapterType adapter_;
  DeviceUniquePtr<DeviceType> persistent_device_ptr_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "DeviceMirror.i.h"

#endif  // base_DeviceMirror_h
