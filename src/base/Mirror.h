//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Mirror.h
//---------------------------------------------------------------------------//
#ifndef base_Mirror_h
#define base_Mirror_h

#include <utility>

#include "DeviceUniquePtr.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Store "equivalent" host and device objects and facilitate memory transfer.
 *
 * This class is best used inside a \c shared_ptr for persistent storage of
 * complex datatypes. It guarantees persistence of associated host and device
 * storage and enforces an interface for their transfer.
 *
 * The constructor *forward* to the host type, which is constructed in-place.
 * It creates an associated \c DeviceStorage object with the
 * default constructor. Calling \c HostToDevice on the mirror will dispatch to
 * the \c HostType's method of the same name. Likewise with \c DeviceToHost .
 *
 * Additionally, the \c Mirror enables concurrent management of a persistent
 * device object that can be referenced by other device objects as needed. The
 * first call to the non-const \c DevicePointer() method will allocate a \c
 * DeviceType on the device and return the corresponding device pointer. This
 * allows a network of Host objects with \c Mirror member data to build complex
 * objects.
 *
 * The Host type and its associated device storage class must have the
 * following interface: \code
    struct HostType {
      using DeviceStorageType = ...;
      using DeviceType = ...;
      HostType(...);
      void HostToDevice(DeviceStorageType* device_storage) const;
      void DeviceToHost(const DeviceStorageType& device_storage);
    };

    struct DeviceStorage {
      DeviceStorage();
      DeviceType View() const;
    };
\endcode
 *
 * Additionaly, \c DeviceType *must* be trivially copyable.
 */
template <class HostT>
class Mirror {
 public:
  //@{
  //! Type aliases
  using HostType = typename HostT;
  using DeviceStorageType = typename HostT::DeviceStorageType;
  using DeviceType = typename HostT::DeviceType;
  //@}

 public:
  //! Construct the host data in-place.
  template <class... Args>
  explicit inline Mirror(Args&&... host_args)
      : host_(std::forward<Args>(host_args)...) {}

  // Copy data from host to device memory
  inline void HostToDevice();

  // Copy data from device to host memory
  inline void DeviceToHost();

  //@{
  //! Host accessors
  HostType& Host() { return host_; }
  const HostType& Host() const { return host_; }
  DeviceStorageType& DeviceStorage() { return device_; }
  const DeviceStorageType& DeviceStorage() const { return device_; }
  //@}

  //! Host-callable accessor to device-resident memory
  DeviceType View() const { return device_.View(); }

  // Access a persistent device pointer if created
  inline const DeviceType* DevicePointer() const;

  // Access a persistent device pointer, creating if needed
  inline DeviceType* DevicePointer();

 private:
  // Replace the allocated device object
  inline void UpdateDevicePtr();

 private:
  HostType host_;
  DeviceStorageType device_storage_;
  DeviceUniquePtr<DeviceType> persistent_device_ptr_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#include "Mirror.i.h"

#endif  // base_Mirror_h
