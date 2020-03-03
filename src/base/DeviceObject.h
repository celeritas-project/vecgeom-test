//---------------------------------*-CUDA-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file DeviceObject.h
//---------------------------------------------------------------------------//
#ifndef base_DeviceObject_h
#define base_DeviceObject_h

#include "Macros.h"

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Trivial container for an on-device object for type-safety and annotation.
 *
 * This class is primarily used by MakeSharedDevicePtr. Because this contains
 * no CUDA code, and shared pointers' deleters are opaque, it's possible to
 * carry around a \c shared_ptr<DeviceObject<T>> in host-only (no NVCC needed)
 * classes.
 *
 * The \c DeviceObject object can be allocated and freed in place of the
 * pointed-to-object, since \code &dobj == &dobj.object \endcode. Its contents
 * can be `cudaMemcpy`d into since it's trivially copyable.
 */
template <class T>
struct DeviceObject {
  static_assert(CELERITAS_IS_TRIVIALLY_COPYABLE(T),
                "T is not trivially copyable");

  T object;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // base_DeviceObject_h
