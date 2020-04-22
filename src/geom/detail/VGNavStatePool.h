//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file VGNavStatePool.h
//---------------------------------------------------------------------------//
#ifndef geom_VGNavStatePool_h
#define geom_VGNavStatePool_h

namespace celeritas {
//---------------------------------------------------------------------------//
/*!
 * Implementation detail for creating a navstatepool.
 *
 * Construction of the navstatepool has to be in a host compliation unit due to
 * VecGeom macro magic.
 *
 * Note that NavStatePool can't be forward declared because of inline
 * namespaces, so we can't use unique_ptr.
 */
class VGNavStatePool {
 public:
  // Construct without storage
  VGNavStatePool() = default;

  // Construct with sizes, allocating on GPU
  VGNavStatePool(int size, int depth);

  // Destructor in .cc file frees memory PIMPL
  ~VGNavStatePool();

  // Get allocated GPU state pointer
  void* DevicePointer() const;

 private:
  // vecgeom::cxx::NavStatePool
  void* pool_ = nullptr;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas

#endif  // geom_VGNavStatePool_h
