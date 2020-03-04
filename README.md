# Description

This is a very simple test of using VecGeom to raytrace across a geometry, with
prototyping of CUDA-friendly coding techniques based on the work done in
CUDA-enabled Shift.

# File names

All `__device__` and `__global__` code must be compiled with NVCC to generate
device objects. However, code that merely uses CUDA API calls such as
`cudaMalloc` does *not* have to be compiled with NVCC. Instead, it only has to
be linked against the CUDA runtime library and include `cuda_runtime_api.h`.

Since NVCC is slower and other compilers' warning/error output is more
readable, it's preferable to use NVCC for as little compilation as possible.
Furthermore, not requiring NVCC lets us play nicer with downstream libraries
and front-end apps.

Finally, it provides a check against surprise kernel launches. For example,
trying to compile
```
thrust::device_vector<double> dv(10);
```
actually launches a kernel to fill the vector's initial state. The code will
not compile in a `.cc` file run through the host compiler, but it will silently
generate kernel code when run through NVCC.

Thus we 

- `.cuh` is for header files that contain device code (require NVCC)
- `.cu` is for `__global__` kernels and functions that launch them
- `.h` is for code compatible with host compilers, with no decoration. It may
  include thrust objects.

# On-device memory management

The key features of CUDA memory management are:
1. Allocating device memory for structures needed in kernels,
2. Transforming data *on the host* to device-friendly types and layout,
3. Transferring objects from host to device,
4. Providing kernel-friendly accessors to this device-memory data,
5. Reversing the process as needed for "output" data (transferring back to
   host, un-flattening, etc.)

## Large struct-of-array storage

Currently there is no template interface enforcement of array-like storage,
just a pattern defined here.

### Container

Like the `State_Vector` in Shift's `sce_cuda` or `Particle_Vector` in
`shift_cuda`, it's often necessary to allocate a struct of arrays (SOA) or
array of structs (AOS) whose size is that of the level of parallelism in the
problem, e.g. the number of particles being transported. The objects that
manage (allocate and deallocate) such data in Celeritas are *containers*.

### View

A view is a pointer (or pointers) plus other metadata (size) to the data owned
by a container. It contains only primitive datatypes capable of being passed
directly to the device via a kernel launch. Views are primarily meant as an
abstraction for simple loading/storing of data.

To simplify construction and management of views, every `View` class should
define a public `Params` struct that comprises the construction arguments for
the view class. These arguments should be pointers, sizes, and/or `Span`
objects. To simplify view construction, the `Params` might be set up to
contain exactly the same data that it needs to store, using implicit copy
construction rather than risk missing a parameter in an explicit constructor.

Views should be `__host__`-decorated return-by-value objects from a container
function canonically named `View`. (Perhaps change to `DeviceView` to make it
really obvious that it's using device data? No other class should have a `View`
to host memory, though.)

Like standard-library containers, views should have a `size()` method and an
`operator[]` method. Since a `View` is generally more complex than a `Span`
(which has a single pointer and a size), the bracket operator will return by
value a reference *object*.

### Ref

A *reference* (or `Ref` as an abbreviation) is a "safe" pointer to a single
element of a view. This may represent, for example, a single (thread-local)
particle. It usually has more than one piece of data, and the reference object
can be constructed to "protect" the pointed-to data (for example by validating
its contents).

## Complex object storage

### On-device unique pointers

Persistent on-device memory management of complex objects can be achieved
trivially with a `std::unique_ptr` that uses a custom deleter. The
`DeviceUniquePtr.h` defines a type alias
```c++
template<class T> using DeviceUniquePtr = std::unique_ptr<T, CudaDeleter>;
```
as well as helper functions for allocating and/or constructing such pointers.

In general, these should be used only for single "persistent" objects as
opposed to arrays of data and/or objects that are frequently replaced.

### Host↔︎device mirroring

The `Mirror` templated object and the standard library's memory management
classes allow deep hierarchies of host and device objects to be represented and
their contents transferred. 

## Comparison to Shift

A `std::shared_ptr<celeritas::Mirror>` is a direct replacement for 
`shift::Shared_Device_Ptr` (in the single-instance constructor case). The
primary difference is that Celeritas doesn't require any allocation of the
shallow object on device so that, for example, it can be created on-the-fly to
be passed to kernel invocations as an argument. The `Device_Memory_Manager` is
equivalent to the `DeviceStorageType`.

A Celeritas view differs slightly from a Shift `Device_View`, which is a
combination of a "Container" and a "View" as presented above. Each Shift
`Device_View` contains a host vector of C++ shared pointers (each a
`Device_Memory_Manager`), each of which has allocated and owned some objects on
device. From this vector, which it retains to keep the shared pointers alive
and the device memory allocated, the `Device_View` allocates a device vector
where each element is an on-device object (with only POD data and with all
member functions decorated with `__device__`).  

The `Device_View` provides a `get_view()` accessor that returns a
`Device_View_Field` referencing these on-device objects. Shift's
`Device_View_Field` is exactly equivalent to the Celeritas/C++ `Span`
object.

