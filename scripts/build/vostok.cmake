# Set rpath based on environment (loaded Spack modules); VecGeom does not
# correctly set rpath for downstream use
string(REPLACE ":" ";" _rpath "$ENV{DYLD_FALLBACK_LIBRARY_PATH}")
set(CMAKE_BUILD_RPATH "${_rpath}" CACHE STRING "")
set(CMAKE_INSTALL_RPATH "${_rpath};$ENV{prefix_dir}/lib" CACHE STRING "")

# Enable color diagnostics when using Ninja
foreach(LANG C CXX Fortran)
  set(CMAKE_${LANG}_FLAGS "${CMAKE_${LANG}_FLAGS} -fcolor-diagnostics"
      CACHE STRING "" FORCE)
endforeach()
