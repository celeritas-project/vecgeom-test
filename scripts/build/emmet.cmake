# Set rpath based on environment (loaded Spack modules); VecGeom does not
# correctly set rpath for downstream use
string(REPLACE ":" ";" _rpath "$ENV{LD_RUN_PATH}")
set(CMAKE_BUILD_RPATH "${_rpath}" CACHE STRING "")
set(CMAKE_INSTALL_RPATH "$ENV{prefix_dir}/lib;${_rpath}" CACHE STRING "")

# Export compile commands for microsoft visual code
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "")

# Use CUDA
set(VGT_USE_CUDA ON CACHE BOOL "")

# Enable color diagnostics when using Ninja
foreach(LANG C CXX)
  set(CMAKE_${LANG}_FLAGS "${CMAKE_${LANG}_FLAGS} -fdiagnostics-color=always"
      CACHE STRING "" FORCE)
endforeach()
