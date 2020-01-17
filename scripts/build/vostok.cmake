string(REPLACE ":" ";" _rpath "$ENV{DYLD_FALLBACK_LIBRARY_PATH}")
set(CMAKE_BUILD_RPATH "${_rpath}" CACHE STRING "")
set(CMAKE_INSTALL_RPATH "${_rpath};$ENV{prefix_dir}/lib" CACHE STRING "")
