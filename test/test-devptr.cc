#include "base/DeviceUniquePtr.h"

struct MyData {
  int i;
  int j;
};

int main(int argc, char** argv) {
  auto myptr = celeritas::MakeDeviceUnique<MyData>(MyData{10, 20});
  return 0;
}
