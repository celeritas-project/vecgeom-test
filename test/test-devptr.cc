#include "base/Mirror.h"

struct MyData {
  int i;
  int j;
};

int main(int argc, char** argv) {
  auto myptr = celeritas::MakeSharedDevicePtr<MyData>(MyData{10, 20});
  return 0;
}
