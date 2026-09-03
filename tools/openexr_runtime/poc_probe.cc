#include <cstddef>
#include <cstdint>

#include <ImfCheckFile.h>
#include <ImfNamespace.h>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const char* bytes = reinterpret_cast<const char*>(data);
  (void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, false);
  (void)OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, true);
  return 0;
}
