#include <cstddef>
#include <fstream>
#include <ios>
#include <vector>

#include <ImfCheckFile.h>
#include <ImfNamespace.h>

namespace {
constexpr std::size_t kMaximumInputBytes = 1'048'576;
constexpr int kUsageError = 2;
constexpr int kOpenError = 3;
constexpr int kSizeError = 4;
constexpr int kReadError = 5;
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return kUsageError;
  }

  std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
  if (!input) {
    return kOpenError;
  }
  const std::streamoff input_size = input.tellg();
  if (input_size < 0 || static_cast<std::size_t>(input_size) > kMaximumInputBytes) {
    return kSizeError;
  }
  input.seekg(0, std::ios::beg);
  std::vector<char> storage(static_cast<std::size_t>(input_size));
  if (!storage.empty() && !input.read(storage.data(), input_size)) {
    return kReadError;
  }

  const char* bytes = storage.data();
  const std::size_t size = storage.size();
  // The pinned implementation returns its `hadfail` flag; its header comment
  // describes the opposite polarity. Match exrcheck's implementation semantics.
  const bool has_error =
      OPENEXR_IMF_NAMESPACE::checkOpenEXRFile(bytes, size, true, true, true);
  return has_error ? 1 : 0;
}
