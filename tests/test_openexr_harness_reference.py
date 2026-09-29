"""Loading and parsing the pinned reference OpenEXR harness."""

from pathlib import Path

from dllm_agent.harness_synthesis.regions import statement_regions
from dllm_agent.targets.openexr.harness_reference import load_reference_harness

_HARNESS = """\
#include <ImfCheckFile.h>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return 0;
}
"""


def test_load_reference_harness_reads_the_source_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "poc_probe.cc"
    _ = path.write_text(_HARNESS)

    harness = load_reference_harness(path)

    assert harness.source == _HARNESS


def test_load_reference_harness_detects_the_same_regions_as_statement_regions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "poc_probe.cc"
    _ = path.write_text(_HARNESS)

    harness = load_reference_harness(path)

    assert harness.regions == statement_regions(_HARNESS)
    assert harness.regions[0].text == "#include <ImfCheckFile.h>"
