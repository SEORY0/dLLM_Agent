# dLLM-native multi-target 24-hour recovery report

Status: 15/15 finished cells verified against the frozen schedule `6c23b391452ce71ddcf5d4ae707c27beb8e52d3f2e90a7d187c0cf60b17a9a69`.
This is an n=1 descriptive comparison; percentages summarize this run only.
A fatal canary trigger is benchmark evidence, not a vulnerability.
No novel vulnerability claim is made; independent proof is limited to the hashes below.

## Recovered Outcomes

First trigger time (logical search seconds) includes feedback-arm probe time.

| Target | Arm | Final coverage | Features | Executed units | Effective units/s | Unique crash candidates | Distinct triggered canaries | First trigger time | OOM state | Independently proved in this cell |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| libpng | libfuzzer-only | 1413 | 5566 | 574570301 | 6650.119 | 7566 | 3 | 15 | False | 1 |
| libpng | structured-one-shot | 1338 | 4972 | 310583764 | 3594.719 | 2919 | 2 | 15 | False | 0 |
| libpng | feedback-no-remask | 1322 | 4864 | 308771491 | 3573.744 | 2540 | 3 | 975 | False | 0 |
| libpng | selective-revision | 1327 | 4936 | 371794905 | 4303.182 | 2815 | 2 | 975 | False | 0 |
| libpng | pseudo-feedback | 1401 | 5553 | 801722056 | 9279.190 | 5498 | 2 | 980 | False | 0 |
| libtiff | libfuzzer-only | 1564 | 3850 | 18774152 | 217.293 | 28 | 1 | 265 | False | 0 |
| libtiff | structured-one-shot | 2103 | 4850 | 18546409 | 214.658 | 350 | 3 | 55 | True | 0 |
| libtiff | feedback-no-remask | 1998 | 4623 | 11008849 | 127.417 | 215 | 3 | 975 | True | 0 |
| libtiff | selective-revision | 2433 | 5859 | 10141375 | 117.377 | 306 | 3 | 975 | True | 0 |
| libtiff | pseudo-feedback | 1850 | 4638 | 26979761 | 312.266 | 28 | 1 | 1915 | False | 0 |
| libxml2 | libfuzzer-only | 5537 | 26530 | 139532146 | 1614.955 | 0 | 0 | none | False | 0 |
| libxml2 | structured-one-shot | 5697 | 27735 | 106861095 | 1236.818 | 292 | 2 | 185 | False | 6 |
| libxml2 | feedback-no-remask | 3485 | 10025 | 2703680 | 31.293 | 97 | 1 | 975 | False | 0 |
| libxml2 | selective-revision | 5584 | 29177 | 18403609 | 213.005 | 29499 | 2 | 1280 | False | 0 |
| libxml2 | pseudo-feedback | 5196 | 24552 | 26639911 | 308.332 | 5 | 2 | 39830 | False | 0 |

Unique crash candidates: 52158.
Distinct triggered canary IDs: 9 (PNG003, PNG006, PNG007, TIF007, TIF012, TIF014, XML009, XML012, XML017)

## Selective vs Pseudo

Primary selective vs pseudo percent effects:
- libpng: coverage -5.28%, features -11.11%, executed units -53.63%, coverage-per-million-executions +104.25%.
- libtiff: coverage +31.51%, features +26.33%, executed units -62.41%, coverage-per-million-executions +249.87%.
- libxml2: coverage +7.47%, features +18.84%, executed units -30.92%, coverage-per-million-executions +55.56%.
geometric mean direction: selective-revision higher than pseudo-feedback (+10.21%).
geometric mean coverage-per-million-executions direction: selective-revision higher than pseudo-feedback (+123.18%).

## Independent Proof

Status counts: proved: 7, not_reproduced: 52151, crash_without_sanitizer: 0, failed: 0.
Independently ASan-proved target/hash pairs: libpng/3c552bbc99d63aefe1c7118a31e2b3e26620b3861a87d5d6a0e50e71c70ce02b, libxml2/000d88bec236cd20d746c80ecab84b1ffba0cae19545e4c28c604260b819c579, libxml2/3fe48ae32499ce045e8fc5179e67fd400f7f12ca0ea6ad33eea8f3ca4ae58b71, libxml2/5d662e6fc74c53205e63442082c80dffe1c919964535f0188a0b779672c4b28d, libxml2/6a86e62c324b1165956c97223bea8618ef23c534b8a89a4ef9fdff0283c55474, libxml2/786c384048e0c14bcea39c46cdc57903a5671812a752c3ec3314dde1e94c3bc3, libxml2/d583b4e3c659c3baa492ecc5b19238f78ca6e5b66bde29b8abbd3a11600ef907.
