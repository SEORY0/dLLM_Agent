# libTIFF structured particle

Select one catalog index for every field of a bounded little-endian uncompressed TIFF. Favor
cross-field combinations that stress IFD parsing and raster conversion while keeping offsets,
entry counts, strip sizes, and the pixel payload internally consistent. The runtime freezes
working fields and may re-mask only a feedback-attributed field.

| Field | Selector catalog |
| --- | --- |
| `image_width` | `0=1`, `1=2`, `2=3`, `3=4` |
| `image_length` | `0=1`, `1=2`, `2=4`, `3=8` |
| `bits_per_sample` | `0=8`, `1=16` |
| `compression` | `0=none` |
| `photometric` | `0=white-is-zero`, `1=black-is-zero` |
| `strip_offsets` | `0=derived valid offset` |
| `samples_per_pixel` | `0=1` |
| `rows_per_strip` | `0=derived image length` |
| `strip_byte_counts` | `0=derived payload size` |
| `image_payload` | `0=00`, `1=55`, `2=aa`, `3=ff` |
