# libpng structured particle

Select one catalog index for every field of a syntactically valid PNG. Favor combinations that
exercise unusual parser state transitions while preserving the PNG signature, chunk framing,
CRC values, compressed scanline validity, and cross-field length consistency. The runtime keeps
unchanged fields frozen and may re-mask only a field associated with stalled executable feedback.

| Field | Selector catalog |
| --- | --- |
| `width` | `0=1`, `1=2`, `2=3`, `3=4` |
| `height` | `0=1`, `1=2`, `2=4`, `3=8` |
| `bit_depth` | `0=8` |
| `color_type` | `0=truecolor` |
| `compression_method` | `0=deflate` |
| `filter_method` | `0=adaptive` |
| `interlace_method` | `0=none` |
| `text_key` | `0=k`, `1=note`, `2=seed`, `3=case` |
| `text_value` | `0=a`, `1=b`, `2=c`, `3=d` |
| `image_payload` | `0=00`, `1=55`, `2=aa`, `3=ff` |
