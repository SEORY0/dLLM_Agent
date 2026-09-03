# libxml2 structured particle

Select one catalog index for every field of a well-formed XML document. Favor combinations that
exercise declaration, namespace, attribute, text, repetition, and whitespace handling together
without breaking XML well-formedness. The runtime freezes productive fields and may re-mask only
a field associated with stalled executable feedback.

| Field | Selector catalog |
| --- | --- |
| `xml_version` | four quote/spacing forms of XML 1.0 |
| `encoding` | `0=UTF-8`, `1=US-ASCII`, `2=ISO-8859-1`, `3=UTF-8` |
| `namespace_uri` | `0..3=urn:dllm:a..d` |
| `root_name` | `0=root` |
| `item_name` | `0=item`, `1=node`, `2=case`, `3=leaf` |
| `attr_name` | `0=id`, `1=kind`, `2=slot`, `3=flag` |
| `attr_value` | `0=a`, `1=b`, `2=c`, `3=d` |
| `text_value` | `0=alpha`, `1=beta`, `2=gamma`, `3=delta` |
| `child_count` | `0=1`, `1=2`, `2=3`, `3=4` |
| `whitespace` | newline, indented newline, space, or tabbed newline |
