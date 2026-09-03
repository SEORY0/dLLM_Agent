# OpenEXR PoC Revision Prompt

Generate `EXR_PARTICLE_V1` selector slots for the `selective-revision/v1`
treatment. Preserve anchor grammar tokens and only change explicitly remasked
field slots when revision feedback selects a field.

Target behavior:

- Keep OpenEXR magic/version, required attribute names, and grammar delimiters frozen.
- Use the provided ten selector slots in order.
- Prefer boundary values that exercise data/display windows, chunk offsets, channel
  metadata, payload patterns, compression, and line order.
- Do not claim a vulnerability from a probe signal; proof is separate.
