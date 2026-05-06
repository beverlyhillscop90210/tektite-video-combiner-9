# Tektite Video Combiner 9.0

Standalone ComfyUI custom node variant with a new class name so it can be installed next to older versions without conflicts.

## Node Name
- `Tektite Video Combiner 9.0`

## Key Features
- `clip1..clip16` inputs
- Accepts video paths, `.mp4`, single `.png`, PNG folders/globs, image sequences, and Comfy `IMAGE` batches
- Preserves input slot order (`clip1`, `clip2`, ...)
- Wait/poll logic with timeout and stable polls
- Optional `audio` input for final mux

## Outputs
- `video` (VIDEO)
- `path` (STRING)

## Install
1. Copy this folder into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. Add node: `Tektite Video Combiner 9.0`

## Notes
- This package intentionally uses a different backend class (`TektiteVideoCombiner9`) so it can live next to older versions.
