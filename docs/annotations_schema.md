# Annotations Schema

## File: `annotations.json`

Each pitch recording folder (`pitches/<id>/`) can contain an `annotations.json` file
with manually-labelled ball positions for every frame.

## Schema

```json
{
  "schema_version": "1.0",
  "folder": "pitches/20260301_030216",
  "n_frames": 120,
  "image_dimensions": {
    "width": 1164,
    "height": 892
  },
  "frame_files": [
    "frame_000000.png",
    "frame_000001.png",
    "..."
  ],
  "annotations": {
    "0": {
      "x": null,
      "y": null,
      "visible": false,
      "frame_file": "frame_000000.png"
    },
    "55": {
      "x": 612,
      "y": 185,
      "visible": true,
      "frame_file": "frame_000055.png"
    }
  }
}
```

## Field Reference

### Top-level

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Schema version identifier (currently `"1.0"`) |
| `folder` | string | Path to the pitch recording folder |
| `n_frames` | int | Total number of frames in the folder |
| `image_dimensions` | object | `{"width": int, "height": int}` of source frames |
| `frame_files` | list[string] | Ordered list of frame filenames |
| `annotations` | dict | Frame-index → annotation mapping |

### Per-frame annotation

| Field | Type | Description |
|---|---|---|
| `x` | int or null | Ball center X coordinate (pixels, frame-relative) |
| `y` | int or null | Ball center Y coordinate (pixels, frame-relative) |
| `visible` | bool | Whether the ball is visible in this frame |
| `frame_file` | string | Filename of the annotated frame |

## Rules

- **Keys** in `annotations` are string-encoded frame indices (e.g., `"0"`, `"55"`)
- When `visible` is `false`, `x` and `y` must be `null`
- When `visible` is `true`, `x` and `y` must be integers
- Coordinates are in **frame-relative pixels** (not screen-absolute)
- Not all frames need to be annotated - unannotated frames are treated as "unknown"
- The annotator (`frame_annotator.py`) auto-saves on quit

## Creating Annotations

```bash
python frame_annotator.py pitches/20260301_030216
```

Controls:
- **Left-click** = Mark ball center
- **N** = Mark frame as "not visible"
- **B / Left arrow** = Go back one frame
- **R** = Remove current annotation
- **S** = Save
- **Q / ESC** = Quit (auto-saves)

## Consuming Annotations

Used by:
- `tools/export_yolo.py` - converts to YOLO bounding box labels
- `validate_tracking.py` - compares tracker output against ground truth
- `track_folder.py` - optional error comparison overlay
