# Audio Mood Classifier — Spectrogram Generator Implementation

**Date:** 2026-06-03  
**Script:** `generate_spectrograms.py` — Pipeline Step 2  
**Role:** Reads `docs/catalog_with_paths.md`, validates every track, extracts three 30-second audio segments per song, and saves one Mel-spectrogram PNG per segment into `data/<category>/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration Constants](#2-configuration-constants)
3. [Robust Markdown Catalog Parsing](#3-robust-markdown-catalog-parsing)
4. [Pre-Run Verification and Interactive Menu Loop](#4-pre-run-verification-and-interactive-menu-loop)
5. [Three-Segment Extraction Math](#5-three-segment-extraction-math)
6. [Peak Volume Normalization](#6-peak-volume-normalization)
7. [Spectrogram Generation](#7-spectrogram-generation)
8. [Output Naming and Directory Structure](#8-output-naming-and-directory-structure)
9. [Error Handling Strategy](#9-error-handling-strategy)
10. [Full Execution Flow](#10-full-execution-flow)
11. [Dependencies](#11-dependencies)

---

## 1. Overview

`generate_spectrograms.py` is the second step of the audio mood classifier pipeline. It consumes the Markdown report produced by `prepare_dataset.py` (`docs/catalog_with_paths.md`) rather than reading from the original audio library directly, keeping each pipeline stage independent.

The script deliberately performs no file copying. Its only side-effect on disk is writing PNG images into the three pre-existing `data/` subdirectories:

```
data/
├── calm_melancholic/
├── moderate_neutral/
└── energetic_upbeat/
```

For each matched track, three PNG files are written:

```
data/<category>/<Sanitized_Track_Name>_1.png   ← segment 1
data/<category>/<Sanitized_Track_Name>_2.png   ← segment 2
data/<category>/<Sanitized_Track_Name>_3.png   ← segment 3
```

---

## 2. Configuration Constants

All tunable values live at the top of the script so nothing else needs to be touched between runs.

| Constant | Value | Purpose |
|---|---|---|
| `SAMPLE_RATE` | `22050` Hz | Librosa resampling target. Standard for speech/music ML. |
| `SEGMENT_SECS` | `30` | Duration of each extracted window in seconds. |
| `N_MELS` | `128` | Number of Mel filterbank frequency bins. |
| `HOP_LENGTH` | `512` | STFT hop length in samples (~23 ms at 22050 Hz). |
| `IMAGE_SIZE` | `(4, 4)` inches | Figure size → 400 × 400 px at 100 dpi. |

Three compiled regex constants control all catalog parsing:

| Constant | Pattern | Matches |
|---|---|---|
| `CATEGORY_RE` | `^## (calm_melancholic\|moderate_neutral\|energetic_upbeat)` | Category section headings |
| `TRACK_RE` | `^- \*\*(.+?)\*\*\s*$` | Track title bullets (trailing `\r`/spaces tolerated) |
| `NETWORK_PATH_RE` | `` `([^`]+)` `` | Any backtick-quoted token on any line |

---

## 3. Robust Markdown Catalog Parsing

### 3.1 Design Philosophy

The original parsing used a strict `PATH_RE` that required a precise surrounding format (`  - \`path\` — [MM:SS]`). Any deviation — extra whitespace, a missing em-dash, different indentation — caused a silent miss. The rewritten approach separates two concerns:

- **Position detection** — handled by the state machine (which category and track are currently active).
- **Path extraction** — handled by `_extract_network_path()`, which scans for any backtick-quoted content that looks like a filesystem path regardless of its surrounding characters.

### 3.2 `_extract_network_path(line)`

```python
def _extract_network_path(line: str) -> Path | None:
    m = NETWORK_PATH_RE.search(line)   # re.search, not re.match
    if not m:
        return None
    candidate = m.group(1).strip()
    if candidate.startswith(r"\\") or (len(candidate) >= 2 and candidate[1] == ":"):
        return Path(candidate)
    return None
```

**Qualifying paths accepted:**

- UNC network shares: `\\SERVER\share\...` (starts with two backslashes)
- Windows absolute paths: `C:\...` (second character is `:`)

Everything else — inline code snippets, `[MM:SS]` duration tokens, table cell values — is rejected. Using `re.search` instead of `re.match` means the path token can appear anywhere on the line without any leading-character requirement.

### 3.3 State Machine

Both `parse_catalog` and `verify_catalog` use the same three-state line classifier:

```
1. CATEGORY_RE matches → set current_category, reset current_track
2. line.startswith("## ") but CATEGORY_RE did NOT match
   → non-category heading (e.g. ## Summary) — reset BOTH to None
   → prevents the last energetic_upbeat track from "leaking" into
     the summary section and receiving false path attributions
3. TRACK_RE matches → set current_track, reset claimed/resolved flag
4. Any other line, while current_category and current_track are set
   → call _extract_network_path; if a path is found, record the entry
```

**Line stripping:** Both parsers use `raw_line.rstrip()` (strips `\n`, `\r`, spaces, tabs) rather than the narrower `raw_line.rstrip("\n")`. On Windows, files commonly have `\r\n` line endings; `rstrip("\n")` alone leaves a trailing `\r` that causes `TRACK_RE`'s `$` anchor to fail silently, dropping the entire track from the index with no error.

**First-path-only rule:** A `track_claimed` / `resolved_keys` flag ensures only the first qualifying path per track is consumed. Paths in `docs/catalog_with_paths.md` are written in descending match-score order by `prepare_dataset.py`, so the first path is always the highest-confidence match.

### 3.4 `parse_catalog` vs `verify_catalog`

| Function | Purpose | Returns |
|---|---|---|
| `parse_catalog` | Lightweight parse for internal use | `[(category, name, path), ...]` — matched tracks only |
| `verify_catalog` | Full audit pass with error classification | `(valid_entries, errors)` — all tracks accounted for |

`verify_catalog` adds two additional data structures absent from `parse_catalog`:

- `all_track_keys` — ordered list of every `(category, track_name)` pair seen during the scan, deduplicated by `seen_track_keys`.
- `resolved_keys` — set of keys for which a path or `[NO MATCH FOUND]` sentinel was encountered.

After the full file pass, any key in `all_track_keys` that is not in `resolved_keys` is flagged as **"Missing path structure in catalog"** via a post-loop sweep. This approach is position-independent: it does not matter how many blank lines, comments, or extra bullets separate a track heading from its path line — resolution is determined by whether a path was found *anywhere* in the file under that track's scope.

---

## 4. Pre-Run Verification and Interactive Menu Loop

### 4.1 Error Classification

`verify_catalog` classifies every track into exactly one outcome:

| Outcome | Condition | Error message |
|---|---|---|
| **Valid** | Path line found and `Path.exists()` is True | *(added to `valid_entries`)* |
| **File not found** | Path line found but `Path.exists()` is False | `"File not found on network: <path>"` |
| **Unmatched** | `[NO MATCH FOUND]` sentinel encountered | `"No file path in catalog (unmatched track)"` |
| **Missing structure** | No path or sentinel found anywhere under the heading | `"Missing path structure in catalog"` |

Because every track in `all_track_keys` ends up in either `valid_entries` or `errors`, the invariant `len(all_track_keys) == len(valid_entries) + len(errors)` always holds.

### 4.2 `run_verification_loop`

The function contains two nested `while True` loops that implement the interactive flow.

**Outer loop** — runs `verify_catalog` on each iteration:

```
verify_catalog()
    ↓
Silent failure guard
(if both lists empty → inject synthetic error)
    ↓
Print numbered track list (always)
    ↓
Print error report (only if errors exist)
    ↓
Print summary line
    ↓
Enter inner menu loop
```

**Silent failure guard:**

```python
if not errors and not valid_entries:
    errors = [("catalog", "(entire file)",
               "No tracks found — docs/catalog_with_paths.md may be empty...")]
```

If regex patterns match nothing at all (empty catalog, corrupt file, unexpected encoding), both lists remain empty. Without this guard, `not errors` would be True and the script would call `process_catalog([])`, doing nothing silently. The guard converts this into a visible error and forces the menu to appear.

**Numbered track list:** After every scan, `valid_entries` is grouped into a `dict[category → [names]]` using `dict.setdefault` (preserves insertion order). A single counter increments across all three categories, producing a continuous numbered list from 1 to N.

**Inner loop** — handles the menu prompt:

```
[1] → return valid_entries  (exits both loops)
[2] → pause, wait for Enter, break inner loop
      (outer loop re-runs verify_catalog from scratch)
anything else → re-prompt
```

**Menu label adapts to error state:**

- With errors: *"Skip the N problematic track(s) and generate spectrograms for the M valid tracks."*
- Without errors: *"Generate spectrograms for all N verified tracks."*

**No automatic bypass:** There is no code path that returns from `run_verification_loop` without the user having explicitly pressed `[1]`. The function always pauses, regardless of whether errors were found.

---

## 5. Three-Segment Extraction Math

### 5.1 Standard Case (track duration ≥ 90 seconds)

Three fixed 30-second windows are extracted using exact timestamps:

| Segment | Offset formula | Window |
|---|---|---|
| 1 — Shifted start | `30.0` | `[30 s → 60 s]` |
| 2 — Midpoint | `total / 2 − 15` | `[mid−15 → mid+15]` |
| 3 — End | `total − 30` | `[end−30 → end]` |

**Why skip the first 30 seconds for Segment 1?** Tracks frequently open with intros, fade-ins, or spoken introductions that are not representative of the song's primary mood content. Starting at 30 seconds places Segment 1 in the first chorus or main verse of most songs.

**Why 90 seconds as the threshold?** At `total = 90`:
- Segment 1 occupies `[30, 60]`
- Segment 2 occupies `[30, 60]` (mid = 45, window = 45−15 to 45+15)
- Segment 3 occupies `[60, 90]`

Segments 1 and 2 are adjacent without overlap. Below 90 seconds, the fixed timestamps would cause overlap, so a different strategy is applied.

**Boundary guarantees for standard case (total ≥ 90):**

- Segment 1 offset `30.0 ≥ 0` always. ✓
- Segment 2 offset `total/2 − 15 ≥ 0` when `total ≥ 30`. ✓
- Segment 3 offset `total − 30 ≥ 0` when `total ≥ 30`. ✓
- No offset exceeds `total_secs`. ✓

### 5.2 Short-Track Fallback (track duration < 90 seconds)

```python
third = total_secs / 3.0
return {1: 0.0, 2: third, 3: third * 2.0}
```

The track is divided into three equal thirds. Each segment begins at the start of its respective third. This guarantees:

- All three start positions are distinct.
- `third × 2 = 2/3 × total < total` — no offset ever exceeds the file length.
- No clamping required.

For tracks shorter than 30 seconds, a segment will load fewer than `SEGMENT_SECS × SAMPLE_RATE` samples. The zero-padding step in `process_catalog` fills the remainder with silence:

```python
if len(y) < seg_samples:
    y = np.pad(y, (0, seg_samples - len(y)))
```

This ensures every segment fed to `save_spectrogram` is exactly `seg_samples` samples long regardless of source duration.

### 5.3 Efficient Loading with `librosa.load` Offset/Duration

Instead of loading the full audio file into memory and slicing it, the script uses librosa's `offset` and `duration` parameters:

```python
y, sr = librosa.load(
    audio_path,
    sr=SAMPLE_RATE,
    mono=True,
    offset=offset,
    duration=float(SEGMENT_SECS),
)
```

Total duration is read first from mutagen metadata (no audio decoding required):

```python
meta = mutagen.File(audio_path)
total_secs = float(meta.info.length)
```

For a 5-minute track at 22 050 Hz, this decodes approximately 2.6 M samples (3 × 30 s) rather than 13.2 M samples (full track), reducing memory allocation and I/O by roughly 80%.

---

## 6. Peak Volume Normalization

### 6.1 Implementation

After zero-padding and before spectrogram generation, each 30-second waveform segment is peak-normalized:

```python
peak = np.max(np.abs(y))
if peak > 0.0:
    y = y / peak
```

### 6.2 What It Does

Peak normalization scales the waveform so that its maximum absolute amplitude equals 1.0. Every segment, regardless of the original track's mastering volume, recording era, or dynamic range, enters `save_spectrogram` with the same amplitude ceiling.

Without normalization, a quietly mastered 1970s recording would produce a visually dim spectrogram while a loudly mastered modern track would produce a bright one — not because the mood content differs, but because of production choices. This inconsistency would introduce spurious variance into the training data that the classifier would have to learn to ignore.

### 6.3 Division-by-Zero Guard

The `if peak > 0.0:` condition skips normalization for segments that are entirely silent (all samples are exactly zero). A silent segment has `np.max(np.abs(y)) == 0.0`; dividing by zero would produce a `nan`-filled array, which would cause a black or undefined spectrogram image and potentially corrupt the training data. The guard allows silent segments to pass through as all-zeros, which produce a valid (uniformly dark) spectrogram.

---

## 7. Spectrogram Generation

### 7.1 `save_spectrogram(y, sr, out_path)`

```python
S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
S_dB = librosa.power_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=IMAGE_SIZE)
ax.imshow(S_dB, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
plt.close(fig)
```

### 7.2 Parameters

**Mel-spectrogram:**
- `n_mels=128` — 128 frequency bins distributed on the Mel scale, which approximates human auditory perception.
- `hop_length=512` — STFT hop of ~23 ms at 22 050 Hz, giving ~1300 time frames for a 30-second segment.
- `librosa.power_to_db(S, ref=np.max)` — converts power to decibels, referenced to the peak power within the segment. This produces a spectrogram where the loudest frequency component is always 0 dB, and quieter components are negative.

**Image rendering:**
- `cmap="magma"` — perceptually uniform, print-friendly colormap. High energy → bright yellow/white; low energy → dark purple/black.
- `origin="lower"` — low frequencies at the bottom, high frequencies at the top (standard musicological convention).
- `aspect="auto"` — stretches the time/frequency axes to fill the 4×4 inch canvas.
- `ax.axis("off")` + `fig.subplots_adjust(left=0, right=1, top=1, bottom=0)` + `pad_inches=0` — produces a borderless image with no axes, tick marks, labels, or whitespace. The PNG contains only pixel data.

**Output size:** 4 inches × 100 dpi = **400 × 400 pixels** per PNG.

**Backend:** `matplotlib.use("Agg")` is set before `import matplotlib.pyplot` to select the non-interactive Agg renderer. This is required for headless server execution (no display required) and prevents matplotlib from attempting to open GUI windows during a batch run.

---

## 8. Output Naming and Directory Structure

### 8.1 Filename Sanitization

Track names from the catalog may contain characters illegal in Windows or POSIX filenames. `sanitize(name)` applies two transformations:

```python
name = re.sub(r'[\\/:*?"<>|]', "", name)   # remove illegal characters
name = re.sub(r"\s+", "_", name.strip())    # collapse whitespace → underscore
return name[:120]                            # guard against OS path-length limits
```

The 120-character truncation is a safety margin. Windows has a 260-character `MAX_PATH` limit; typical project root paths consume 60–80 characters, leaving ~180 for the filename. Truncating at 120 provides a comfortable buffer while retaining enough characters to be identifiable.

### 8.2 Output Path Formula

```
data/<category>/<sanitized_track_name>_<segment_index>.png
```

Example for *Massive Attack — Teardrop* in `calm_melancholic`:

```
data/calm_melancholic/Massive_Attack_-_Teardrop_1.png
data/calm_melancholic/Massive_Attack_-_Teardrop_2.png
data/calm_melancholic/Massive_Attack_-_Teardrop_3.png
```

The destination directory is created with `dest_dir.mkdir(parents=True, exist_ok=True)` before any write, so the script is safe to run even if the `data/` subdirectories were removed between runs.

---

## 9. Error Handling Strategy

### 9.1 Pre-Run vs Runtime

The script uses two complementary error handling layers:

**Layer 1 — Pre-run verification** (`verify_catalog` + `run_verification_loop`): Catches catalog-level problems before any audio processing begins. Errors here are actionable — the user can fix `docs/catalog_with_paths.md` and re-scan without losing any progress.

**Layer 2 — Runtime per-track exception handling** (`process_catalog`): Catches file-level problems during processing. A corrupt audio file, a broken codec, or an unexpected librosa error will print a warning via `tqdm.write` (which does not disrupt the progress bar) and increment `skipped`, but does not abort the entire batch.

```python
except Exception as exc:
    tqdm.write("  [WARN] Skipped '" + track_name + "': " + str(exc))
    skipped += 1
```

### 9.2 Librosa Warning Suppression

Librosa and its backends emit deprecation and codec warnings on many audio files. These are suppressed per-load using:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    y, sr = librosa.load(...)
```

The `catch_warnings` context manager ensures suppression is scoped strictly to the `librosa.load` call and does not hide warnings from any other part of the script.

---

## 10. Full Execution Flow

```
python generate_spectrograms.py
        │
        ▼
main()
├── Guard: docs/catalog_with_paths.md exists?
│   └── No → SystemExit with instructions
│
└── run_verification_loop(CATALOG_PATH)
    │
    └── [outer while True]
        │
        ├── verify_catalog(md_path)
        │   ├── Single sequential pass over docs/catalog_with_paths.md
        │   ├── Classifies every track → valid_entries or errors
        │   └── Post-loop: unresolved tracks → "Missing path structure"
        │
        ├── Silent failure guard (both lists empty → inject error)
        │
        ├── Print numbered track list (always)
        │
        ├── Print error report (if errors exist)
        │
        ├── Print summary
        │
        └── [inner while True] — interactive menu
            ├── [1] → return valid_entries  ──────────────────────┐
            ├── [2] → pause, re-scan on Enter (break inner loop)  │
            └── other → re-prompt                                  │
                                                                   │
        ┌──────────────────────────────────────────────────────────┘
        ▼
process_catalog(entries)
    │
    └── [tqdm loop over valid_entries]
        │
        ├── mutagen: read total_secs (metadata only, no decode)
        ├── segment_offsets(total_secs) → {1: t1, 2: t2, 3: t3}
        │
        └── [for each of 3 offsets]
            ├── librosa.load(offset=t, duration=30)
            ├── zero-pad if segment < 30 s
            ├── peak normalize (skip if silent)
            └── save_spectrogram → PNG
```

---

## 11. Dependencies

| Package | Role |
|---|---|
| `librosa` | Audio loading (`librosa.load`), Mel-spectrogram (`librosa.feature.melspectrogram`), dB conversion (`librosa.power_to_db`) |
| `mutagen` | Audio metadata reading without full decoding (used to obtain total duration for offset calculation) |
| `matplotlib` | Spectrogram rendering and PNG export (Agg non-interactive backend) |
| `numpy` | Array operations: zero-padding (`np.pad`), peak normalization (`np.max`, `np.abs`) |
| `tqdm` | Terminal progress bar wrapping the main track loop |

All five packages are listed in `requirements.txt` in the project root. Install with:

```powershell
pip install -r requirements.txt
```
