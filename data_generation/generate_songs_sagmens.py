"""
generate_songs_sagmens.py  —  Pipeline Step 2b: catalog → MP3 segments

Reads docs/catalog_with_paths.md, loads three 30-second segments from each
matched audio file (beginning / middle / end), and saves each segment as a
clean MP3 file into the matching data/<category>/ folder.

No source audio files are copied or modified.  Only MP3 segments are written.

Segment naming:  <Sanitized_Track_Name>_seg1.mp3
                 <Sanitized_Track_Name>_seg2.mp3
                 <Sanitized_Track_Name>_seg3.mp3

Requires Python 3.7+
Requires ffmpeg installed and on PATH — pydub delegates MP3 encoding to it.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import librosa
import mutagen
import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
# This script lives inside data_generation/, so the project root is one level up.
SCRIPT_DIR   = Path(__file__).parent          # …/data_generation/
PROJECT_ROOT = SCRIPT_DIR.parent              # …/audio_mood_classifier/
CATALOG_PATH = PROJECT_ROOT / "docs" / "catalog_with_paths.md"
DATA_DIR     = PROJECT_ROOT / "data" / "mp3_data"

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE           = 22050   # Hz resampling target (matches project standard)
SEGMENT_SECS          = 30      # fixed boundary buffer: seconds skipped at each end of every track
SEGMENT_DURATION_SECS = 10.0   # duration (seconds) of each extracted segment — adjust freely
NUM_SEGMENTS          = 6       # number of segments to extract per track — adjust freely
TARGET_LUFS           = -20.0  # integrated loudness target for EBU R128 normalization
MP3_BITRATE           = "192k"  # pydub/ffmpeg MP3 export quality

# ── Catalog regex patterns ────────────────────────────────────────────────────
# Matches: ## calm_melancholic (85 tracks)
CATEGORY_RE     = re.compile(r"^## (calm_melancholic|moderate_neutral|energetic_upbeat)")
# Matches: - **Artist - Title**  (trailing whitespace / \r tolerated)
TRACK_RE        = re.compile(r"^- \*\*(.+?)\*\*\s*$")
# Extracts the content of any backtick-quoted token on a line.
NETWORK_PATH_RE = re.compile(r"`([^`]+)`")


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    """
    Remove characters that are illegal in Windows / POSIX filenames and
    replace whitespace with underscores.  Truncated to 120 characters to
    stay well within OS path-length limits.
    """
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:120]


def segment_offsets(total_secs: float) -> dict[int, float]:
    """
    Return NUM_SEGMENTS start offsets (in seconds), keyed 1 … NUM_SEGMENTS.

    Three independent constants govern the layout:
        buffer  = SEGMENT_SECS          (30 s)   — fixed safety zone at each end
        seg     = SEGMENT_DURATION_SECS (configurable) — length of each window
        n       = NUM_SEGMENTS          (configurable) — number of windows

    Standard case — total duration ≥ (2 × buffer + n × seg):
        A symmetrical buffer-second safety zone is enforced at both ends of the
        track; the opening and closing buffer-seconds are never sampled.
        The n windows are distributed with evenly-spaced start positions across
        the interior region [buffer, total − buffer − seg] using np.linspace.

        linspace produces n values from the first anchor (buffer) to the last
        anchor (total − buffer − seg), inclusive.  This guarantees:
          • Segment 1 always starts exactly at buffer (the start guard boundary).
          • Segment n always starts exactly at total − buffer − seg (its window
            ends at total − buffer, touching the end guard boundary).
          • All intermediate segments are spaced uniformly between the two anchors.
          • No segment window exceeds the track or the boundary buffers.

        The threshold  total ≥ 2 × buffer + n × seg  is the minimum length
        that fits n non-overlapping seg-second windows with a buffer-second
        guard at each end.

    Short-track fallback — total duration < threshold:
        The track is divided into n equal parts starting from 0.  Each segment
        starts at the beginning of its respective part.  All offsets are
        guaranteed to be within [0, total_secs), so librosa never seeks past
        EOF.  Segments that extend beyond the end of the file are zero-padded
        by process_catalog to the configured segment length.
    """
    seg    = float(SEGMENT_DURATION_SECS)   # configurable segment length
    buffer = float(SEGMENT_SECS)            # 30.0 — fixed boundary guard, separate concern
    n      = int(NUM_SEGMENTS)

    threshold = 2.0 * buffer + n * seg

    if total_secs >= threshold:
        # np.linspace places n anchors from the first start to the last start,
        # inclusive.  The last start is total − buffer − seg so that the final
        # window ends exactly at total − buffer.
        anchors = np.linspace(buffer, total_secs - buffer - seg, n)
        return {i + 1: float(anchors[i]) for i in range(n)}

    # Short-track fallback: evenly divide the available duration into n parts.
    # (n-1)/n * total_secs < total_secs is always true, so no offset exceeds the file.
    step = total_secs / n
    return {i + 1: float(i * step) for i in range(n)}


def save_segment_mp3(y: np.ndarray, sr: int, out_path: Path) -> None:
    """
    Write a float32 waveform array to disk as an MP3 file via pydub.

    The float32 samples (range −1.0 … 1.0) are converted to signed 16-bit
    integers, wrapped in a pydub AudioSegment, then exported at MP3_BITRATE.
    pydub delegates the actual encoding to ffmpeg, which must be on PATH.
    """
    y_int16 = (y * 32767.0).clip(-32768, 32767).astype(np.int16)
    audio_segment = AudioSegment(
        y_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,   # int16 → 2 bytes per sample
        channels=1,
    )
    audio_segment.export(str(out_path), format="mp3", bitrate=MP3_BITRATE)


# ── Path extraction helper ────────────────────────────────────────────────────

def _extract_network_path(line: str) -> Path | None:
    """
    Extract the first backtick-quoted UNC or Windows absolute path from a
    catalog line.  Returns None if no qualifying path token is found.

    Qualifying paths:
      • UNC shares    — start with two backslashes, e.g. ``\\GAMES\\...``
      • Drive paths   — second character is ``:``,    e.g. ``C:\\...``

    Imposes no constraints on surrounding whitespace, bullet characters,
    em-dash suffixes, or indentation depth.
    """
    m = NETWORK_PATH_RE.search(line)
    if not m:
        return None
    candidate = m.group(1).strip()
    if candidate.startswith(r"\\") or (len(candidate) >= 2 and candidate[1] == ":"):
        return Path(candidate)
    return None


# ── Catalog parsing ───────────────────────────────────────────────────────────

def parse_catalog(md_path: Path) -> list[tuple[str, str, Path]]:
    """
    Parse catalog_with_paths.md and return:
        [(category, track_display_name, audio_path), ...]

    Path detection uses _extract_network_path so any backtick-quoted UNC or
    absolute path is captured regardless of its surrounding Markdown formatting.
    Only the first qualifying path per track is kept (paths are written in
    descending match-score order by prepare_dataset.py).
    Tracks with *[NO MATCH FOUND]* produce no entries and are silently skipped.
    """
    entries: list[tuple[str, str, Path]] = []
    current_category: str | None = None
    current_track:    str | None = None
    track_claimed = False

    with open(md_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()   # strips \n, \r, and trailing spaces

            m = CATEGORY_RE.match(line)
            if m:
                current_category = m.group(1)
                current_track    = None
                track_claimed    = False
                continue

            # Any other ## heading (e.g. ## Summary) exits the category scope
            if line.startswith("## "):
                current_category = None
                current_track    = None
                continue

            m = TRACK_RE.match(line)
            if m:
                current_track = m.group(1)
                track_claimed = False
                continue

            if current_category and current_track and not track_claimed:
                audio_path = _extract_network_path(line)
                if audio_path is not None:
                    entries.append((current_category, current_track, audio_path))
                    track_claimed = True

    return entries


# ── Pre-run verification ──────────────────────────────────────────────────────

def verify_catalog(
    md_path: Path,
) -> tuple[list[tuple[str, str, Path]], list[tuple[str, str, str]]]:
    """
    Scan the entire catalog and classify every track as valid or problematic.

    Rules applied per track:
      • A qualifying path line is found and the file exists on disk  → valid.
      • A qualifying path line is found but the file is absent       → error:
            "File not found on network: <path>"
      • A *[NO MATCH FOUND]* line is found before any path line     → error:
            "No file path in catalog (unmatched track)"
      • No qualifying path or NO MATCH line found anywhere under the
        track heading before the end of its category section        → error:
            "Missing path structure in catalog"

    Returns
    -------
    valid_entries : [(category, track_name, audio_path), ...]
    errors        : [(category, track_name, reason_string), ...]
    """
    valid_entries: list[tuple[str, str, Path]] = []
    errors:        list[tuple[str, str, str]]  = []

    current_category: str | None = None
    current_track:    str | None = None

    all_track_keys:  list[tuple[str, str]] = []
    seen_track_keys: set[tuple[str, str]]  = set()
    resolved_keys:   set[tuple[str, str]]  = set()

    with open(md_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()

            m = CATEGORY_RE.match(line)
            if m:
                current_category = m.group(1)
                current_track    = None
                continue

            if line.startswith("## "):
                current_category = None
                current_track    = None
                continue

            m = TRACK_RE.match(line)
            if m:
                current_track = m.group(1)
                if current_category:
                    key = (current_category, current_track)
                    if key not in seen_track_keys:
                        all_track_keys.append(key)
                        seen_track_keys.add(key)
                continue

            if not current_category or not current_track:
                continue

            key = (current_category, current_track)

            if re.search(r"\[NO MATCH FOUND\]", line) and key not in resolved_keys:
                errors.append((
                    current_category, current_track,
                    "No file path in catalog (unmatched track)",
                ))
                resolved_keys.add(key)
                continue

            if key not in resolved_keys:
                audio_path = _extract_network_path(line)
                if audio_path is not None:
                    resolved_keys.add(key)
                    if not audio_path.exists():
                        errors.append((
                            current_category, current_track,
                            "File not found on network: " + str(audio_path),
                        ))
                    else:
                        valid_entries.append((current_category, current_track, audio_path))

    for cat, trk in all_track_keys:
        if (cat, trk) not in resolved_keys:
            errors.append((cat, trk, "Missing path structure in catalog"))

    return valid_entries, errors


def run_verification_loop(md_path: Path) -> list[tuple[str, str, Path]]:
    """
    Run verify_catalog in a loop.  The menu is ALWAYS displayed after every
    scan so the user can review the full track list before committing to a
    potentially long export run — even when zero errors are found.

    Flow (every iteration)
    ----------------------
    1. Run verify_catalog.
    2. Print a numbered list of all valid tracks grouped by category.
    3. If errors exist, print the error report below the track list.
    4. Show the interactive menu:
         [1]  Proceed with the valid tracks (label adapts to error state).
         [2]  Pause, fix catalog_with_paths.md, press Enter to re-scan.
       Option 2 loops back to step 1 with a fresh scan.
       Option 1 returns the valid entry list to process_catalog.
    """
    while True:
        print("Verifying catalog…")
        valid_entries, errors = verify_catalog(md_path)

        # Silent parse failure: both lists empty means nothing was parsed at
        # all — treat as a hard error so the menu still appears.
        if not errors and not valid_entries:
            errors = [(
                "catalog",
                "(entire file)",
                "No tracks found — catalog_with_paths.md may be empty, "
                "unreadable, or formatted in an unexpected way.",
            )]

        # ── Numbered track list grouped by category ────────────────────────
        print()
        print("─" * 60)
        if valid_entries:
            by_category: dict[str, list[str]] = {}
            for cat, name, _ in valid_entries:
                by_category.setdefault(cat, []).append(name)

            print("VERIFIED TRACKS  (" + str(len(valid_entries)) + " ready for processing)")
            print("─" * 60)
            counter = 1
            for cat, names in by_category.items():
                print("  ## " + cat + "  (" + str(len(names)) + " tracks)")
                for name in names:
                    print("  " + str(counter).rjust(3) + ".  " + name)
                    counter += 1
                print()
        else:
            print("VERIFIED TRACKS  (0 ready for processing)")
            print("─" * 60)
            print()

        # ── Error report (only printed when problems exist) ────────────────
        if errors:
            print("─" * 60)
            print("VERIFICATION ISSUES  (" + str(len(errors)) + " problematic tracks)")
            print("─" * 60)
            for category, track_name, reason in errors:
                print("  [" + category + "]")
                print("  Track  : " + track_name)
                print("  Reason : " + reason)
                print()
            print("─" * 60)

        # ── Summary line ───────────────────────────────────────────────────
        print()
        print("  Valid tracks ready   : " + str(len(valid_entries)))
        print("  Problematic tracks   : " + str(len(errors)))
        print()

        # ── Interactive menu ───────────────────────────────────────────────
        if errors:
            option1 = (
                "  [1]  Skip the " + str(len(errors)) + " problematic track(s) and export\n"
                "       MP3 segments for the " + str(len(valid_entries)) + " valid tracks."
            )
        else:
            option1 = (
                "  [1]  Export MP3 segments for all "
                + str(len(valid_entries)) + " verified tracks."
            )

        while True:
            print("How would you like to proceed?")
            print()
            print(option1)
            print()
            print("  [2]  Pause execution so you can fix catalog_with_paths.md.")
            print("       Press Enter after saving to re-run the verification scan.")
            print()
            choice = input("Your choice (1 or 2): ").strip()

            if choice == "1":
                print()
                print("Proceeding with " + str(len(valid_entries)) + " tracks.")
                print()
                return valid_entries

            if choice == "2":
                print()
                print("Execution paused.  Edit catalog_with_paths.md now.")
                input("Press Enter when ready to re-scan…")
                print()
                break   # exit inner loop → re-enter outer loop (fresh scan)

            print("  Invalid input — please enter 1 or 2.")
            print()


# ── Main processing loop ──────────────────────────────────────────────────────

def process_catalog(entries: list[tuple[str, str, Path]]) -> None:
    """
    Iterate over every catalog entry, extract three 30-second segments via
    librosa, and save one MP3 per segment into data/<category>/.

    Segments are loaded individually using librosa's offset + duration
    parameters so only 30 seconds of audio are decoded at a time — far
    faster than loading full tracks into memory.

    Total duration is read through mutagen (metadata only, no decoding)
    to calculate the middle and end offsets before any audio is loaded.
    """
    generated = 0
    skipped   = 0
    seg_samples = int(SEGMENT_DURATION_SECS * SAMPLE_RATE)

    for category, track_name, audio_path in tqdm(
        entries, desc="MP3 Segments", unit="track", dynamic_ncols=True
    ):
        dest_dir = DATA_DIR / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = sanitize(track_name)

        try:
            # Read total duration from metadata (no audio decoding required)
            meta = mutagen.File(audio_path)
            if meta is None or meta.info is None:
                raise ValueError("mutagen could not read audio metadata")
            total_secs = float(meta.info.length)

            offsets = segment_offsets(total_secs)

            for idx, offset in offsets.items():
                # Load only the required segment window
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, sr = librosa.load(
                        audio_path,
                        sr=SAMPLE_RATE,
                        mono=True,
                        offset=offset,
                        duration=float(SEGMENT_DURATION_SECS),
                    )

                # Zero-pad if the track ended before the segment was complete
                if len(y) < seg_samples:
                    y = np.pad(y, (0, seg_samples - len(y)))

                # ── EBU R128 loudness normalization ───────────────────────
                # Measure the integrated loudness of the segment and scale it
                # to TARGET_LUFS.  pyloudnorm requires float64 input.
                # Silent segments (loudness = -inf) are skipped safely — the
                # meter raises a special warning rather than returning a valid
                # number, so we guard with a try/except and leave them as-is.
                try:
                    meter = pyln.Meter(sr)   # BS.1770 meter at the segment's sample rate
                    loudness = meter.integrated_loudness(y.astype(np.float64))
                    y = pyln.normalize.loudness(
                        y.astype(np.float64), loudness, TARGET_LUFS
                    ).astype(np.float32)
                except Exception:
                    pass   # silent or unmeasurable segment — write as-is

                # Safety clip: normalization boosts can push quiet segments
                # above ±1.0; clip to the float32 legal range before int16 conversion.
                y = np.clip(y, -1.0, 1.0)

                out_path = dest_dir / (stem + "_seg" + str(idx) + ".mp3")
                save_segment_mp3(y, sr, out_path)

            generated += len(offsets)

        except Exception as exc:
            tqdm.write("  [WARN] Skipped '" + track_name + "': " + str(exc))
            skipped += 1

    print("─" * 60)
    print("MP3 segments generated : " + str(generated))
    print("Tracks skipped         : " + str(skipped))
    print("─" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("generate_songs_sagmens.py  —  catalog → MP3 segments")
    print("=" * 60)
    print("Catalog  : " + str(CATALOG_PATH))
    print("Output   : " + str(DATA_DIR))
    print()

    if not CATALOG_PATH.exists():
        raise SystemExit(
            "ERROR: catalog_with_paths.md not found at:\n"
            "  " + str(CATALOG_PATH) + "\n"
            "Run prepare_dataset.py first to generate the catalog."
        )

    entries = run_verification_loop(CATALOG_PATH)

    process_catalog(entries)


if __name__ == "__main__":
    main()
