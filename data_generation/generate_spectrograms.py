"""
generate_spectrograms.py  —  Pipeline Step 2: catalog → mel-spectrograms

Reads catalog_with_paths.md, loads three 30-second segments from each matched
audio file (beginning / middle / end), converts each segment to a
Mel-spectrogram, and saves a clean PNG into the matching data/<category>/ folder.

No audio files are copied or modified.  Only PNG images are written.

Requires Python 3.7+
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import mutagen
import numpy as np
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
# This script lives inside data_generation/, so the project root is one level up.
SCRIPT_DIR   = Path(__file__).parent          # …/data_generation/
PROJECT_ROOT = SCRIPT_DIR.parent              # …/audio_mood_classifier/
CATALOG_PATH = PROJECT_ROOT / "docs" / "catalog_with_paths.md"
DATA_DIR     = PROJECT_ROOT / "data" / "spectrograms"

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE           = 22050   # Hz resampling target (librosa default)
SEGMENT_SECS          = 30      # fixed boundary buffer: seconds skipped at each end of every track
SEGMENT_DURATION_SECS = 05.0   # duration (seconds) of each extracted segment — adjust freely
N_MELS                = 128     # mel filterbank bins
HOP_LENGTH            = 512     # STFT hop length in samples
IMAGE_SIZE            = (4, 4)  # output figure size in inches → 400×400 px at 100 dpi

# ── Catalog regex patterns ────────────────────────────────────────────────────
# Matches: ## calm_melancholic (85 tracks)
CATEGORY_RE     = re.compile(r"^## (calm_melancholic|moderate_neutral|energetic_upbeat)")
# Matches: - **Artist - Title**  (trailing whitespace / \r tolerated)
TRACK_RE        = re.compile(r"^- \*\*(.+?)\*\*\s*$")
# Extracts the content of any backtick-quoted token on a line.
# Used by _extract_network_path — no assumptions about surrounding formatting.
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
    Return the three segment start offsets (in seconds).

    Two independent constants govern the layout:
        buffer  = SEGMENT_SECS          (30 s) — fixed safety zone at each end
        seg     = SEGMENT_DURATION_SECS (configurable) — length of each window

    Standard case — total duration ≥ (2 × buffer + 3 × seg):
        A symmetrical buffer-second safety zone is enforced at both ends of the
        track; the opening and closing buffer-seconds are never sampled.
        The three windows sit inside the remaining interior:

        Segment 1  shifted start  : offset  buffer                 window [buffer → buffer+seg]
        Segment 2  midpoint       : offset  mid − seg/2            window [mid−seg/2 → mid+seg/2]
        Segment 3  buffered end   : offset  total − buffer − seg   window [total−buffer−seg → total−buffer]

        The threshold is derived from the no-overlap constraint between
        Segment 2 and Segment 3:
            mid + seg/2  ≤  total − buffer − seg
            →  total  ≥  2 × buffer + 3 × seg

    Short-track fallback — total duration < threshold:
        The track is split into three equal thirds starting from 0; each
        segment starts at the beginning of its respective third.  All offsets
        are guaranteed to be within [0, total_secs), so librosa never seeks
        past EOF.  Segments that extend beyond the end of the file are
        zero-padded by process_catalog to the configured segment length.
    """
    seg    = float(SEGMENT_DURATION_SECS)   # configurable segment length
    buffer = float(SEGMENT_SECS)            # 30.0 — fixed boundary guard, separate concern

    threshold = 2.0 * buffer + 3.0 * seg

    if total_secs >= threshold:
        return {
            1: buffer,                                # [buffer → buffer+seg]
            2: total_secs / 2.0 - seg / 2.0,         # [mid−seg/2 → mid+seg/2]
            3: total_secs - buffer - seg,             # [total−buffer−seg → total−buffer]
        }

    # Short-track fallback: evenly divide the available duration into thirds.
    # third * 2 < total_secs is always true, so no offset exceeds the file.
    third = total_secs / 3.0
    return {
        1: 0.0,
        2: third,
        3: third * 2.0,
    }


def save_spectrogram(y: np.ndarray, sr: int, out_path: Path) -> None:
    """
    Convert a waveform array to a dB-scaled Mel-spectrogram and write a
    clean PNG with no axes, tick marks, titles, or whitespace padding.
    """
    S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=IMAGE_SIZE)
    ax.imshow(S_dB, aspect="auto", origin="lower", cmap="magma", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ── Path extraction helper ────────────────────────────────────────────────────

def _extract_network_path(line: str) -> Path | None:
    """
    Extract the first backtick-quoted UNC or Windows absolute path from a
    catalog line.  Returns None if no qualifying path token is found.

    Qualifying paths:
      • UNC shares    — start with two backslashes, e.g. ``\\GAMES\...``
      • Drive paths   — second character is ``:``,    e.g. ``C:\\...``

    This helper replaces the old PATH_RE approach.  It imposes no constraints
    on surrounding whitespace, bullet characters, em-dash suffixes, or
    indentation depth, so it captures valid paths regardless of minor
    formatting variations in the Markdown source.
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

    Path detection uses _extract_network_path so any backtick-quoted UNC or
    absolute path is accepted regardless of surrounding Markdown formatting.

    Rules applied per track:
      • A qualifying path line is found and the file exists on disk  → valid.
      • A qualifying path line is found but the file is absent       → error:
            "File not found on network: <path>"
      • A *[NO MATCH FOUND]* line is found before any path line     → error:
            "No file path in catalog (unmatched track)"
      • No qualifying path or NO MATCH line found anywhere under the
        track heading before the end of its category section        → error:
            "Missing path structure in catalog"

    Tracks seen in the document are collected in order; after the full
    pass, any track not resolved (path found or NO MATCH detected) is
    reported as "Missing path structure".

    Returns
    -------
    valid_entries : [(category, track_name, audio_path), ...]
    errors        : [(category, track_name, reason_string), ...]
    """
    valid_entries: list[tuple[str, str, Path]] = []
    errors:        list[tuple[str, str, str]]  = []

    current_category: str | None = None
    current_track:    str | None = None

    # Ordered list of every track heading seen (for end-of-pass unresolved check)
    all_track_keys:  list[tuple[str, str]] = []
    seen_track_keys: set[tuple[str, str]]  = set()
    # Tracks whose status has been determined (path found or NO MATCH detected)
    resolved_keys:   set[tuple[str, str]]  = set()

    with open(md_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()   # strips \n, \r, and trailing spaces

            # ── Category heading ───────────────────────────────────────────
            m = CATEGORY_RE.match(line)
            if m:
                current_category = m.group(1)
                current_track    = None
                continue

            # Any other ## heading (e.g. ## Summary) exits the category scope.
            # This prevents paths in the summary section being attributed to
            # the last energetic_upbeat track still in current_track.
            if line.startswith("## "):
                current_category = None
                current_track    = None
                continue

            # ── Track heading ──────────────────────────────────────────────
            m = TRACK_RE.match(line)
            if m:
                current_track = m.group(1)
                if current_category:
                    key = (current_category, current_track)
                    if key not in seen_track_keys:
                        all_track_keys.append(key)
                        seen_track_keys.add(key)
                continue

            # ── Content lines — only act when inside a known track ─────────
            if not current_category or not current_track:
                continue

            key = (current_category, current_track)

            # NO MATCH sentinel — only record the first occurrence per track
            if re.search(r"\[NO MATCH FOUND\]", line) and key not in resolved_keys:
                errors.append((
                    current_category, current_track,
                    "No file path in catalog (unmatched track)",
                ))
                resolved_keys.add(key)
                continue

            # Network / absolute path — only evaluate the first per track
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

    # Any track that was never resolved → no path and no NO MATCH line found
    for cat, trk in all_track_keys:
        if (cat, trk) not in resolved_keys:
            errors.append((cat, trk, "Missing path structure in catalog"))

    return valid_entries, errors


def run_verification_loop(md_path: Path) -> list[tuple[str, str, Path]]:
    """
    Run verify_catalog in a loop.  The menu is ALWAYS displayed after every
    scan so the user can review the full track list before committing to a
    potentially long generation run — even when zero errors are found.

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
                "  [1]  Skip the " + str(len(errors)) + " problematic track(s) and generate\n"
                "       spectrograms for the " + str(len(valid_entries)) + " valid tracks."
            )
        else:
            option1 = (
                "  [1]  Generate spectrograms for all "
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
    librosa, and save one PNG per segment into data/<category>/.

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
        entries, desc="Spectrograms", unit="track", dynamic_ncols=True
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

                # Peak normalization — scale so max absolute amplitude = 1.0.
                # Skipped for silent segments to avoid division-by-zero.
                peak = np.max(np.abs(y))
                if peak > 0.0:
                    y = y / peak

                out_path = dest_dir / (stem + "_" + str(idx) + ".png")
                save_spectrogram(y, sr, out_path)

            generated += 3

        except Exception as exc:
            tqdm.write("  [WARN] Skipped '" + track_name + "': " + str(exc))
            skipped += 1

    print("─" * 60)
    print("PNGs generated : " + str(generated))
    print("Tracks skipped : " + str(skipped))
    print("─" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("generate_spectrograms.py  —  catalog → spectrograms")
    print("=" * 60)
    print("Catalog  : " + str(CATALOG_PATH))
    print("Output   : " + str(DATA_DIR))
    print()

    if not CATALOG_PATH.exists():
        raise SystemExit(
            "ERROR: catalog_with_paths.md not found.\n"
            "Run prepare_dataset.py first to generate the catalog."
        )

    entries = run_verification_loop(CATALOG_PATH)

    process_catalog(entries)


if __name__ == "__main__":
    main()
