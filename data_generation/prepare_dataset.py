"""
prepare_dataset.py  —  Pipeline Step 1: catalog → match → report

Reads songs_catalog.md, searches the source music library for ALL matching
files for each listed track, and writes a human-readable report to
catalog_with_paths.txt in the project root.

No files are copied or modified.  The report is the only output.
Review it to resolve multiple versions before running the copy step.

Requires Python 3.7+
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import mutagen
from tqdm import tqdm

# ─── CONFIGURE THIS before running ────────────────────────────────────────────
MUSIC_LIBRARY_PATH = r"\\GAMES\Utils\iTunes\iTunes Media\Music"   # ← set this
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR       = Path(__file__).parent
CATALOG_PATH     = SCRIPT_DIR / "docs" / "songs_catalog.md"
REPORT_PATH      = SCRIPT_DIR / "docs" / "catalog_with_paths.md"

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac", ".wma", ".opus"}

HEADING_RE       = re.compile(r"^#\s+(calm_melancholic|moderate_neutral|energetic_upbeat)")
ARTIST_TITLE_RE  = re.compile(r"^(.+?)\s+-\s+(.+)$")   # handles multiple spaces around "-"

# Minimum SequenceMatcher ratio (or containment score) to accept a file match.
# Lower → more permissive; raise it if you see too many false positives.
MATCH_THRESHOLD  = 0.72


# ── Text normalisation ────────────────────────────────────────────────────────

def norm_ascii(text: str) -> str:
    """Lowercase, keep only ASCII alphanumerics (strips accents, punctuation, spaces)."""
    return re.sub(r"[^a-z0-9]", "", text.lower())

def norm_unicode(text: str) -> str:
    """Lowercase and collapse whitespace, keeping all Unicode characters.
    Used as a fallback for non-Latin scripts (Hebrew, etc.)."""
    return re.sub(r"\s+", "", text.lower())


# ── Duration extraction ───────────────────────────────────────────────────────

def get_duration(path: Path) -> str:
    """
    Return the playback duration of an audio file formatted as MM:SS.

    Uses mutagen.File() which supports mp3, m4a, flac, ogg, opus, wav, etc.
    Returns "[Unknown Duration]" if the file is corrupt, unreadable, or
    carries no length metadata, so the report never crashes on bad files.
    """
    try:
        audio = mutagen.File(path)
        if audio is not None and audio.info is not None:
            total_seconds = int(audio.info.length)
            return "{:02d}:{:02d}".format(total_seconds // 60, total_seconds % 60)
    except Exception:
        pass
    return "[Unknown Duration]"


# ── Catalog parsing ───────────────────────────────────────────────────────────

def parse_catalog(catalog_path: Path) -> dict[str, list[dict]]:
    """
    Parse songs_catalog.md and return:
        {category_name: [{"artist": str, "title": str, "raw": str}, ...]}

    Lines under a heading that contain " - " are split into artist / title.
    Lines without " - " (e.g. bare song titles) are stored with artist="".
    """
    categories: dict[str, list[dict]] = {}
    current: str | None = None

    with open(catalog_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            m = HEADING_RE.match(line)
            if m:
                current = m.group(1)
                categories[current] = []
                continue

            if current is None:
                continue

            m2 = ARTIST_TITLE_RE.match(line)
            if m2:
                artist = m2.group(1).strip()
                title  = m2.group(2).strip()
            else:
                artist = ""
                title  = line

            categories[current].append({"artist": artist, "title": title, "raw": line})

    return categories


# ── Library index ─────────────────────────────────────────────────────────────

def _artist_keys_from_path(p: Path, library_root: Path) -> list[str]:
    """
    Derive candidate normalized artist keys for a single audio file.

    Two sources are checked:

    1. Top-level subfolder relative to the library root.
       The overwhelming convention is Library/Artist/Album/Track.ext, so
       the first path component after the root is the artist name.

    2. Filename stem prefix for flat libraries where files are named
       "Artist - Title.ext" — everything before the first " - " is treated
       as the artist.

    Each candidate is emitted in both ASCII-only and full-Unicode
    normalisations so that non-Latin folder names (e.g. Hebrew) are
    still reachable via their Unicode key.
    """
    keys: set[str] = set()

    # Source 1: first directory component under the library root
    try:
        rel_parts = p.relative_to(library_root).parts
        if len(rel_parts) >= 2:          # at least one folder + filename
            folder_artist = rel_parts[0]
            ka = norm_ascii(folder_artist)
            ku = norm_unicode(folder_artist)
            if ka:
                keys.add(ka)
            if ku and ku != ka:
                keys.add(ku)
    except ValueError:
        pass

    # Source 2: "Artist - Title" stem prefix
    if " - " in p.stem:
        stem_artist = p.stem.split(" - ", 1)[0].strip()
        ka = norm_ascii(stem_artist)
        ku = norm_unicode(stem_artist)
        if ka:
            keys.add(ka)
        if ku and ku != ka:
            keys.add(ku)

    return list(keys)


def build_library_index(
    library_root: str,
) -> tuple[list[tuple[str, str, Path]], dict[str, list]]:
    """
    Walk the library recursively and return two structures:

    full_index
        list of (norm_ascii_stem, norm_unicode_stem, full_path) for every
        audio file — used as a fallback when artist pre-filtering yields
        no candidates.

    artist_index
        dict mapping a normalised artist key → list of index entries.
        Each file is registered under every artist key derivable from its
        path (folder name and/or filename prefix), in both ASCII and
        Unicode normalisations.
    """
    root = Path(library_root)
    if not root.exists():
        raise FileNotFoundError(f"Music library not found: {library_root!r}")

    full_index: list[tuple[str, str, Path]] = []
    artist_index: dict[str, list] = defaultdict(list)

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            p = Path(dirpath) / fname
            if p.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            entry = (norm_ascii(p.stem), norm_unicode(p.stem), p)
            full_index.append(entry)
            for key in _artist_keys_from_path(p, root):
                artist_index[key].append(entry)

    return full_index, artist_index


# ── Matching ──────────────────────────────────────────────────────────────────

def _similarity(query: str, ascii_stem: str, unicode_stem: str) -> float:
    """
    Return the best similarity score between `query` and a library-file stem.

    Two strategies are tried in both ASCII and Unicode normalisations:
      1. Containment check — the normalised query is a substring of the stem.
         This catches prefixes such as track numbers ("01 - Title") and gives
         a strong baseline score (0.90 for long queries, 0.80 for short ones).
      2. SequenceMatcher ratio — character-level similarity fallback.
    """
    q_ascii   = norm_ascii(query)
    q_unicode = norm_unicode(query)

    best = 0.0
    for q, s in ((q_ascii, ascii_stem), (q_unicode, unicode_stem)):
        if not q or not s:
            continue
        if q in s:
            # Short queries (< 6 chars) are more likely to collide accidentally.
            containment_score = 0.90 if len(q) >= 6 else 0.80
            best = max(best, containment_score)
        else:
            best = max(best, SequenceMatcher(None, q, s).ratio())
    return best


def find_all_matches(
    track: dict,
    full_index: list[tuple[str, str, Path]],
    artist_index: dict[str, list],
    threshold: float = MATCH_THRESHOLD,
) -> list[tuple[float, Path]]:
    """
    Return ALL library files whose best similarity score meets `threshold`,
    sorted by score descending.

    Search-space selection (the key optimisation):
      • Track has an artist → look up norm_ascii and norm_unicode keys in
        artist_index, merging both buckets.  This typically narrows 32 000+
        files down to the tens or hundreds belonging to that artist.
      • Artist bucket is empty (multi-artist string, Hebrew name, or artist
        not found in any folder/stem) → fall back to full_index so no
        track is silently dropped.
      • Track has no artist → always use full_index.

    Queries tried (richest first):
      • "Artist Title"   — joined without separator
      • "Artist - Title" — as written in the catalog
      • "Title"          — title alone (catches files missing an artist prefix)
    """
    artist = track["artist"]
    title  = track["title"]

    queries: list[str] = []
    if artist:
        queries.append(artist + " " + title)
        queries.append(artist + " - " + title)
    queries.append(title)

    # ── Select search space ───────────────────────────────────────────────────
    if artist:
        # Merge entries from ASCII and Unicode artist keys, deduplicating by
        # object identity so each entry is scored at most once.
        seen: set[int] = set()
        search_space: list[tuple[str, str, Path]] = []
        for key in (norm_ascii(artist), norm_unicode(artist)):
            for entry in artist_index.get(key, []):
                eid = id(entry)
                if eid not in seen:
                    seen.add(eid)
                    search_space.append(entry)
        if not search_space:
            # Artist not found in index — could be multi-artist, Hebrew, or
            # a flat library.  Fall back to the complete index.
            search_space = full_index
    else:
        search_space = full_index

    # ── Score and collect ─────────────────────────────────────────────────────
    results: list[tuple[float, Path]] = []
    for ascii_stem, uni_stem, filepath in search_space:
        best_score = 0.0
        for q in queries:
            s = _similarity(q, ascii_stem, uni_stem)
            if s > best_score:
                best_score = s
        if best_score >= threshold:
            results.append((best_score, filepath))

    results.sort(key=lambda x: x[0], reverse=True)
    return results


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(
    categories: dict[str, list[dict]],
    full_index: list[tuple[str, str, Path]],
    artist_index: dict[str, list],
    report_path: Path,
) -> None:
    """
    Write catalog_with_paths.md.  For each track list all matching library
    paths and durations using Markdown syntax; tracks with no match are
    clearly flagged.
    """
    total_tracks    = sum(len(t) for t in categories.values())
    total_matched   = 0
    total_unmatched = 0
    multi_version   = 0

    lines: list[str] = []

    # ── Document header ───────────────────────────────────────────────────────
    lines.append("# catalog_with_paths")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Generated | " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " |")
    lines.append("| Library | `" + MUSIC_LIBRARY_PATH + "` |")
    lines.append("| Catalog | `" + str(CATALOG_PATH) + "` |")
    lines.append("| Match threshold | " + str(MATCH_THRESHOLD) + " |")
    lines.append("")
    lines.append("---")

    # ── One section per category ──────────────────────────────────────────────
    for category, tracks in categories.items():
        lines.append("")
        lines.append("## " + category + " (" + str(len(tracks)) + " tracks)")
        lines.append("")

        for track in tqdm(tracks, desc=category, unit="track", dynamic_ncols=True):
            matches = find_all_matches(track, full_index, artist_index)

            # Track name as a bold bullet
            lines.append("- **" + track["raw"] + "**")

            if not matches:
                lines.append("  - *[NO MATCH FOUND]*")
                total_unmatched += 1
            else:
                total_matched += 1
                if len(matches) > 1:
                    multi_version += 1
                for score, path in matches:
                    duration = get_duration(path)
                    lines.append("  - `" + str(path) + "` — [" + duration + "]")

        lines.append("")
        lines.append("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---|")
    lines.append("| Total tracks | " + str(total_tracks) + " |")
    lines.append("| Matched | " + str(total_matched) + " |")
    lines.append("| No match | " + str(total_unmatched) + " |")
    lines.append("| Multiple versions | " + str(multi_version) + " |")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("─" * 60)
    print("Total tracks       : " + str(total_tracks))
    print("Matched            : " + str(total_matched))
    print("No match           : " + str(total_unmatched))
    print("Multiple versions  : " + str(multi_version))
    print("─" * 60)
    print("Report written     : " + str(report_path))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("prepare_dataset.py  —  catalog → match → report")
    print("=" * 60)
    print("Catalog  : " + str(CATALOG_PATH))
    print("Library  : " + MUSIC_LIBRARY_PATH)
    print("Report   : " + str(REPORT_PATH))
    print()

    if MUSIC_LIBRARY_PATH == r"C:\path\to\your\music\library":
        raise SystemExit(
            "ERROR: MUSIC_LIBRARY_PATH has not been set.\n"
            "Open prepare_dataset.py and update the variable at the top of the file."
        )

    categories = parse_catalog(CATALOG_PATH)
    total_tracks = sum(len(t) for t in categories.values())
    print("Parsed   " + str(total_tracks) + " tracks across " + str(len(categories)) + " categories.")

    full_index, artist_index = build_library_index(MUSIC_LIBRARY_PATH)
    print("Indexed  " + str(len(full_index)) + " audio files from library.")
    print("Buckets  " + str(len(artist_index)) + " unique artist keys in index.")
    print()

    generate_report(categories, full_index, artist_index, REPORT_PATH)


if __name__ == "__main__":
    main()
