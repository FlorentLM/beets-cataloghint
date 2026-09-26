"""
Beets plugin `cataloghint`: tries to narrow beets' MusicBrainz release-group match down to the
exact release, before the candidate list is shown to the user.

cataloghint scores release from a release-group against whatever is found in the album folder
(barcode, catalog number, disambiguation text, country, year)

- If exactly one release stands out, candidates are restricted to it and the recommendation is
  forced to strong.
- If none stands out but the folder text still shows beets' own top pick scores worse than
  another release in the same group, the recommendation is forced to none.
- A disc under a parent folder where sibling disc(s) have already resolved is matched that same release
  directly (as long as it still fits within the remaining multi-dic track count).
- Multi-discs per-disc track-counts veto any release that doesn't match, and strongly favours those that do.
"""
from __future__ import annotations
import difflib
import functools
import os
import re
import threading
from datetime import date
from typing import TYPE_CHECKING, Optional, Tuple
import mediafile
import requests
from beets import config, plugins
from beets.autotag import Recommendation
from beets.importer.state import ImportState
from beets.plugins import BeetsPlugin

from beetsplug.cataloghint.cuefiles import find_and_parse, has_cue

if TYPE_CHECKING:
    from beets.autotag.distance import Distance
    from beets.autotag.match import AlbumMatch
    from beets.importer import ImportSession, ImportTask
    from beetsplug.musicbrainz import MusicBrainzPlugin


# TODO: PR locking in beets itself
def _patch_import_state_locking() -> None:
    """Serialise Beets' ImportState's unlocked read-modify-write of state.pickle."""
    if getattr(ImportState, '_cataloghint_locked', False):
        return

    lock = threading.RLock()
    orig_open = ImportState._open
    orig_save = ImportState._save

    @functools.wraps(orig_open)
    def _open(self):
        with lock:
            return orig_open(self)

    @functools.wraps(orig_save)
    def _save(self):
        with lock:
            return orig_save(self)

    def _serialize(orig):
        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):
            with lock:
                self._open()
                return orig(self, *args, **kwargs)
        return wrapper

    ImportState._open = _open
    ImportState._save = _save
    ImportState.history_add = _serialize(ImportState.history_add)
    ImportState.progress_add = _serialize(ImportState.progress_add)
    ImportState.progress_reset = _serialize(ImportState.progress_reset)
    ImportState._cataloghint_locked = True


_patch_import_state_locking()


## Configs

MIN_OVERLAP = 5     # below this, a (non-year) shared digit run is more likely coincidence
PARTIAL_DISC_PENALTIES = {'missing_tracks'}     # penalties a partial disc incurs
LAYOUT_MATCH_BONUS = 2.5     # > than max match_score (2.0): a multi-disc layout match beats text matches   # TODO: Tune this?


## Compiled regexes

CATALOG_RE = re.compile(r"^(\d{12,13})$")
NORMALIZE_RE = re.compile(r"[\W_]")     # strip punctuation/symbols/whitespace

BRACKETED_RE = re.compile(r"[\[\(\{][^\[\]\(\)\{\}]*[\]\)\}]")

# Country: only used if a bracket/parenthese/brace/dash-delimited whole token is a real MB area code
COUNTRY_BRACKET_RE = re.compile(r"[-\[\(\{]\s*?([A-Za-z]{2}|USA)\s*?[-\]\)\}]")
COUNTRY_CODE_ALIASES = {'UK': 'GB', 'USA': 'US'}
COUNTRY_CODE_EXCLUDE = {'CD'}   # sorry Congo...
COUNTRY_CODES_MB = {
    'AF', 'AX', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ',
    'BS', 'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR',
    'IO', 'VG', 'BN', 'BG', 'BF', 'BI', 'KH', 'CM', 'CA', 'CV', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX',
    'CC', 'CO', 'KM', 'CG', 'CK', 'CR', 'CI', 'HR', 'CU', 'CW', 'CY', 'CZ', 'XC', 'CD', 'DK', 'DJ',
    'DM', 'DO', 'XG', 'EC', 'EG', 'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'XE', 'FK', 'FO', 'FM', 'FJ',
    'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU',
    'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 'HM', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE',
    'IM', 'IL', 'IT', 'JM', 'JP', 'JE', 'JO', 'KZ', 'KE', 'KI', 'XK', 'KW', 'KG', 'LA', 'LV', 'LB',
    'LS', 'LR', 'LY', 'LI', 'LT', 'LU', 'MO', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR',
    'MU', 'YT', 'MX', 'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'AN',
    'NC', 'NZ', 'NI', 'NE', 'NG', 'NU', 'NF', 'MP', 'KP', 'MK', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA',
    'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 'PT', 'PR', 'QA', 'RE', 'RO', 'RU', 'RW', 'BL', 'SH', 'KN',
    'LC', 'MF', 'PM', 'VC', 'WS', 'SM', 'ST', 'SA', 'SN', 'RS', 'CS', 'SC', 'SL', 'SG', 'SX', 'SK',
    'SI', 'SB', 'SO', 'ZA', 'GS', 'KR', 'SS', 'SU', 'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY',
    'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA',
    'AE', 'GB', 'US', 'UM', 'UY', 'VI', 'UZ', 'VU', 'VA', 'VE', 'VN', 'WF', 'EH', 'XW', 'YE', 'YU',
    'ZM', 'ZW',
}

# An isolated 4-digit number: plausible as a release year (original, reissue/remaster, ...)
YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")

DISC_WORD_NUMS = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20
}

# Classic tokens found in disc-only folder names ("Disc 1", "CD2", "Bonus Disc", "B-Sides"...)
# cd/disc/vinyl only count as a token if followed by a number (digit or spelled-out)
DISC_TOKEN_RE = re.compile(
    r"(?i)\b(?:cd|dis[ck]|vinyl|d)\s*(?P<num>\d{1,2}(?!\d)|" + '|'.join(DISC_WORD_NUMS) + r")\b"
    r"|\b(?:dvd|bonus|b-?sides?|remix(es)?|album|vol(?:ume)?|part|pt)(?![a-z])"
)

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

DISC_NAME_MAX_LEN = 10

AUDIO_EXTENSIONS = {f'.{ext}' for ext in mediafile.TYPES}

# MusicBrainz medium formats
CD_FORMATS = frozenset({
    'CD', 'CD-R', 'Data CD', 'Enhanced CD', 'Copy Control CD', 'HDCD', 'DTS CD', 'Mixed Mode CD', 'HQCD', 'CD+G',
    'SHM-CD', 'Blu-spec CD', 'SACD', 'Hybrid SACD', 'SHM-SACD', 'DualDisc', 'CD-LE', '8cm CD', 'Minimax CD', '8cm CD+G',
    'MiniDisc', 'Blu-ray', 'Blu-ray-R', 'VCD'
})
VINYL_FORMATS = frozenset({'Vinyl', '7" Vinyl', '10" Vinyl', '12" Vinyl', 'Flexi-disc', '7" Flexi-disc', '12" Flexi-disc'})
DIGITAL_FORMATS = frozenset({'Digital Media', 'Download Card'})
CASSETTE_FORMATS = frozenset({'Cassette', 'Microcassette', 'VHS'})
DVD_FORMATS = frozenset({'DVD', 'DVD-Audio', 'DVD-Video', 'HD-DVD'})

MEDIA_HINT_RE = re.compile(r'(?i)\b(cd|vinyl|web|cassette|dvd)\b')
MEDIA_HINT_FORMATS: dict[str, frozenset[str]] = {
    'cd': CD_FORMATS, 'vinyl': VINYL_FORMATS, 'web': DIGITAL_FORMATS, 'cassette': CASSETTE_FORMATS, 'dvd': DVD_FORMATS,
}


## Custom error

class MusicBrainzUnavailable(Exception):
    """A MusicBrainz lookup couldn't complete (already exhausted the HTTP client's own retries)."""



## Helpers

def task_source(task: ImportTask) -> Tuple[Optional[str], Optional[str]]:
    """
    Artist, and album beets guessed.
    (beets < 2.14 doesn't have `task.source`)
    """
    source = getattr(task, 'source', None)
    if source is not None:
        return source.artist, source.name
    return task.cur_artist, task.cur_album


def normalize(s: str) -> str:
    return NORMALIZE_RE.sub("", s).casefold()


def longest_overlap(a: str, b: str) -> str:
    """Longest run `a` and `b` have in common (anywhere in either string)."""
    if not a or not b:
        return ''
    match = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return a[match.a:match.a + match.size]


def build_strip_pattern(value: str) -> re.Pattern | None:
    """Case-insensitive, word-bounded pattern matching any token of `value`."""
    tokens = set(re.findall(r'\w+', value, re.UNICODE))
    if not tokens:
        return None
    body = '|'.join(re.escape(t) for t in tokens)
    return re.compile(rf'(?i)(?<!\w)(?:{body})(?!\w)')


def looks_like_disc(folder: str, artist: Optional[str] = None, album: Optional[str] = None) -> Tuple[bool, Optional[int]]:
    """Whether `folder` looks like a single disc name (not a release folder)."""

    if artist and (pattern := build_strip_pattern(artist)):
        folder = pattern.sub(' ', folder)

    if album and (pattern := build_strip_pattern(album)):
        folder = pattern.sub(' ', folder)

    if folder.strip().isdigit() and 0 < int(folder.strip()) < 20:
        return True, int(folder.strip())

    token_matches = list(re.finditer(DISC_TOKEN_RE, folder))
    numbers = [
        int(n) if n.isdigit() else DISC_WORD_NUMS[n.lower()]
        for n in (m.group('num') for m in token_matches)
        if n
    ]

    folder = DISC_TOKEN_RE.sub(' ', folder)
    folder = YEAR_RE.sub(' ', folder)     # just a year isn't disc name evidence
    folder = PUNCT_RE.sub('', folder).strip()

    if not folder and not numbers and not token_matches:
        # Nothing but artist/album/year/punctuation: this is a decorated album folder, not a disc name
        return False, None

    if len(folder) > DISC_NAME_MAX_LEN:
        return False, None

    if len(numbers) == 1:
        return True, numbers[0]

    return True, None


def match_score(needle: str, haystack_exact: str, haystack_loose: str) -> Tuple[float, str]:
    """
    How well `needle` (a release's barcode/catalog/disambiguation string) is covered by the
    haystack (stuff found in the folder name / cue file), as a score in [0, 2].

    Up to 1.0: How much of `needle` is found with punctuation/spacing is stripped.
    Up to another 1.0: Same coverage but with the original formatting preserved:
        an exact unstripped hit is much less likely to be a coincidence than one that only lines up after normalizing.
    """

    if not needle:
        return 0.0, ''

    needle_loose = normalize(needle)
    overlap_loose = longest_overlap(needle_loose, haystack_loose)
    if not needle_loose or len(overlap_loose) < MIN_OVERLAP:
        return 0.0, ''

    coverage_loose = len(overlap_loose) / len(needle_loose)

    needle_exact = needle.casefold()
    overlap_exact = longest_overlap(needle_exact, haystack_exact)
    coverage_exact = len(overlap_exact) / len(needle_exact) if needle_exact else 0.0

    matched = overlap_exact if len(overlap_exact) >= len(overlap_loose) else overlap_loose
    return coverage_loose + coverage_exact, matched


def strip_outside_brackets(text: str, pattern: re.Pattern) -> str:
    """
    Apply a regex sub `pattern` everywhere in `text` except inside brackets/parentheses/braces.
    """
    parts = []
    pos = 0
    for m in BRACKETED_RE.finditer(text):
        parts.append(pattern.sub(' ', text[pos:m.start()]))
        parts.append(m.group(0))
        pos = m.end()
    parts.append(pattern.sub(' ', text[pos:]))
    return ''.join(parts)


def validate_preferred_countries(codes: list[str]) -> list[str]:
    """Validate user-supplied country codes."""
    seen = set()
    result = []
    for code in codes:
        aliased = COUNTRY_CODE_ALIASES.get(code.upper(), code.upper())
        if aliased in COUNTRY_CODES_MB and aliased not in seen:
            seen.add(aliased)
            result.append(aliased)
    return result


def extract_years(text: str) -> set[int]:
    """
    All plausible release years found in `text`.
    (it's very rarely more than 2: earliest = original release, later one = reissue/remaster).
    """
    current_year = date.today().year
    return {
        year for match in YEAR_RE.findall(text)
        if 1800 <= (year := int(match)) <= current_year + 1
    }


def extract_media_hint(text: str) -> Optional[frozenset[str]]:
    """The medium format (CD/Vinyl/WEB/Cassette) if unambiguous."""
    buckets = {MEDIA_HINT_FORMATS[m.group(1).lower()] for m in MEDIA_HINT_RE.finditer(text)}
    return next(iter(buckets)) if len(buckets) == 1 else None


def hit_track_counts(hit: dict) -> dict[int, int]:
    """A MusicBrainz release's track count, per disc."""
    counts = {}
    for medium in hit.get('media') or []:
        try:
            position = int(medium['position'])
            count = int(medium['track_count'])
        except (KeyError, TypeError, ValueError):
            continue
        if medium.get('pregap'):    # hidden pregap track
            count += 1
        counts[position] = count
    return counts


def layout_verdict(
        hit: dict,
        disc_layout: Optional[dict[int, int]] = None,
        local_total: Optional[int] = None,
    ) -> Optional[bool]:
    """
    Whether the local vs MusicBrainz track counts match. Pass exactly one of:

    Local count can never exceed MusicBrainz's count, and MusicBrainz reporting more
    tracks/discs than found locally is plausible (a bonus DVD or data track never ripped as audio).

    True if every disc counted (or the whole-folder total) is an exact fit, False if any is
    impossible, None otherwise.
    """
    hit_counts = hit_track_counts(hit)
    if not hit_counts:
        return None

    if local_total is not None:
        total = sum(hit_counts.values())
        if local_total > total:
            return False
        return True if local_total == total else None

    if not disc_layout:
        return None

    max_disc = max(hit_counts)
    known = exact_matches = 0
    for disc_number, on_disk_count in disc_layout.items():
        if disc_number > max_disc:
            return False

        hit_count = hit_counts.get(disc_number)
        if hit_count is None:
            continue
        known += 1
        if on_disk_count > hit_count:
            return False
        if on_disk_count == hit_count:
            exact_matches += 1

    if known == 0:
        return None
    return True if exact_matches == known else None


def hit_formats(hit: dict) -> set[str]:
    """A MusicBrainz release's medium formats."""
    return {medium['format'] for medium in hit.get('media') or [] if medium.get('format')}


def format_veto(hit: dict, target_formats: frozenset[str]) -> bool:
    """True if none of this release's media are a format in `target_formats`."""
    formats = hit_formats(hit)
    return bool(formats) and formats.isdisjoint(target_formats)


def hit_year(hit: dict) -> Optional[int]:
    """A MusicBrainz release's year."""
    if match := YEAR_RE.match(str(hit.get('date') or '')):
        return int(match.group(1))
    return None


def core_distance(dist: Distance) -> float:
    """Beets' distance (ignoring penalties a partial disc legitimately has)."""
    raw = max_raw = 0.0
    for key, penalty in dist._penalties.items():
        if key in PARTIAL_DISC_PENALTIES:
            continue
        weight = dist._weights[key]
        raw += sum(penalty) * weight
        max_raw += len(penalty) * weight
    return raw / max_raw if max_raw else 0.0


def is_plausible(match: AlbumMatch, max_distance: float) -> bool:
    """Whether `match` fits to the local files enough to be trusted over beets' recommendation."""
    return core_distance(match.distance) <= max_distance


## Classes

class TaskLog:
    """Wraps Beets' logger, tags every message with the album's folder name."""

    def __init__(self, log, tag: str) -> None:
        self._log = log
        self._tag = tag.replace('{', '{{').replace('}', '}}')

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log.debug(f'[{self._tag}] {msg}', *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log.warning(f'[{self._tag}] {msg}', *args, **kwargs)


class DummyLog:
    """Logger that does not forward to Beets (for tests)."""

    def debug(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass


dummy_log = DummyLog()


## Core functions

def gather_haystack(
        item_dir: str,
        artist: Optional[str],
        album: Optional[str],
        check_cue: bool = True,
        # TODO: Maybe check .log and .m3u files?
        filenames: Optional[set[str]] = None,
    ) -> Tuple[str, str, set[str], set[int], Optional[frozenset[str]]]:
    """
    Build the haystack: All the hints derived for `item_dir`, with the `artist`/`album`
    stripped out of it. If the folder itself looks like a bare disc name, the parent
    folder's name is folded in too.

    Returns two haystack forms: "exact" which only case-folds and "loose" which also strips
    punctuation/whitespace, any bracketed country code(s) and plausible year(s) found, and the
    medium format (CD/Vinyl/Digital/Cassette).
    """

    raw_folder = os.path.basename(item_dir)

    is_disc, disc_number = looks_like_disc(raw_folder, artist, album)
    if is_disc:
        if parent := os.path.basename(os.path.dirname(item_dir)):
            raw_folder = f'{parent} {raw_folder}'

    if check_cue:
        cue_content = find_and_parse(item_dir, filenames=filenames)
        if cue_content:
            for key in ('TITLE', 'PERFORMER'):
                if value := cue_content.get(key):
                    raw_folder += ' ' + str(value)
            if (catalog := cue_content.get('CATALOG')) and CATALOG_RE.match(str(catalog).strip()):
                raw_folder += ' ' + str(catalog)

    # Country codes: only bracket/dash-delimited tokens that are real MB area codes (excluding "CD")
    folder_countries = {
        aliased for code in COUNTRY_BRACKET_RE.findall(raw_folder)
        if (aliased := COUNTRY_CODE_ALIASES.get(code.upper(), code.upper())) in COUNTRY_CODES_MB
        and aliased not in COUNTRY_CODE_EXCLUDE
    }

    for known in (artist, album):
        if known and (pattern := build_strip_pattern(known)):
            raw_folder = strip_outside_brackets(raw_folder, pattern)

    folder_years = extract_years(raw_folder)
    media_hint = extract_media_hint(raw_folder)

    return raw_folder.casefold(), normalize(raw_folder), folder_countries, folder_years, media_hint


def count_audio_files(directory: str) -> int:
    try:
        with os.scandir(directory) as entries:
            return sum(
                1 for e in entries
                if e.is_file() and os.path.splitext(e.name)[1].lower() in AUDIO_EXTENSIONS
            )
    except OSError:
        return 0


def gather_disc_layout(item_dir: str, artist: Optional[str], album: Optional[str]) -> dict[int, int]:
    """
    Track count per disc number, counted from every sibling folder under `item_dir`'s parent
    that look like a numbered disc (so, including discs not yet imported).
    """
    layout: dict[int, int] = {}
    try:
        siblings = list(os.scandir(os.path.dirname(item_dir)))
    except OSError:
        return layout

    for entry in siblings:
        if not entry.is_dir():
            continue
        is_disc, disc_number = looks_like_disc(entry.name, artist, album)
        if is_disc and disc_number is not None:
            layout[disc_number] = count_audio_files(entry.path)

    return dict(sorted(layout.items()))


def gather_needles(hit: dict) -> list[str]:
    """A hit's barcode/disambiguation/catalog-number values, deduplicated."""
    needles = []
    if barcode := hit.get('barcode'):
        # UPC-A/EAN-13/GTIN-14 pad the same code with leading zeros, ignore that
        needles.append(barcode.lstrip('0') or barcode)
    if disambiguation := hit.get('disambiguation'):
        needles.append(disambiguation)

    seen = set()
    for label_info in hit.get('label_info', []):
        if (catno := label_info.get('catalog_number')) and (key := normalize(catno)) not in seen:
            seen.add(key)
            needles.append(catno)

    return needles


def score_hits(
        hits: list[dict],
        folder_exact: str,
        folder_loose: str,
        folder_countries: set[str],
        folder_years: set[int],
        disc_layout: Optional[dict[int, int]] = None,
        target_formats: Optional[frozenset[str]] = None,
        preferred_countries: Optional[list[str]] = None,
        log=None,
        local_total: Optional[int] = None,
    ) -> Tuple[list[str], list[Tuple[str, float, str, bool, bool]]]:
    """
    Score hits against text/country/year by barcode/catalog/disambiguation coverage.
    Substring that recurs across several releases in the group: score divided by how many releases have it.
    A release contradicted by non-matching disc layout, or target format is vetoed.
    One with a fully matching disc layout or, lacking that, a whole-folder track count that matches exactly, gets a bonus.
    A text match that's shared with another release in the group (a reused disambiguation label, for instance) is
    weak evidence: a differently-dated survivor whose year matches an explicit folder year outranks it instead.
    Finally if several releases are still tied, `preferred_countries` preference order breaks the tie.
    Returns the selected ids and the score per hit.
    """

    logger = log or dummy_log

    if disc_layout:
        verdicts = [layout_verdict(hit, disc_layout=disc_layout) for hit in hits]
    elif local_total:
        verdicts = [layout_verdict(hit, local_total=local_total) for hit in hits]
    else:
        verdicts = [None] * len(hits)

    # A verdict shared by every hit tells us nothing (e.g. every release has the exact same track count)
    layout_discriminates = len(set(verdicts)) > 1

    survivor_ids = {hit['id'] for hit, v in zip(hits, verdicts) if v is not False}
    if not survivor_ids:
        # On-disk track counts contradict every release in the group: data isn't trustworthy here
        survivor_ids = {hit['id'] for hit in hits}
        verdicts = [None] * len(hits)

    if target_formats:
        format_survivors = {hit['id'] for hit in hits if not format_veto(hit, target_formats)}
        if format_survivors:
            survivor_ids = survivor_ids & format_survivors or format_survivors

    if len(survivor_ids) == 1:
        release_id = next(iter(survivor_ids))
        logger.debug('{0!r} is the only release left after the disc-layout/medium-format checks', release_id)
        scored = [
            (hit['id'], LAYOUT_MATCH_BONUS if hit['id'] in survivor_ids else -1.0, '', False, False)
            for hit in hits
        ]
        return [release_id], scored

    hit_needles = [gather_needles(hit) for hit in hits]
    hit_looses = [{loose for v in needles if (loose := normalize(v))} for needles in hit_needles]

    def commonality(substring: str) -> int:
        return sum(1 for looses in hit_looses if any(substring in loose for loose in looses))

    scored: list[Tuple[str, float, str, bool, bool]] = []
    diluted_ids: set[str] = set()
    for hit, needles, verdict in zip(hits, hit_needles, verdicts):

        if hit['id'] not in survivor_ids:
            scored.append((hit['id'], -1.0, '', False, False))
            continue

        text_score = 0.0
        matched_text = ''
        diluted = False
        for needle in needles:
            score, matched = match_score(needle, folder_exact, folder_loose)
            if score <= 0:
                continue

            shared_by = commonality(normalize(matched))
            if shared_by > 1:
                score /= shared_by

            if score > text_score:
                text_score = score
                matched_text = matched
                diluted = shared_by > 1

        if diluted:
            diluted_ids.add(hit['id'])

        if verdict is True and layout_discriminates:
            text_score += LAYOUT_MATCH_BONUS

        country_match = bool(hit.get('country')) and hit['country'] in folder_countries

        year = hit_year(hit)
        year_match = year is not None and year in folder_years

        logger.debug('  scoring {0} country={1!r} year={2} text_score={3:.2f} matched={4!r} '
                  'country_match={5} year_match={6} layout_match={7}',
                  hit['id'], hit.get('country'), year, text_score, matched_text or None,
                  country_match, year_match, verdict is True)

        scored.append((hit['id'], text_score, matched_text, country_match, year_match))

    best_score = max((score for _, score, _, _, _ in scored), default=0.0)

    if best_score > 0:
        matches = {release_id for release_id, score, _, _, _ in scored if abs(score - best_score) < 1e-9}

        if len(matches) > 1:
            # Same top text score (e.g. identical regional pressings sharing one barcode)? Year and/or country can break tie
            for wants_country in (True, False):
                narrowed = {
                    release_id for release_id, _, _, country_match, year_match in scored
                    if release_id in matches and (country_match if wants_country else year_match)
                }
                if len(narrowed) == 1:
                    matches = narrowed
                    break
    else:
        matches = {
            release_id for release_id, _, _, country_match, year_match in scored
            if country_match or year_match
        }

    if not matches and len(survivor_ids) < len(hits):
        matches = survivor_ids

    if len(matches) == 1 and next(iter(matches)) in diluted_ids:
        # The unique winner only got there on a disambiguation label reused by another release: weak evidence
        # If exactly one survivor has a year that fits, prefer that one
        year_matches = {release_id for release_id, _, _, _, year_match in scored if year_match}
        if len(year_matches) == 1 and year_matches != matches:
            matches = year_matches

    preferred_country_tiebreak = False
    if len(matches) > 1 and preferred_countries:
        hits_by_id = {hit['id']: hit for hit in hits}
        ranked = {
            release_id: preferred_countries.index(country)
            for release_id in matches
            if (country := hits_by_id[release_id].get('country')) in preferred_countries
        }
        if ranked:
            best_rank = min(ranked.values())
            narrowed = {release_id for release_id, rank in ranked.items() if rank == best_rank}
            if len(narrowed) == 1:
                matches = narrowed
                preferred_country_tiebreak = True

    if not matches:
        return [], scored

    if len(matches) > 1:
        tied_ids = [hit['id'] for hit in hits if hit['id'] in matches]
        logger.debug('{0} releases tied, none stands out on its own: {1!r}', len(tied_ids), tied_ids)
        return tied_ids, scored

    release_id = next(iter(matches))
    _, score, matched_text, country_match, year_match = next(s for s in scored if s[0] == release_id)

    if preferred_country_tiebreak:
        reason = 'preferred country'
    elif score == best_score and score > 0:
        reason = f'text overlap {matched_text!r}'
    elif year_match:
        reason = 'year match'
    else:
        reason = 'country match'

    logger.debug('{0!r} stands out via {1}', release_id, reason)
    return [release_id], scored


def resolve_release(
        hits: list[dict],
        item_dir: str,
        artist: Optional[str],
        album: Optional[str],
        check_cue: bool = True,
        filenames: Optional[set[str]] = None,
        preferred_countries: Optional[list[str]] = None,
        log=None,
    ) -> Tuple[list[str], list[Tuple[str, float, str, bool, bool]]]:
    """
    Pick the release(s) in `hits` (from a given release-group) that the hints point to.

    Returns the matching ids, plus the per-hit score so external callers can still tell a clearly
    worse release from an untried one even if no single hit stood out.
    """

    logger = log or dummy_log

    if len(hits) == 1:
        logger.debug('only release in this release-group, trusting it: {0!r}', hits[0]['id'])
        return [hits[0]['id']], []

    folder_exact, folder_loose, folder_countries, folder_years, media_hint = gather_haystack(
        item_dir, artist, album, check_cue, filenames
    )

    is_disc, disc_number = looks_like_disc(os.path.basename(item_dir), artist, album)
    disc_layout = gather_disc_layout(item_dir, artist, album) if is_disc and disc_number is not None else {}

    on_disk_total = None if is_disc else count_audio_files(item_dir) or None

    # A cue file is evidence of a CD-like medium
    hinted_formats = {media_hint} if media_hint else set()
    if check_cue and has_cue(item_dir):
        hinted_formats.add(CD_FORMATS)
    target_formats = next(iter(hinted_formats)) if len(hinted_formats) == 1 else None

    logger.debug('{0} releases in group, folder hint = {1!r}, country code(s) = {2}, year(s) = {3}, '
              'disc layout = {4}, local track count = {5}, target medium = {6}',
              len(hits), folder_loose, folder_countries or None, folder_years or None,
              disc_layout or None, on_disk_total, sorted(target_formats) if target_formats else None)

    return score_hits(
        hits, folder_exact, folder_loose, folder_countries, folder_years, disc_layout, target_formats,
        preferred_countries, logger, on_disk_total,
    )


class CatalogHintPlugin(BeetsPlugin):

    def __init__(self):
        super().__init__()

        self.config.add(
            {
                'check_cue': True,
                'check_sibling_discs': True,
                'auto_apply': False,
                'preferred_countries': [],
            }
        )
        self.register_listener('import_task_before_choice', self.before_choice)
        self.register_listener('import_task_choice', self._record_manual_choice)
        self.register_listener('import_task_apply', self._record_incremental_history)

        # parent directory -> [release id, (artist, album) identity, tracks found so far, release's total track count]
        self._sibling_releases: dict[str, list] = {}

        self._counted_dirs: set[bytes] = set()

        # Directories for which cataloghint resolved a release but are still not applied by user
        self._pending_history_dirs: set[bytes] = set()

    def _musicbrainz(self) -> MusicBrainzPlugin | None:
        from beetsplug.musicbrainz import MusicBrainzPlugin

        for plugin in plugins.find_plugins():
            if isinstance(plugin, MusicBrainzPlugin):
                return plugin

        return None

    def before_choice(self, session: ImportSession, task: ImportTask) -> AlbumMatch | None:

        if not task.is_album or not task.items:
            return None

        mb = self._musicbrainz()
        if mb is None:
            self._log.warning('musicbrainz plugin not loaded, skipping')
            return None

        album_dir = os.path.dirname(os.fsdecode(task.items[0].path))
        parent_dir = os.path.dirname(album_dir)

        log = TaskLog(self._log, os.path.basename(album_dir))

        # Sibling discs matching relies on this
        src_artist, src_album = task_source(task)
        identity = (normalize(src_artist or ''), normalize(src_album or ''))
        log.debug('examining {0!r} (artist={1!r}, album={2!r})',
                  album_dir, src_artist, src_album)

        if self.config['check_sibling_discs'].get(bool):
            sibling = self._sibling_releases.get(parent_dir)

            if sibling and identity != ('', '') and sibling[1] == identity:
                release_id, _, tracks_so_far, total_tracks = sibling

                if tracks_so_far + len(task.items) <= total_tracks:
                    log.debug('reusing {0!r} for sibling disc under {1!r}', release_id, parent_dir)
                    return self._inject(task, release_id, parent_dir, identity, log)

                log.debug('sibling resolution {0!r} under {1!r} would exceed its track count, '
                          'skipping', release_id, parent_dir)

        release_group_id = self._release_group(task)
        if release_group_id is None:
            log.debug('beets did not resolve a release-group, nothing to narrow down')
            return None

        try:
            hits = self._browse_releases(mb, release_group_id, log)

        except MusicBrainzUnavailable as e:
            # Couldn't run the check, there's no basis to vouch for whatever beets' candidate is.
            # -> Force a low recommendation
            log.warning('MusicBrainz unavailable ({0}), cannot verify this release - '
                        'forcing manual review instead of trusting beets\' own guess', e)
            task.rec = Recommendation.none
            return None

        if not hits:
            log.debug('release-group {0!r} has no releases?', release_group_id)
            return None

        release_ids, scored = self._resolve_release(hits, task, log)

        if len(release_ids) == 1:
            log.debug('resolved release {0!r}', release_ids[0])
            return self._inject(task, release_ids[0], parent_dir, identity, log)

        if len(release_ids) > 1:
            log.debug('{0} releases tied ({1!r}), narrowing beets\' candidates to those.',
                      len(release_ids), release_ids)
            self._inject_shortlist(task, release_ids, log)
            return None

        log.debug('{0} release(s) in group, none stood out, deferring to beets', len(hits))
        self._flag_if_outscored(task, scored, log)
        return None

    def _release_group(self, task: ImportTask) -> str | None:
        """
        The release-group id beets' pre-injection candidate search resolved.
        """
        for c in task.candidates or []:
            if rg_id := getattr(c.info, 'releasegroup_id', None):
                return rg_id

        return None

    def _browse_releases(self, mb: MusicBrainzPlugin, release_group_id: str, log: TaskLog) -> list[dict]:
        """
        Every release in a release-group.
        """
        try:
            return mb.mb_api._browse(
                'release', **{'release-group': release_group_id},
                includes=[
                    'labels',
                    'media',
                    'recordings'    # needed for hidden pregap tracks   # TODO: This might be making the requests too slow?
                ],
                limit=100,
            )
        except requests.exceptions.RequestException as e:
            raise MusicBrainzUnavailable(f'release-group {release_group_id}: {e}') from e

    def _resolve_release(self,
            hits: list[dict],
            task: ImportTask,
            log: TaskLog
        ) -> Tuple[list[str], list[Tuple[str, float, str, bool, bool]]]:

        item_dir = os.path.dirname(os.fsdecode(task.items[0].path))
        filenames = {os.path.basename(os.fsdecode(item.path)) for item in task.items}

        preferred_countries = validate_preferred_countries(self.config['preferred_countries'].as_str_seq())
        src_artist, src_album = task_source(task)

        return resolve_release(
            hits, item_dir, src_artist, src_album,
            check_cue=self.config['check_cue'].get(bool), filenames=filenames,
            preferred_countries=preferred_countries, log=log,
        )

    def _flag_if_outscored(self,
            task: ImportTask,
            scored: list[Tuple[str, float, str, bool, bool]],
            log: TaskLog
        ) -> None:
        """
        No release stands out enough to inject. But beets' default pick might be bad too, and we
        might have positive evidence that it is a *worse* match than any other release in the same group.
            -> Force it to a low recommendation so it isn't auto-applied unreviewed
        """

        if not task.candidates:
            return

        best_score = max((score for _, score, _, _, _ in scored), default=0.0)
        if best_score <= 0:
            return

        scores_by_id = {release_id: score for release_id, score, _, _, _ in scored}
        top_id = task.candidates[0].info.album_id
        top_score = scores_by_id.get(top_id)

        if top_score is not None and top_score < best_score - 1e-9:
            log.warning("beets' own top candidate {0!r} only scores {1:.2f} against the folder, "
                        "but {2} other candidate(s) score higher (best {3:.2f}) -> forcing low recommendation",
                        top_id, top_score,
                        sum(1 for s in scores_by_id.values() if abs(s - best_score) < 1e-9), best_score)
            task.rec = Recommendation.none

    def _inject(self,
            task: ImportTask,
            release_id: str,
            parent_dir: str,
            identity: Tuple[str, str],
            log: TaskLog
        ) -> AlbumMatch | None:
        """
        Restrict candidates to `release_id` and force recommendation to strong, independent of
        beets' distance (partial discs *need* to have large distance against the whole multi-disc
        release).
        """

        original_candidates, original_rec = task.candidates, task.rec

        task.lookup_candidates(search_ids=[release_id])
        if not task.candidates:
            log.warning('resolved release {0!r} produced no usable candidate, reverting', release_id)
            task.candidates, task.rec = original_candidates, original_rec
            return None

        if not self._check_plausible(task, log):
            task.candidates, task.rec = original_candidates, original_rec
            return None

        total_tracks = len(task.candidates[0].info.tracks)

        sibling = self._sibling_releases.get(parent_dir)
        same_group = bool(sibling) and identity != ('', '') and sibling[1] == identity

        if same_group and sibling[0] == release_id:
            tracks_so_far = sibling[2] + len(task.items)
        else:
            if same_group and sibling[0] != release_id:
                # Could be an earlier mismatch, or a genuine reissue mix-up
                log.warning('{0!r} previously resolved to {1!r}, but this one '
                            'resolved to {2!r}: double check if these should match',
                            parent_dir, sibling[0], release_id)

            tracks_so_far = len(task.items)

        self._sibling_releases[parent_dir] = [release_id, identity, tracks_so_far, total_tracks]
        self._counted_dirs.add(os.path.dirname(task.items[0].path))

        log.debug('{0} usable candidate(s) after restricting to {1!r} -> recommendation set to strong, '
                  'track tally under {2!r}: {3}/{4} tracks',
                  len(task.candidates), release_id, parent_dir, tracks_so_far, total_tracks)

        # A disc merged into an already-imported release overwrites the incremental-import history,
        # and the previous disk will thus pop up again at next incremental import
        self._pending_history_dirs.add(os.path.dirname(task.items[0].path))

        task.rec = Recommendation.strong
        if self.config['auto_apply'].get(bool):
            return task.candidates[0]

        return None

    def _inject_shortlist(self, task: ImportTask, release_ids: list[str], log: TaskLog) -> None:
        """
        Several releases are equally plausible: restrict beets' candidates to those
        and let beets' distance scoring rank/recommend them.
        """

        original_candidates, original_rec = task.candidates, task.rec

        task.lookup_candidates(search_ids=release_ids)
        if not task.candidates:
            log.warning('shortlist {0!r} produced no usable candidate, reverting', release_ids)
            task.candidates, task.rec = original_candidates, original_rec
            return

        if not self._check_plausible(task, log):
            task.candidates, task.rec = original_candidates, original_rec
            return

        log.debug('{0} usable candidate(s) after narrowing to the tied releases, '
                  'beets recommendation: {1}', len(task.candidates), task.rec)

    def _check_plausible(self, task: ImportTask, log: TaskLog) -> bool:
        """Guard against wrong release-group: best candidate must fit the local files."""
        max_distance = config['match']['medium_rec_thresh'].as_number()
        best = task.candidates[0]
        if is_plausible(best, max_distance):
            return True

        log.warning('{0!r} ({1} - {2}) is too distant from the local files ({3:.2f} > {4:.2f}): '
                    'wrong release-group? reverting to beets\' original candidates',
                    best.info.album_id, best.info.artist, best.info.album,
                    core_distance(best.distance), max_distance)
        return False

    def _record_incremental_history(self, session: ImportSession, task: ImportTask) -> None:
        """
        Record cataloghint-resolved directories in beets' incremental import history.
        (circumvents beets' incremental history bug)
        """
        if not self._pending_history_dirs:
            return

        matched = {d for d in self._pending_history_dirs if d in task.paths}
        for directory in matched:
            ImportState().history_add([directory])
            self._log.debug('recorded {0!r} in incremental import history', os.fsdecode(directory))

        self._pending_history_dirs -= matched

    def _record_manual_choice(self, session: ImportSession, task: ImportTask) -> None:
        """
        Record manually selected releases so other disc subfolders can inherit them.
        """
        if not task.is_album or not task.apply or task.match is None:
            return

        counted_dir = os.path.dirname(task.items[0].path)
        if counted_dir in self._counted_dirs:
            return

        src_artist, src_album = task_source(task)
        identity = (normalize(src_artist or ''), normalize(src_album or ''))
        if identity == ('', ''):
            return

        album_dir = os.path.dirname(os.fsdecode(task.items[0].path))
        parent_dir = os.path.dirname(album_dir)
        release_id = task.match.info.album_id
        total_tracks = len(task.match.info.tracks)

        sibling = self._sibling_releases.get(parent_dir)
        if sibling and sibling[1] == identity and sibling[0] == release_id:
            tracks_so_far = sibling[2] + len(task.items)
        else:
            tracks_so_far = len(task.items)

        self._sibling_releases[parent_dir] = [release_id, identity, tracks_so_far, total_tracks]
        self._counted_dirs.add(counted_dir)

        self._log.debug('[{0}] recorded manually chosen release {1!r} for multi-disc reuse under {2!r}',
                         os.path.basename(album_dir), release_id, parent_dir)