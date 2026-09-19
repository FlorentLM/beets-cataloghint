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
"""
from __future__ import annotations
import difflib
import os
import re
from datetime import date
from typing import TYPE_CHECKING, Optional
import requests
from beets import plugins
from beets.autotag import Recommendation
from beets.importer.state import ImportState
from beets.plugins import BeetsPlugin

from beetsplug.cataloghint.cuefiles import find_and_parse

if TYPE_CHECKING:
    from beets.autotag.match import AlbumMatch
    from beets.importer import ImportSession, ImportTask
    from beetsplug.musicbrainz import MusicBrainzPlugin


## Configs

MIN_OVERLAP = 5     # below this, a (non-year) shared digit run is more likely coincidence


## Compiled regexes

CATALOG_RE = re.compile(r"^(\d{12,13})$")
NORMALIZE_RE = re.compile(r"[\W_]")     # strip punctuation/symbols/whitespace

BRACKETED_RE = re.compile(r"[\[\(\{][^\[\]\(\)\{\}]*[\]\)\}]")

# Country: only used if a bracket/parenthese/brace whole content is a 2-letter code (for instance "[US]")
COUNTRY_BRACKET_RE = re.compile(r"[\[\(\{]\s*([A-Za-z]{2})\s*[\]\)\}]")
COUNTRY_CODE_ALIASES = {'UK': 'GB'}

# An isolated 4-digit number: plausible as a release year (original, reissue/remaster, ...)
YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# Classic tokens found in disc-only folder names ("Disc 1", "CD2", "Bonus Disc", "B-Sides"...)
# cd/disc/vinyl only count as a token if followed by a number
DISC_TOKEN_RE = re.compile(
    r"(?i)\b(?:cd|dis[ck]|vinyl)\s*\d{1,2}(?!\d)|\b(?:dvd|bonus|b-?sides?|remix(es)?|album|vol(?:ume)?|part|pt)(?![a-z])"
)

PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

DISC_NAME_MAX_LEN = 10


## Custom error

class MusicBrainzUnavailable(Exception):
    """A MusicBrainz lookup couldn't complete (already exhausted the HTTP client's own retries)."""



## Helpers

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


def looks_like_disc(folder: str, artist: Optional[str] = None, album: Optional[str] = None) -> bool:
    """Whether `folder` looks like a single disc name (not a release folder)."""

    if artist and (pattern := build_strip_pattern(artist)):
        folder = pattern.sub(' ', folder)

    if album and (pattern := build_strip_pattern(album)):
        folder = pattern.sub(' ', folder)

    folder = DISC_TOKEN_RE.sub(' ', folder)
    folder = PUNCT_RE.sub('', folder).strip()

    return len(folder) <= DISC_NAME_MAX_LEN


def match_score(needle: str, haystack_exact: str, haystack_loose: str) -> tuple[float, str]:
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
    ) -> tuple[str, str, set[str], set[int]]:
    """
    Build the haystack: All the hints derived for `item_dir`, with the `artist`/`album`
    stripped out of it. If the folder itself looks like a bare disc name, the parent
    folder's name is folded in too.

    Returns two haystack forms: "exact" which only case-folds and "loose" which also strips
    punctuation/whitespace, and any bracketed country code(s) and plausible year(s) found.
    """

    raw_folder = os.path.basename(item_dir)

    if looks_like_disc(raw_folder, artist, album):
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

    # Country codes (only bracketed 2-letter codes are considered)
    folder_countries = {
        COUNTRY_CODE_ALIASES.get(code.upper(), code.upper())
        for code in COUNTRY_BRACKET_RE.findall(raw_folder)
    }

    for known in (artist, album):
        if known and (pattern := build_strip_pattern(known)):
            raw_folder = strip_outside_brackets(raw_folder, pattern)

    folder_years = extract_years(raw_folder)

    return raw_folder.casefold(), normalize(raw_folder), folder_countries, folder_years


def score_hits(
        hits: list[dict],
        folder_exact: str,
        folder_loose: str,
        folder_countries: set[str],
        folder_years: set[int],
        log=None,
    ) -> tuple[str | None, list[tuple[str, float, str, bool, bool]]]:
    """
    Score hits against text/country/year by barcode/catalog/disambiguation coverage.
    Returns the unique best match's id (or None if none stands out), plus the full per-hit score.
    """

    logger = log or dummy_log

    scored: list[tuple[str, float, str, bool, bool]] = []
    for hit in hits:
        needles = [hit.get('barcode') or '', hit.get('disambiguation') or '']

        for label_info in hit.get('label_info', []):
            if catno := label_info.get('catalog_number'):
                needles.append(catno)

        text_score = 0.0
        matched_text = ''
        for needle in needles:
            score, matched = match_score(needle, folder_exact, folder_loose)
            if score > text_score:
                text_score = score
                matched_text = matched

        country_match = bool(hit.get('country')) and hit['country'] in folder_countries

        year = None
        if match := YEAR_RE.match(str(hit.get('date') or '')):
            year = int(match.group(1))

        year_match = year is not None and year in folder_years

        logger.debug('  scoring {0} country={1!r} year={2} text_score={3:.2f} matched={4!r} '
                  'country_match={5} year_match={6}',
                  hit['id'], hit.get('country'), year, text_score, matched_text or None,
                  country_match, year_match)

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

    if len(matches) != 1:
        return None, scored

    release_id = next(iter(matches))
    _, score, matched_text, country_match, year_match = next(s for s in scored if s[0] == release_id)

    if score == best_score and score > 0:
        reason = f'text overlap {matched_text!r}'
    elif year_match:
        reason = 'year match'
    else:
        reason = 'country match'

    logger.debug('{0!r} stands out via {1}', release_id, reason)
    return release_id, scored


def resolve_release(
        hits: list[dict],
        item_dir: str,
        artist: Optional[str],
        album: Optional[str],
        check_cue: bool = True,
        filenames: Optional[set[str]] = None,
        log=None,
    ) -> tuple[str | None, list[tuple[str, float, str, bool, bool]]]:
    """
    Pick the release in `hits` (from a given release-group) that the hints point to.

    Returns its id (or None if none stands out), plus the per-hit score so external callers
    can still tell a clearly-worse release from an untried one even if no single hit stood out.
    """

    logger = log or dummy_log

    if len(hits) == 1:
        logger.debug('only release in this release-group, trusting it: {0!r}', hits[0]['id'])
        return hits[0]['id'], []

    folder_exact, folder_loose, folder_countries, folder_years = gather_haystack(
        item_dir, artist, album, check_cue, filenames
    )
    logger.debug('{0} releases in group, folder hint = {1!r}, country code(s) = {2}, year(s) = {3}',
              len(hits), folder_loose, folder_countries or None, folder_years or None)

    return score_hits(hits, folder_exact, folder_loose, folder_countries, folder_years, logger)


class CatalogHintPlugin(BeetsPlugin):

    def __init__(self):
        super().__init__()

        self.config.add(
            {
                'check_cue': True,
                'check_sibling_discs': True,
                'auto_apply': False,
            }
        )
        self.register_listener('import_task_before_choice', self.before_choice)
        self.register_listener('import_task_apply', self._record_incremental_history)

        # parent directory -> [release id, (artist, album) identity, tracks found so far, release's total track count]
        self._sibling_releases: dict[str, list] = {}

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
        identity = (normalize(task.source.artist or ''), normalize(task.source.name or ''))
        log.debug('examining {0!r} (artist={1!r}, album={2!r})',
                  album_dir, task.source.artist, task.source.name)

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

        release_id, scored = self._resolve_release(hits, task, log)
        if release_id is not None:
            log.debug('resolved release {0!r}', release_id)
            return self._inject(task, release_id, parent_dir, identity, log)

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
                'release', **{'release-group': release_group_id}, includes=['labels'], limit=100
            )
        except requests.exceptions.RequestException as e:
            raise MusicBrainzUnavailable(f'release-group {release_group_id}: {e}') from e

    def _resolve_release(self,
            hits: list[dict],
            task: ImportTask,
            log: TaskLog
        ) -> tuple[str | None, list[tuple[str, float, str, bool, bool]]]:

        item_dir = os.path.dirname(os.fsdecode(task.items[0].path))
        filenames = {os.path.basename(os.fsdecode(item.path)) for item in task.items}

        return resolve_release(
            hits, item_dir, task.source.artist, task.source.name,
            check_cue=self.config['check_cue'].get(bool), filenames=filenames, log=log,
        )

    def _flag_if_outscored(self,
            task: ImportTask,
            scored: list[tuple[str, float, str, bool, bool]],
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
            identity: tuple[str, str],
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
