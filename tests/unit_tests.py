"""
Fast offline tests for cataloghint's scoring/text functions
No network, no fixture folders needed.
"""
from __future__ import annotations

from beetsplug.cataloghint import (
    build_strip_pattern,
    gather_haystack,
    match_score,
    score_hits,
)


def test_match_score_prefers_exact_catalog_over_partial_overlap():
    # Folder says "61340-2".
    # One release's catalog is exactly that, another's contains it ("3705-61340-2").
    #   -> Exact one must score higher, (not just tie on shared digits)

    folder_exact, folder_loose = "kyuss 61340-2 flac", "kyuss613402flac"

    exact_score, exact_matched = match_score("61340-2", folder_exact, folder_loose)
    partial_score, _ = match_score("3705-61340-2", folder_exact, folder_loose)

    assert exact_score > partial_score
    assert exact_matched


def test_match_score_ignores_short_coincidental_overlap():
    #  -> 2-3 digit coincidental overlap shouldn't count as a match at all

    score, matched = match_score("12", "some folder with a 12 in it", "somefolderwitha12init")
    assert score == 0.0
    assert matched == ''


def test_build_strip_pattern_is_word_bounded():
    #  -> Must not eat "Kyussy" or similar as a substring match

    pattern = build_strip_pattern("Kyuss")
    assert pattern.sub(' ', "Kyuss - Blues For The Red Sun") == '  - Blues For The Red Sun'
    assert pattern.sub(' ', "Kyussy Fanzine") == 'Kyussy Fanzine'


def test_folder_hint_strips_known_artist_and_album_but_keeps_other_signal():
    folder_exact, folder_loose, countries, years = gather_haystack(
        item_dir="/x/Kyuss - 1992 - Blues For The Red Sun {DALI 61340-2} [FLAC]",
        artist="Kyuss",
        album="Blues For The Red Sun",
        check_cue=False,
    )
    assert 'kyuss' not in folder_loose
    assert 'blues' not in folder_loose
    assert '61340' in folder_loose
    assert years == {1992}


def test_folder_hint_keeps_bracketed_disambiguation_even_with_a_dirty_album_tag():
    # Real-world tags often dirty (e.g. a track tagged with album "Foo {2016 Deluxe Ed.}" instead of just "Foo")
    #  -> Stripping the whole of the tag must not also wipe the bracket's contents

    item_dir = "/x/Some Artist - Foo Album {2016 Deluxe Ed.}"
    clean_exact, clean_loose, _, clean_years = gather_haystack(
        item_dir, artist="Some Artist", album="Foo Album", check_cue=False,
    )
    dirty_exact, dirty_loose, _, dirty_years = gather_haystack(
        item_dir, artist="Some Artist", album="Foo Album {2016 Deluxe Ed.}", check_cue=False,
    )
    assert 'deluxe' in clean_loose and 'deluxe' in dirty_loose
    assert clean_years == dirty_years == {2016}


def test_folder_hint_extracts_bracketed_country_code():
    _, _, countries, _ = gather_haystack(
        item_dir="/x/Some Artist - Some Album [US]",
        artist="Some Artist",
        album="Some Album",
        check_cue=False,
    )
    assert countries == {'US'}


def test_folder_hint_aliases_uk_to_gb():
    _, _, countries, _ = gather_haystack(
        item_dir="/x/Some Artist - Some Album [UK]",
        artist="Some Artist",
        album="Some Album",
        check_cue=False,
    )
    assert countries == {'GB'}


def test_score_hits_resolves_unique_catalog_match():
    hits = [
        {'id': 'good', 'label_info': [{'catalog_number': '61340-2'}]},
        {'id': 'noise', 'label_info': [{'catalog_number': '3705-61340-2'}]},
        {'id': 'unrelated', 'label_info': [{'catalog_number': 'ABC-999'}]},
    ]
    folder_exact, folder_loose = "kyuss 61340-2 flac", "kyuss613402flac"

    release_id, scored = score_hits(hits, folder_exact, folder_loose, set(), set())
    assert release_id == 'good'
    assert len(scored) == 3


def test_score_hits_returns_none_on_tie():
    hits = [
        {'id': 'a', 'country': 'US', 'date': '2015'},
        {'id': 'b', 'country': 'GB', 'date': '2015'},
    ]
    # Neither country nor catalog text distinguishes them, but they share the year
    release_id, scored = score_hits(hits, '', '', set(), {2015})
    assert release_id is None
    assert len(scored) == 2


def test_score_hits_year_breaks_a_text_score_tie():
    # Two releases share an identical catalog number, only one matches the folder's year
    #  -> Should be enough to break tie
    hits = [
        {'id': 'a', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'b', 'date': '1999', 'label_info': [{'catalog_number': 'XYZ-123'}]},
    ]
    release_id, _ = score_hits(hits, 'xyz-123', 'xyz123', set(), {2015})
    assert release_id == 'a'


def test_score_hits_leaves_a_genuine_multi_way_tie_unresolved():
    # Three releases share the same catalog number and the same year
    #   -> Nothing to break tie with, so this must defer (None)
    hits = [
        {'id': 'a', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'b', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'c', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
    ]
    release_id, scored = score_hits(hits, 'xyz-123', 'xyz123', set(), {2015})
    assert release_id is None
    assert len(scored) == 3


def test_score_hits_weak_evidence_resolves_when_only_one_hit_qualifies():
    # No text evidence at all, but only ONE carries any weak (country) signal
    #   -> It wins by default
    # TODO: Not so sure this is advisable actually...

    hits = [
        {'id': 'a', 'country': 'US', 'date': '2015'},
        {'id': 'b', 'country': 'GB', 'date': '1999'},
    ]
    release_id, _ = score_hits(hits, '', '', {'GB'}, set())
    assert release_id == 'b'
