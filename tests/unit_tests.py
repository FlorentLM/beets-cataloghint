"""
Fast offline tests for cataloghint's scoring/text functions
No network, no fixture folders needed.
"""
from __future__ import annotations

import os

from beetsplug.cataloghint import (
    build_strip_pattern,
    gather_disc_layout,
    gather_haystack,
    gather_needles,
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


def test_gather_needles_strips_barcode_leading_zeros():
    # UPC-A vs EAN-13 vs GTIN-14 packaging pad the same code differently
    assert gather_needles({'barcode': '00602557589924'}) == ['602557589924']
    assert gather_needles({'barcode': '0602557589924'}) == ['602557589924']


def test_score_hits_ties_releases_sharing_a_barcode_regardless_of_padding():
    hits = [
        {'id': 'a', 'barcode': '0602557589924'},
        {'id': 'b', 'barcode': '00602557589924'},
    ]
    _, scored = score_hits(hits, '0602557589924', '0602557589924', set(), set())
    scores = {release_id: score for release_id, score, *_ in scored}
    assert scores['a'] == scores['b']


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


def _media(*track_counts):
    return [{'position': i + 1, 'track_count': n} for i, n in enumerate(track_counts)]


def test_gather_disc_layout_counts_audio_files_per_sibling_disc_folder(tmp_path):
    parent = tmp_path / "Some Album"
    disc1, disc2, extras = parent / "Disc 1", parent / "Disc 2", parent / "Scans"
    for d in (disc1, disc2, extras):
        d.mkdir(parents=True)
    for i in range(3):
        (disc1 / f"track{i}.flac").write_bytes(b'')
    for i in range(2):
        (disc2 / f"track{i}.mp3").write_bytes(b'')
    (extras / "cover.jpg").write_bytes(b'')

    layout = gather_disc_layout(str(disc1), artist=None, album=None)
    assert layout == {1: 3, 2: 2}


def test_score_hits_disc_layout_uniquely_resolves_release_with_no_text_evidence():
    # Disc 1 has 12 tracks and disc 2 has 8. Only one release matches both.
    hits = [
        {'id': 'a', 'media': _media(12, 8)},
        {'id': 'b', 'media': _media(12, 9)},
    ]
    release_id, _ = score_hits(hits, '', '', set(), set(), {1: 12, 2: 8})
    assert release_id == 'a'


def test_score_hits_disc_layout_vetoes_a_release_that_cannot_hold_what_is_on_disk():
    # Disc 2 has 9 tracks, 'c' reports only 8 for that disc -> Impossible
    hits = [
        {'id': 'a', 'media': _media(12, 9), 'country': 'US'},
        {'id': 'b', 'media': _media(12, 9), 'country': 'GB'},
        {'id': 'c', 'media': _media(12, 8), 'country': 'US'},
    ]
    release_id, scored = score_hits(hits, '', '', {'GB'}, set(), {1: 12, 2: 9})
    assert release_id == 'b'
    assert dict((rid, score) for rid, score, *_ in scored)['c'] == -1.0


def test_score_hits_does_not_veto_a_release_reporting_more_tracks_than_found():
    # 'b' claims 11 tracks on disc 2 (e.g. bonus DVD track never ripped as audio)
    #   -> Possible, just not confirmed as an exact fit
    #   -> 'a' still wins becaue it exact matches
    hits = [
        {'id': 'a', 'media': _media(12, 8)},
        {'id': 'b', 'media': _media(12, 11)},
    ]
    release_id, scored = score_hits(hits, '', '', set(), set(), {1: 12, 2: 8})
    assert release_id == 'a'
    assert dict((rid, score) for rid, score, *_ in scored)['b'] != -1.0


def test_score_hits_disc_layout_vetoes_a_release_with_too_few_discs():
    # Found "Disc 2" locally. 'b' is single-disc release -> Impossible
    hits = [
        {'id': 'a', 'media': _media(12, 8)},
        {'id': 'b', 'media': _media(12)},
    ]
    release_id, scored = score_hits(hits, '', '', set(), set(), {1: 12, 2: 8})
    assert release_id == 'a'
    assert dict((rid, score) for rid, score, *_ in scored)['b'] == -1.0


def test_score_hits_ignores_disc_layout_contradicting_every_release():
    # Local track counts disagree with all candidates -> Data is not trustworthy, veto dropped
    hits = [
        {'id': 'a', 'media': _media(5)},
        {'id': 'b', 'media': _media(3)},
    ]
    release_id, scored = score_hits(hits, '', '', set(), set(), {1: 12})
    assert release_id is None
    assert all(score >= 0 for _, score, *_ in scored)
