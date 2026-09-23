"""
Fast offline tests for cataloghint's scoring/text functions
No network, no fixture folders needed.
"""
from __future__ import annotations

import os

from beetsplug.cataloghint import (
    CASSETTE_FORMATS,
    CD_FORMATS,
    DIGITAL_FORMATS,
    VINYL_FORMATS,
    build_strip_pattern,
    extract_media_hint,
    format_veto,
    gather_disc_layout,
    gather_haystack,
    gather_needles,
    match_score,
    validate_preferred_countries,
    score_hits,
)
from beetsplug.cataloghint.cuefiles import has_cue


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
    folder_exact, folder_loose, countries, years, _ = gather_haystack(
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
    clean_exact, clean_loose, _, clean_years, _ = gather_haystack(
        item_dir, artist="Some Artist", album="Foo Album", check_cue=False,
    )
    dirty_exact, dirty_loose, _, dirty_years, _ = gather_haystack(
        item_dir, artist="Some Artist", album="Foo Album {2016 Deluxe Ed.}", check_cue=False,
    )
    assert 'deluxe' in clean_loose and 'deluxe' in dirty_loose
    assert clean_years == dirty_years == {2016}


def test_folder_hint_extracts_bracketed_country_code():
    _, _, countries, _, _ = gather_haystack(
        item_dir="/x/Some Artist - Some Album [US]",
        artist="Some Artist",
        album="Some Album",
        check_cue=False,
    )
    assert countries == {'US'}


def test_folder_hint_aliases_uk_to_gb():
    _, _, countries, _, _ = gather_haystack(
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

    release_ids, scored = score_hits(hits, folder_exact, folder_loose, set(), set())
    assert release_ids == ['good']
    assert len(scored) == 3


def test_score_hits_returns_both_tied_candidates_instead_of_picking():
    hits = [
        {'id': 'a', 'country': 'US', 'date': '2015'},
        {'id': 'b', 'country': 'GB', 'date': '2015'},
    ]
    # Neither country nor catalog text distinguishes them, but they share the year
    release_ids, scored = score_hits(hits, '', '', set(), {2015})
    assert sorted(release_ids) == ['a', 'b']
    assert len(scored) == 2


def test_score_hits_year_breaks_a_text_score_tie():
    # Two releases share an identical catalog number, only one matches the folder's year
    #  -> Should be enough to break tie
    hits = [
        {'id': 'a', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'b', 'date': '1999', 'label_info': [{'catalog_number': 'XYZ-123'}]},
    ]
    release_ids, _ = score_hits(hits, 'xyz-123', 'xyz123', set(), {2015})
    assert release_ids == ['a']


def test_score_hits_returns_all_tied_candidates_on_a_genuine_multi_way_tie():
    # Three releases share the same catalog number and the same year
    #   -> Nothing to break tie with, so all three are handed back for beets to pick between
    hits = [
        {'id': 'a', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'b', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'c', 'date': '2015', 'label_info': [{'catalog_number': 'XYZ-123'}]},
    ]
    release_ids, scored = score_hits(hits, 'xyz-123', 'xyz123', set(), {2015})
    assert sorted(release_ids) == ['a', 'b', 'c']
    assert len(scored) == 3


def test_score_hits_preferred_countries_breaks_a_genuine_tie():
    hits = [
        {'id': 'a', 'date': '2015', 'country': 'US', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'b', 'date': '2015', 'country': 'GB', 'label_info': [{'catalog_number': 'XYZ-123'}]},
    ]
    release_ids, _ = score_hits(
        hits, 'xyz-123', 'xyz123', set(), {2015}, preferred_countries=['GB', 'US'],
    )
    assert release_ids == ['b']


def test_score_hits_preferred_countries_is_not_consulted_when_other_evidence_resolves_it():
    hits = [
        {'id': 'strong_match', 'date': '2010', 'country': 'US', 'label_info': [{'catalog_number': '0602527463841'}]},
        {'id': 'year_only', 'date': '2000', 'country': 'GB'},
    ]
    release_ids, _ = score_hits(
        hits, '0602527463841', '0602527463841', set(), {2000}, preferred_countries=['GB'],
    )
    assert release_ids == ['strong_match']


def test_score_hits_preferred_countries_still_returns_the_tie_when_none_of_the_releases_are_listed():
    hits = [
        {'id': 'a', 'date': '2015', 'country': 'FR', 'label_info': [{'catalog_number': 'XYZ-123'}]},
        {'id': 'b', 'date': '2015', 'country': 'DE', 'label_info': [{'catalog_number': 'XYZ-123'}]},
    ]
    release_ids, _ = score_hits(
        hits, 'xyz-123', 'xyz123', set(), {2015}, preferred_countries=['GB', 'US'],
    )
    assert sorted(release_ids) == ['a', 'b']


def test_normalize_preferred_countries_aliases_and_drops_unknowns():
    assert validate_preferred_countries(['uk', 'usa', 'gb', 'zz']) == ['GB', 'US']


def test_score_hits_prefers_year_match_over_a_disambiguation_shared_with_another_release():
    # "deluxe edition" is reused by a wrong year and a wrong format
    #   -> weak evidence. 'correct' has no disambiguation at all, but matches format and year
    hits = [
        {'id': 'wrong_year', 'date': '2009', 'disambiguation': 'deluxe edition', 'media': _media(8, 9)},
        {'id': 'vinyl', 'date': '2016', 'disambiguation': 'deluxe edition',
         'media': [{'position': 1, 'track_count': 8, 'format': 'Vinyl'}, {'position': 2, 'track_count': 9, 'format': 'Vinyl'}]},
        {'id': 'correct', 'date': '2016', 'disambiguation': '', 'media': _media(8, 9)},
    ]
    release_ids, _ = score_hits(
        hits, '2016 deluxe ed', '2016deluxeed', set(), {2016},
        disc_layout={1: 8, 2: 9}, target_formats=CD_FORMATS,
    )
    assert release_ids == ['correct']


def test_score_hits_unique_text_match_still_wins_despite_a_different_years_match():
    # 'strong_match's catalog number is exact *and unshared*
    #   -> Beats a year-only match
    hits = [
        {'id': 'strong_match', 'date': '2010', 'label_info': [{'catalog_number': '0602527463841'}]},
        {'id': 'year_only', 'date': '2000'},
    ]
    release_ids, _ = score_hits(hits, '0602527463841', '0602527463841', set(), {2000})
    assert release_ids == ['strong_match']


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
    release_ids, _ = score_hits(hits, '', '', {'GB'}, set())
    assert release_ids == ['b']


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
    release_ids, _ = score_hits(hits, '', '', set(), set(), {1: 12, 2: 8})
    assert release_ids == ['a']


def test_score_hits_disc_layout_vetoes_a_release_that_cannot_hold_what_is_on_disk():
    # Disc 2 has 9 tracks, 'c' reports only 8 for that disc -> Impossible
    hits = [
        {'id': 'a', 'media': _media(12, 9), 'country': 'US'},
        {'id': 'b', 'media': _media(12, 9), 'country': 'GB'},
        {'id': 'c', 'media': _media(12, 8), 'country': 'US'},
    ]
    release_ids, scored = score_hits(hits, '', '', {'GB'}, set(), {1: 12, 2: 9})
    assert release_ids == ['b']
    assert dict((rid, score) for rid, score, *_ in scored)['c'] == -1.0


def test_score_hits_does_not_veto_a_release_reporting_more_tracks_than_found():
    # 'b' claims 11 tracks on disc 2 (e.g. bonus DVD track never ripped as audio)
    #   -> Possible, just not confirmed as an exact fit
    #   -> 'a' still wins becaue it exact matches
    hits = [
        {'id': 'a', 'media': _media(12, 8)},
        {'id': 'b', 'media': _media(12, 11)},
    ]
    release_ids, scored = score_hits(hits, '', '', set(), set(), {1: 12, 2: 8})
    assert release_ids == ['a']
    assert dict((rid, score) for rid, score, *_ in scored)['b'] != -1.0


def test_score_hits_disc_layout_vetoes_a_release_with_too_few_discs():
    # Found "Disc 2" locally. 'b' is single-disc release -> Impossible
    hits = [
        {'id': 'a', 'media': _media(12, 8)},
        {'id': 'b', 'media': _media(12)},
    ]
    release_ids, scored = score_hits(hits, '', '', set(), set(), {1: 12, 2: 8})
    assert release_ids == ['a']
    assert dict((rid, score) for rid, score, *_ in scored)['b'] == -1.0


def test_score_hits_ignores_disc_layout_contradicting_every_release():
    # Local track counts disagree with all candidates -> Data is not trustworthy, veto dropped
    hits = [
        {'id': 'a', 'media': _media(5)},
        {'id': 'b', 'media': _media(3)},
    ]
    release_ids, scored = score_hits(hits, '', '', set(), set(), {1: 12})
    assert release_ids == []
    assert all(score >= 0 for _, score, *_ in scored)


def test_format_veto_flags_a_release_with_no_medium_in_the_target_formats():
    vinyl_only = {'id': 'a', 'media': [{'format': 'Vinyl'}]}
    cd = {'id': 'b', 'media': [{'format': 'CD'}]}
    assert format_veto(vinyl_only, CD_FORMATS) is True
    assert format_veto(cd, CD_FORMATS) is False


def test_score_hits_target_format_vetoes_a_vinyl_release():
    hits = [
        {'id': 'cd', 'media': [{'format': 'CD'}]},
        {'id': 'vinyl', 'media': [{'format': 'Vinyl'}]},
    ]
    release_ids, scored = score_hits(hits, '', '', set(), set(), target_formats=CD_FORMATS)
    assert release_ids == ['cd']
    assert dict((rid, score) for rid, score, *_ in scored)['vinyl'] == -1.0


def test_score_hits_ignores_target_format_when_it_contradicts_every_release():
    # Every release in the group is vinyl or cassette
    #   -> Target isn't trustworthy, veto dropped
    hits = [
        {'id': 'a', 'media': [{'format': 'Vinyl'}]},
        {'id': 'b', 'media': [{'format': 'Cassette'}]},
    ]
    release_ids, scored = score_hits(hits, '', '', set(), set(), target_formats=CD_FORMATS)
    assert release_ids == []
    assert all(score >= 0 for _, score, *_ in scored)


def test_has_cue_finds_a_cue_file_anywhere_under_the_directory(tmp_path):
    assert has_cue(tmp_path) is False
    (tmp_path / "album.cue").write_text("")
    assert has_cue(tmp_path) is True


def test_extract_media_hint_recognizes_each_keyword():
    assert extract_media_hint("Artist - Album [Vinyl]") == VINYL_FORMATS
    assert extract_media_hint("Artist - Album [WEB]") == DIGITAL_FORMATS
    assert extract_media_hint("Artist - Album [CD]") == CD_FORMATS
    assert extract_media_hint("Artist - Album [Cassette]") == CASSETTE_FORMATS
    assert extract_media_hint("Artist - Album [FLAC]") is None


def test_extract_media_hint_is_none_on_conflicting_keywords():
    # "CD" and "Vinyl" both present
    #   -> Ambiguous
    assert extract_media_hint("Artist - Album [CD+Vinyl bundle]") is None


def test_gather_haystack_detects_media_hint_after_stripping_artist_and_album():
    # The album title contains "Cassette". Without stripping it first, it would
    # collide with the "[Vinyl Reissue]" tag and make the hint ambiguous
    _, _, _, _, media_hint = gather_haystack(
        item_dir="/x/Artist - Cassette Tape Dreams [Vinyl Reissue]",
        artist="Artist",
        album="Cassette Tape Dreams",
        check_cue=False,
    )
    assert media_hint == VINYL_FORMATS


def test_folder_hint_excludes_cd_from_country_codes():
    # "CD" is a real MB area code (Congo), but in a folder name we consider it is Compact Disc
    _, _, countries, _, _ = gather_haystack(
        item_dir="/x/Some Artist - Some Album [CD]",
        artist="Some Artist",
        album="Some Album",
        check_cue=False,
    )
    assert countries == set()
