# beets-cataloghint

A [beets](https://beets.io/) plugin that narrows beets' MusicBrainz match down
to the exact release, before the candidate list is shown.

When beets imports an album, MusicBrainz's [musicbrainz plugin](https://beets.readthedocs.io/en/stable/plugins/musicbrainz.html)
often resolves the right _release-group_ but struggles to pick the correct _release_ in that group
(different pressings, editions, regions, remasters...).

`cataloghint` looks through what info is available in the album folder (barcode, catalog number,
disambiguation text, country, year) and, when it can, picks the exact release for you.

## What it does

- **Exactly one release stands out**: Picked, and recommendation forced to `strong`
- **None stands out, but beets' top pick scores _worse_ other release(s) in the same group**: Recommendation forced to `none`
- **A disc under a parent folder whose sibling disc(s) already resolved as a multi-disc release**: New disc is matched to that same release, as long as it still fits within the release's remaining
  multi-disc track count.

If no MusicBrainz release-group was resolved, or the lookup fails, `cataloghint` defers to beets as usual.

## Installation

```shell
git clone https://github.com/FlorentLM/beets-cataloghint
uv pip install beets-cataloghint
```

Or if you used Beet's default `uv tool` installation method:

```shell
git clone https://github.com/FlorentLM/beets-cataloghint
uv tool install beets --with ./beets-cataloghint --reinstall
```

Then enable the plugin in your beets config:

```yaml
plugins: cataloghint musicbrainz
```

(`cataloghint` requires the `musicbrainz` plugin to be enabled)

## Configuration

All settings optional, defaults are:

```yaml
cataloghint:
    check_cue: yes
    check_sibling_discs: yes
    auto_apply: no
```

- **check_cue**: Look inside any `.cue` file found in the album folder for `TITLE`, `PERFORMER` and `CATALOG` hints.
- **check_sibling_discs**: Reuse an already-resolved release for sibling discs found under the
  same parent folder (matched by artist/album identity), instead of re-scoring from scratch.
- **auto_apply**: When a release is uniquely resolved, apply it automatically instead of just
  forcing the recommendation to `strong`.

## License

MIT
