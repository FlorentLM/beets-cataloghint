import os
import re
from pathlib import Path
from typing import Iterable, Optional, Union


# A cue command is a single all-caps token followed by its value
CUE_RE = re.compile(r"^\s*([A-Z_]+)\s+(.*)$")

# FILE's value is a quoted filename plus an unquoted type keyword (WAVE, MP3, ...)
FILE_VALUE_RE = re.compile(r'^"(?P<name>[^"]*)"\s*(?P<type>\S+)?$')


def parse_cue(path: Path) -> dict:

    path = Path(path)
    if not path.is_file():
        return {}

    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except UnicodeDecodeError:
        lines = path.read_text(encoding='cp1252').splitlines()

    album = {'TRACKS': []}
    cur_track = None
    cur_file = None
    cur_file_type = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = CUE_RE.match(line)
        if not match:
            continue

        key, value = match.groups()

        if key == 'FILE':
            file_match = FILE_VALUE_RE.match(value)
            if file_match:
                cur_file = file_match.group('name')
                cur_file_type = file_match.group('type')
            else:
                cur_file = value.strip('"')
                cur_file_type = None
            continue

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]

        if key == 'TRACK':
            cur_track = {
                'TRACK': value,
            }
            if cur_file:
                cur_track['FILE'] = cur_file
                if cur_file_type:
                    cur_track['FILE_TYPE'] = cur_file_type
            album['TRACKS'].append(cur_track)

        else:
            target = cur_track if cur_track is not None else album

            if key in target:
                if isinstance(target[key], list):
                    target[key].append(value)
                else:
                    target[key] = [target[key], value]
            else:
                target[key] = value

    return album


def find_and_parse(
        directory: Union[str, os.PathLike],
        filenames: Optional[Iterable[str]] = None
    ) -> Optional[dict]:
    """
    Find the .cue file in `directory` whose FILE entries best match `filenames`
    and return its parsed content.

    If `filenames` isn't passed, the content of `directory` is used.
    """

    directory = Path(directory)
    candidates = sorted(directory.glob('**/*.cue'))

    if not candidates:
        return None

    parsed = [parse_cue(p) for p in candidates]
    if len(parsed) == 1:
        return parsed[0]

    if filenames is None:
        filenames = (p.name for p in directory.iterdir() if p.is_file() and p.suffix.lower() != '.cue')

    known = {name.lower() for name in filenames}

    best_album, best_score = parsed[0], -1.0

    for album in parsed:
        referenced = {Path(t['FILE']).name.lower() for t in album.get('TRACKS', []) if 'FILE' in t}
        if not referenced:
            continue

        score = len(referenced & known) / len(referenced)
        if score > best_score:
            best_album, best_score = album, score

    return best_album
