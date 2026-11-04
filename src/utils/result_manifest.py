"""Filter a remote output inventory for local analysis; never transfer model weights.

This creates a download plan only. Feed its members to the existing verified SFTP
workflow; do not recursively copy output trees or archives. No network or deletion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from src.utils.atomic import write_text

ANALYSIS_SUFFIXES = frozenset({'.csv', '.json', '.jsonl', '.log', '.out', '.err',
                               '.txt', '.md', '.pdf', '.ipynb', '.xml'})
EXCLUDED_DIRECTORIES = frozenset({'checkpoints', 'prior_cache', 'data', 'runtimes'})


def analysis_manifest(inventory: dict) -> dict:
    """Allowlisted evidence with unchanged source hashes; reject ambiguous paths."""
    retained, excluded, seen = [], [], {}
    for member in inventory['members']:
        name = member['path']
        if (not isinstance(name, str) or not name
                or any(ord(c) < 32 or c in '\\:"<>|?*' for c in name)
                or any(p in {'', '.', '..'} or p.endswith((' ', '.'))
                       or PureWindowsPath(p).is_reserved() for p in name.split('/'))):
            raise ValueError(f'Unsafe inventory path: {name!r}')
        path = PurePosixPath(name)
        size = member['bytes']
        digest = member['sha256']
        if (type(size) is not int or size < 0 or not isinstance(digest, str)
                or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError(f'Invalid size/hash for {name}')
        key = name.casefold()
        if key in seen:
            # A merged DATA/PROJECT inventory may repeat byte-identical files.
            if seen[key] != (name, size, digest):
                raise ValueError(f'Conflicting local destination: {name}')
            continue
        seen[key] = name, size, digest
        parts = [p.casefold() for p in path.parts]
        owner = parts[0]
        in_output = (owner in {'reference', 'figures', 'manifests', 'logs'}
                     or re.fullmatch(r'experiment_[0-3]', owner)
                     or name in {'All_Results.md', 'CAPTIONS.md'})
        keep = (in_output and not EXCLUDED_DIRECTORIES.intersection(parts[:-1])
                and path.suffix.casefold() in ANALYSIS_SUFFIXES)
        if keep:
            retained.append(dict(member))
        else:
            excluded.append({'path': name, 'bytes': size})
    return {'scope': 'analysis-only; weights, archives, caches and lock files excluded',
            'production_commit': inventory.get('production_commit'),
            'captured_at': inventory.get('at'), 'roots': inventory.get('roots', {}),
            'members': retained, 'files': len(retained),
            'bytes': sum(m['bytes'] for m in retained),
            'excluded': excluded, 'excluded_bytes': sum(m['bytes'] for m in excluded)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.inventory.resolve() == args.out.resolve():
        parser.error('Preserve the original inventory; choose a different output path')
    raw = args.inventory.read_bytes()
    result = analysis_manifest(json.loads(raw.decode('utf-8-sig')))
    result['source_inventory_sha256'] = hashlib.sha256(raw).hexdigest()
    write_text(args.out, json.dumps(result, indent=2) + '\n')
    print(f"Analysis plan: {result['files']} files, {result['bytes'] / 1024**2:.1f} MiB; "
          f"excluded {result['excluded_bytes'] / 1024**3:.2f} GiB. Nothing downloaded or deleted.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
