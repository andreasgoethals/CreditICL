"""Rebuild an immutable runtime from byte-recorded installed package artifacts.

Run snapshot on a CPU allocation using the accepted legacy interpreter. This
archives distribution payloads as wheels; it does not relocate a populated venv.
The image recipe installs those wheels into a fresh /opt/venv. Original Git
origins are retained alongside artifact hashes for the project's source audit.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import sys
import zipfile
from pathlib import Path

from packaging.tags import parse_tag, sys_tags


def compatible_tag(wheel_metadata: str) -> str:
    supported = set(sys_tags())
    return next(line.partition(': ')[2] for line in wheel_metadata.splitlines()
                if line.startswith('Tag: ') and parse_tag(line.partition(': ')[2]) & supported)


def snapshot(destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"Snapshot destination already exists: {destination}")
    destination.mkdir(parents=True)
    wheels = destination / "wheels"
    wheels.mkdir()
    inventory, origins = [], {}
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        name, version = dist.metadata['Name'], dist.version
        key = name.lower().replace('-', '_').replace('.', '_')
        relatives = [str(f).replace('\\', '/') for f in (dist.files or [])]
        info = next((f.split('/')[0] for f in relatives if '.dist-info/' in f), None)
        if not info:  # The editable project's dependency metadata is mounted with its checkout.
            if key != 'crediticl':
                raise ValueError(f"Missing wheel metadata: {name}")
            info = f"{key}-{version}.dist-info"
        wheel_metadata = dist.read_text('WHEEL') or 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'
        tag = compatible_tag(wheel_metadata)
        target = wheels / f"{key}-{version}-{tag}.whl"
        origin = dist.read_text('direct_url.json')
        if origin and key != 'crediticl':
            origins[info] = json.loads(origin)
        records = []
        with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            def add(relative: str, payload: bytes, executable: bool = False) -> None:
                entry = zipfile.ZipInfo(relative)
                entry.external_attr = (0o100755 if executable else 0o100644) << 16
                archive.writestr(entry, payload)
                digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b'=').decode()
                records.append((relative, 'sha256=' + digest, str(len(payload))))

            if key != 'crediticl':
                for item in dist.files or []:
                    relative = str(item).replace('\\', '/')
                    # Pip recreates console scripts. Never transplant old shebangs,
                    # editable path hooks, caches or files outside site-packages.
                    if '..' in Path(relative).parts:
                        source = Path(dist.locate_file(item)).resolve()
                        # Standalone installed ELF tools (e.g. Ruff) are wheel
                        # script assets. Python console entrypoints are regenerated.
                        if source.is_file() and source.parent == (Path(sys.prefix) / 'bin').resolve():
                            payload = source.read_bytes()
                            if payload.startswith(b'\x7fELF'):
                                add(info.removesuffix('.dist-info') + '.data/scripts/' + source.name,
                                    payload, executable=True)
                        continue
                    if '__pycache__' in Path(relative).parts:
                        continue
                    if relative.endswith(('/RECORD', '/direct_url.json', '.pyc')):
                        continue
                    source = Path(dist.locate_file(item))
                    if source.is_file():
                        add(relative, source.read_bytes(), executable=bool(source.stat().st_mode & 0o111))
            else:
                add(info + '/METADATA', dist.metadata.as_string().encode())
                add(info + '/WHEEL', wheel_metadata.encode())
            records.append((info + '/RECORD', '', ''))
            buffer = io.StringIO(newline='')
            csv.writer(buffer, lineterminator='\n').writerows(records)
            archive.writestr(info + '/RECORD', buffer.getvalue())
        with target.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        inventory.append(dict(name=name, version=version, wheel=target.name,
                              sha256=digest, files=len(records)-1))
        print(f"SNAPSHOT {name} {version}: {len(records)-1} files", flush=True)
    (destination / 'origins.json').write_text(json.dumps(origins, indent=2), encoding='utf-8')
    result = dict(python=sys.version, executable=sys.executable, packages=inventory,
                  method='installed-payload wheels; fresh image venv; console scripts regenerated')
    (destination / 'inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        inventory = json.loads((args.destination / 'inventory.json').read_text())
        for package in inventory['packages']:
            with (args.destination / 'wheels' / package['wheel']).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != package['sha256']:
                    raise ValueError(f"Changed artifact: {package['name']}")
    else:
        snapshot(args.destination)


if __name__ == '__main__':
    main()
