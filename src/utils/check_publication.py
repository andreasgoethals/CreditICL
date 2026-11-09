"""Check a publication snapshot using institution-only disclosure rules."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from src.data.private_data import PRIVATE_KEYS, publication_text, site, site_path


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=root)


def _decode(path: str, data: bytes) -> str:
    if path.lower().endswith(".pdf"):
        result = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", "-", "-"],
            input=data, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        data = result.stdout
    text = data.decode("utf-8-sig")
    if path.lower().endswith(".ipynb"):
        notebook = json.loads(text)
        for cell in notebook["cells"]:
            for record in [*cell.get("outputs", []), *cell.get("attachments", {}).values()]:
                bundle = record.get("data", record)
                for mime in ("image/png", "image/jpeg", "application/pdf"):
                    bundle.pop(mime, None)
        return json.dumps(notebook, indent=1, ensure_ascii=False)
    return text


def findings(text: str, columns: list[str], row_counts: list[int]):
    """Yield locations and categories without printing the matched secrets."""
    terms = re.compile(
        r"(?<![a-z0-9_])(?:" + "|".join(map(re.escape, columns)) + r")(?![a-z0-9_]|\.[a-z0-9_])",
        re.IGNORECASE,
    ) if columns else None
    aliases = re.compile(r"\b(?:" + "|".join(sorted(PRIVATE_KEYS)) + r")\b", re.IGNORECASE)
    counts = set(row_counts)
    for number, line in enumerate(text.splitlines(), 1):
        normalized = line.replace(r"\_", "_")
        if publication_text(normalized) != normalized:
            yield number, "source identifier"
        if terms is not None and terms.search(normalized):
            yield number, "restricted column"
        if aliases.search(normalized):
            for token in re.findall(r"(?<![\d.])\d{1,3}(?:[, _]\d{3})+|(?<![\d.])\d+(?![\d.])", normalized):
                if int(re.sub(r"[, _]", "", token)) in counts:
                    yield number, "restricted row count"
                    break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Read the index instead of working files")
    parser.add_argument("--rules", type=Path, help="Private JSON with columns and row_counts")
    args = parser.parse_args()
    site(required=True)
    rules_path = args.rules or site_path().with_name("publication-rules.json")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    root = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    index = {}
    for entry in _git(root, "ls-files", "--stage", "-z").decode().strip("\0").split("\0"):
        header, path = entry.split("\t", 1)
        mode, oid, stage = header.split()
        if stage != "0":
            raise ValueError("Resolve unmerged entries before checking publication")
        if mode != "160000":
            index[path] = oid
    if not args.staged:
        index.update(dict.fromkeys(_git(root, "ls-files", "--others", "--exclude-standard", "-z")
                                   .decode().strip("\0").split("\0")))
    failures = 0
    checked = 0
    for path, oid in index.items():
        if not path or (not args.staged and not (root / path).is_file()):
            continue
        if path.startswith(".private-data/"):
            print("A restricted configuration file is tracked.")
            failures += 1
            continue
        data = _git(root, "cat-file", "blob", oid) if args.staged else (root / path).read_bytes()
        checked += 1
        try:
            matches = list(findings(path + "\n" + _decode(path, data), rules["columns"], rules["row_counts"]))
        except (UnicodeError, OSError, subprocess.CalledProcessError):
            matches = [(0, "artifact could not be inspected")]
        for line, category in matches:
            location = publication_text(path)
            print(f"{location}:{line - 1 if line else 0}: {category}")
        failures += len(matches)
    print(f"Inspected {checked} files; {failures} flagged locations.")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
