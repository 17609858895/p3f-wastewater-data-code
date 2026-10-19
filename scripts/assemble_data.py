"""Rejoin byte shards and verify the exact original Parquet SHA-256 checksums."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for item in json.loads((ROOT / "data/file_parts.json").read_text(encoding="utf-8")):
        target = ROOT / item["path"]
        if target.exists() and digest(target) == item["sha256"]:
            print("Verified", item["path"])
            continue
        temp = target.with_name(target.name + ".assembling")
        with temp.open("wb") as out:
            for part in item["parts"]:
                src = ROOT / part["path"]
                if src.stat().st_size != part["bytes"] or digest(src) != part["sha256"]:
                    raise ValueError(f"Corrupt data shard: {part['path']}")
                with src.open("rb") as stream:
                    for block in iter(lambda: stream.read(2**20), b""):
                        out.write(block)
        if temp.stat().st_size != item["bytes"] or digest(temp) != item["sha256"]:
            raise ValueError(f"Assembled data checksum mismatch: {item['path']}")
        temp.replace(target)
        print("Assembled", item["path"])


if __name__ == "__main__":
    main()
