"""Create a reproducible source snapshot for the isolated remote study."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    selected = sorted(p for p in root.rglob("*") if p.is_file()
                      and p.suffix in {".py", ".cpp", ".hpp", ".md", ".json", ".txt"}
                      and not any(part in {"results", "__pycache__", "build_local"}
                                  or part.startswith("build_") for part in p.relative_to(root).parts)
                      and p.name != "RUN_STATUS.json")
    snapshot = {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes() for p in selected}
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                          for name, data in snapshot.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import io
    with tarfile.open(args.output, "w:gz") as archive:
        for name, data in snapshot.items():
            entry = tarfile.TarInfo("test/plan_selection_study/" + name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        entry = tarfile.TarInfo("test/plan_selection_study/source_snapshot.json")
        entry.size = len(data)
        entry.mode = 0o644
        archive.addfile(entry, io.BytesIO(data))
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archive": str(args.output), "file_count": len(snapshot),
                      "archive_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
