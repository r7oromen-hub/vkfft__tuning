#!/usr/bin/env python3
"""Audit a rename-only executable path migration without changing measurements."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_exclusive(path,data):
    if path.exists():
        if path.read_bytes()!=data:
            raise RuntimeError(f"existing audit file differs: {path}")
        return
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def migrate(results_dir,new_executable,expected_old_executable,apply=False):
    manifest_path=results_dir / "manifest.json"
    original_bytes=manifest_path.read_bytes()
    original=json.loads(original_bytes)
    expected_sha=original.get("executable_sha256")
    if not isinstance(expected_sha,str) or len(expected_sha)!=64:
        raise RuntimeError("manifest must contain the original executable_sha256")
    new_path=new_executable.resolve(strict=True)
    if not new_path.is_file():
        raise RuntimeError("new executable must be a regular file")
    actual_sha=digest(new_path.read_bytes())
    if actual_sha!=expected_sha:
        raise RuntimeError("renamed binary SHA256 differs from original; refusing scientific-protocol migration")
    new_path_string=str(new_path)
    old_path=original.get("executable")
    if old_path==new_path_string:
        return dict(status="already_current",executable=new_path_string,executable_sha256=actual_sha,
                    manifest_sha256=digest(original_bytes))
    if old_path!=expected_old_executable:
        raise RuntimeError(f"old executable path differs from explicit guard: {old_path!r}")
    updated=dict(original,executable=new_path_string)
    old_science={k:v for k,v in original.items() if k!="executable"}
    new_science={k:v for k,v in updated.items() if k!="executable"}
    if old_science!=new_science:
        raise RuntimeError("unexpected scientific-field change")
    updated_bytes=(json.dumps(updated,indent=2)+"\n").encode("utf-8")
    before_sha,after_sha=digest(original_bytes),digest(updated_bytes)
    backup=results_dir / f"manifest.before_path_migration.{before_sha[:12]}.json"
    audit_path=results_dir / f"path_migration.{before_sha[:12]}.{after_sha[:12]}.json"
    audit=dict(schema_version=1,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               action="rename_only_executable_path",old_executable=old_path,new_executable=new_path_string,
               executable_sha256_unchanged=actual_sha,original_manifest_sha256=before_sha,
               updated_manifest_sha256=after_sha,original_manifest_backup=str(backup),
               changed_fields=["executable"],scientific_fields_unchanged=True,
               raw_measurements_and_calibration_records_modified=False,
               reason="User-requested project/executable rename; measured binary renamed without rebuild",
               recovery="This audit and immutable backup are written before atomic manifest replacement; compare manifest hashes to determine completion.")
    if not apply:
        return dict(status="verified_dry_run",**audit)
    write_exclusive(backup,original_bytes)
    # Preserve the first declaration on a retry after interruption.
    if audit_path.exists():
        existing=json.loads(audit_path.read_text(encoding="utf-8"))
        if any(existing.get(k)!=audit[k] for k in ("old_executable","new_executable","executable_sha256_unchanged",
                                                   "original_manifest_sha256","updated_manifest_sha256")):
            raise RuntimeError("existing migration audit differs")
    else:
        write_exclusive(audit_path,(json.dumps(audit,indent=2)+"\n").encode("utf-8"))
    temporary=results_dir / "manifest.path_migration.tmp"
    write_exclusive(temporary,updated_bytes)
    if manifest_path.read_bytes()!=original_bytes:
        raise RuntimeError("manifest changed concurrently; refusing replacement")
    os.replace(temporary,manifest_path)
    return dict(status="migrated",audit_file=str(audit_path),backup_file=str(backup),
                executable_sha256=actual_sha,manifest_sha256=after_sha,
                changed_fields=["executable"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir",type=Path,required=True)
    p.add_argument("--new-executable",type=Path,required=True)
    p.add_argument("--expected-old-executable",required=True)
    p.add_argument("--apply",action="store_true",help="Apply the verified single-field change; default only verifies")
    args=p.parse_args()
    print(json.dumps(migrate(args.results_dir,args.new_executable,args.expected_old_executable,args.apply),indent=2))


if __name__=="__main__":
    main()
