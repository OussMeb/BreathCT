#!/usr/bin/env python3
"""
Read-only recovery audit for SGR-FM DVFs.

The script searches for surviving validation DVFs in:
- the MICCAI project directory;
- the parent Desktop directory;
- ~/Downloads;
- Linux Trash;
- ZIP archives in those locations.

It does not copy, move, delete, restore, or overwrite DVFs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED = [f"NLST_{i:04d}_DVF.nii.gz" for i in range(1, 11)]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rglob(root: Path, pattern: str):
    if not root.exists():
        return
    try:
        yield from root.rglob(pattern)
    except (PermissionError, OSError):
        return


def extract_manifest_hashes(obj: Any, source: str, out: list[dict]) -> None:
    if isinstance(obj, dict):
        filename = obj.get("filename") or obj.get("relative_path") or obj.get("path")
        digest = obj.get("sha256")
        if isinstance(filename, str) and isinstance(digest, str):
            base = Path(filename).name
            if base in EXPECTED:
                out.append(
                    {
                        "filename": base,
                        "sha256": digest.lower(),
                        "manifest": source,
                    }
                )
        for value in obj.values():
            extract_manifest_hashes(value, source, out)
    elif isinstance(obj, list):
        for value in obj:
            extract_manifest_hashes(value, source, out)


def scan_manifests(roots: list[Path]) -> list[dict]:
    records: list[dict] = []
    seen = set()
    for root in roots:
        for pattern in ("*.json",):
            for path in safe_rglob(root, pattern) or []:
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    if path.stat().st_size > 50 * 1024 * 1024:
                        continue
                    obj = json.loads(path.read_text(errors="ignore"))
                    extract_manifest_hashes(obj, str(path), records)
                except Exception:
                    continue
    return records


def scan_loose_dvfs(roots: list[Path]) -> list[dict]:
    found = []
    seen = set()
    for root in roots:
        for path in safe_rglob(root, "*_DVF.nii.gz") or []:
            try:
                resolved = str(path.resolve())
                if resolved in seen or path.name not in EXPECTED:
                    continue
                seen.add(resolved)
                found.append(
                    {
                        "filename": path.name,
                        "path": str(path),
                        "parent": str(path.parent),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
            except Exception as exc:
                found.append(
                    {
                        "filename": path.name,
                        "path": str(path),
                        "parent": str(path.parent),
                        "error": repr(exc),
                    }
                )
    return found


def scan_zip_archives(roots: list[Path]) -> list[dict]:
    results = []
    seen = set()
    for root in roots:
        for path in safe_rglob(root, "*.zip") or []:
            try:
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                with zipfile.ZipFile(path, "r") as zf:
                    members = []
                    for info in zf.infolist():
                        base = Path(info.filename).name
                        if base in EXPECTED and not info.is_dir():
                            members.append(
                                {
                                    "filename": base,
                                    "archive_member": info.filename,
                                    "size_bytes": info.file_size,
                                    "crc": info.CRC,
                                }
                            )
                    if members:
                        results.append(
                            {
                                "zip_path": str(path),
                                "zip_sha256": sha256_file(path),
                                "members": members,
                                "expected_count": len({m["filename"] for m in members}),
                                "is_complete_10_case_set": (
                                    sorted({m["filename"] for m in members}) == EXPECTED
                                ),
                            }
                        )
            except Exception:
                continue
    return results


def group_loose_sets(loose: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for rec in loose:
        grouped[rec["parent"]].append(rec)
    sets = []
    for parent, records in grouped.items():
        names = sorted({r["filename"] for r in records})
        sets.append(
            {
                "directory": parent,
                "count": len(names),
                "is_complete_10_case_set": names == EXPECTED,
                "missing": sorted(set(EXPECTED) - set(names)),
                "files": sorted(records, key=lambda x: x["filename"]),
            }
        )
    return sorted(
        sets,
        key=lambda x: (not x["is_complete_10_case_set"], -x["count"], x["directory"]),
    )


def match_manifest_hashes(loose: list[dict], manifests: list[dict]) -> list[dict]:
    by_hash = defaultdict(list)
    for rec in manifests:
        by_hash[(rec["filename"], rec["sha256"])].append(rec["manifest"])

    matches = []
    for rec in loose:
        if "sha256" not in rec:
            continue
        manifests_for_file = by_hash.get((rec["filename"], rec["sha256"].lower()), [])
        if manifests_for_file:
            matches.append(
                {
                    "filename": rec["filename"],
                    "path": rec["path"],
                    "sha256": rec["sha256"],
                    "matching_manifests": sorted(set(manifests_for_file)),
                }
            )
    return matches


def parse_args():
    home = Path.home()
    default_project = home / "Desktop/MICCAI FRANCE"
    p = argparse.ArgumentParser(description="Read-only SGR-FM DVF recovery audit.")
    p.add_argument("--project-root", type=Path, default=default_project)
    p.add_argument(
        "--output-json",
        type=Path,
        default=default_project / "dvf_recovery_audit.json",
    )
    p.add_argument(
        "--output-txt",
        type=Path,
        default=default_project / "dvf_recovery_audit.txt",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    home = Path.home()

    roots = [
        args.project_root,
        args.project_root.parent,
        home / "Downloads",
        home / ".local/share/Trash/files",
    ]

    # Remove duplicate/nested identical roots while preserving order.
    unique_roots = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    print("Scanning for surviving DVFs and submission archives...")
    loose = scan_loose_dvfs(unique_roots)
    zip_sets = scan_zip_archives(unique_roots)
    manifests = scan_manifests([args.project_root])
    loose_sets = group_loose_sets(loose)
    hash_matches = match_manifest_hashes(loose, manifests)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "searched_roots": [str(p) for p in unique_roots],
        "expected_filenames": EXPECTED,
        "complete_loose_sets": [
            s for s in loose_sets if s["is_complete_10_case_set"]
        ],
        "partial_loose_sets": [
            s for s in loose_sets if not s["is_complete_10_case_set"]
        ],
        "zip_sets": zip_sets,
        "manifest_hash_record_count": len(manifests),
        "files_matching_saved_manifest_hashes": hash_matches,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "SGR-FM DVF RECOVERY AUDIT",
        f"Created: {report['created_utc']}",
        "Read-only: yes",
        "",
        f"Complete loose 10-case sets: {len(report['complete_loose_sets'])}",
    ]
    for item in report["complete_loose_sets"]:
        lines.append(f"  PASS {item['directory']}")

    lines += ["", f"ZIP archives containing DVFs: {len(zip_sets)}"]
    for item in zip_sets:
        state = "COMPLETE" if item["is_complete_10_case_set"] else "PARTIAL"
        lines.append(
            f"  {state} ({item['expected_count']}/10) {item['zip_path']}"
        )

    lines += ["", f"Partial loose sets: {len(report['partial_loose_sets'])}"]
    for item in report["partial_loose_sets"][:30]:
        lines.append(
            f"  {item['count']}/10 {item['directory']} "
            f"missing={','.join(item['missing'])}"
        )

    lines += [
        "",
        f"Recovered files matching saved manifest hashes: {len(hash_matches)}",
    ]
    for item in hash_matches[:50]:
        lines.append(f"  MATCH {item['filename']} {item['path']}")
        for manifest in item["matching_manifests"][:3]:
            lines.append(f"        manifest: {manifest}")

    lines += [
        "",
        f"JSON report: {args.output_json}",
        "No files were moved, copied, restored, or deleted.",
    ]
    args.output_txt.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
