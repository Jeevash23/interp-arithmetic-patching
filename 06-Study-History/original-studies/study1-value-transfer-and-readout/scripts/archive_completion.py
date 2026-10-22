"""Snapshot delivered source and hash artifacts after a passing audit.

This is a completion archive, never a pre-outcome or held-out freeze.
Existing completion snapshots cannot be overwritten by this utility.
"""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/v3/source-at-completion"
MANIFEST = ROOT / "results/v3/completion-manifest.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def kept(path):
    return path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"


def main():
    audit = json.loads((ROOT / "results/v3/analysis/artifact-audit.json").read_text())
    assert audit["pass"] and audit["validation_model_runs"] == 0
    assert not ARCHIVE.exists() and not MANIFEST.exists(), "Completion archive already exists"
    ARCHIVE.mkdir(parents=True)
    sources = [ROOT / name for name in ("README.md", "RESULTS.md", "pyproject.toml", ".gitignore")]
    for name in ("src", "scripts", "tests", "configs", "docs", "paper"):
        sources.extend(p for p in (ROOT / name).rglob("*") if kept(p) and "literature" not in p.parts)
    for source in sorted(sources):
        target = ARCHIVE / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(source) == digest(target)
    paths = set(sources)
    for name in ("data", "results", "logs", "docs"):
        paths.update(p for p in (ROOT / name).rglob("*") if kept(p))
    entries = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted(paths)]
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "completion archive; not a preregistration or held-out freeze",
        "source_archive": str(ARCHIVE.relative_to(ROOT)),
        "source_files": len(sources),
        "audit_pass": audit["pass"],
        "audit_checks": len(audit["checks"]),
        "held_out_pairs_evaluated": 0,
        "held_out_status": "reserved; English development progression gate failed",
        "model_files": "Pinned checkpoint hashes are in data/model-manifest.json and were checked by the artifact audit.",
        "environment": "Shared .venv is not copied; exact package inventory and source lockfile are included.",
        "exclusions": "Model weight files, environment, caches and this manifest itself are not directly listed.",
        "files": entries,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"manifest": str(MANIFEST), "source_files": len(sources),
                      "hashed_files": len(entries), "audit_checks": len(audit["checks"])}))


if __name__ == "__main__":
    main()
