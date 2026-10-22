"""Record the actual environment and cache a revision-pinned public checkpoint."""
import hashlib
import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import torch
from huggingface_hub import HfApi, snapshot_download

def command(args):
    p = subprocess.run(args, capture_output=True, text=True)
    return {"command": args, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def main():
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'data').mkdir(exist_ok=True)
    env = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "versions": {}, "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free,memory.used", "--format=csv"]),
        "ram": command(["free", "-b"]),
        "packages": command([sys.executable, "-m", "pip", "freeze"]),
        "distribution_inventory": sorted({d.metadata['Name']+'=='+d.version for d in md.distributions() if d.metadata.get('Name')}),
    }
    for package in ["torch", "transformers", "transformer-lens", "transformer-bridge", "huggingface-hub", "numpy", "scipy", "safetensors", "accelerate", "matplotlib"]:
        try:
            env["versions"][package] = md.version(package)
        except md.PackageNotFoundError:
            env["versions"][package] = None
    if torch.cuda.is_available():
        env["cuda_mem_free_total_bytes"] = torch.cuda.mem_get_info()
        env["device"] = str(torch.cuda.get_device_properties(0))
        env["matmul_smoke"] = float((torch.ones(4,4,device="cuda") @ torch.ones(4,4,device="cuda"))[0,0])
    (ROOT / "results/environment.json").write_text(json.dumps(env, indent=2))
    print(json.dumps({k:v for k,v in env.items() if k != "packages"}, indent=2), flush=True)
    repo = "Qwen/Qwen3-0.6B"
    manifest_path = ROOT / "data/model-manifest.json"
    revision = json.loads(manifest_path.read_text())["revision"] if manifest_path.exists() else HfApi().model_info(repo).sha
    manifest_path.write_text(json.dumps({"repo": repo, "revision": revision, "status": "download_started"}, indent=2))
    snapshot_download(repo, revision=revision, local_dir=ROOT / "models/Qwen3-0.6B", allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "README.md", "LICENSE"], max_workers=2)
    hashes = {}
    for p in sorted((ROOT / "models/Qwen3-0.6B").glob("*")):
        if p.is_file():
            h=hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda:f.read(8*1024*1024), b""):
                    h.update(block)
            hashes[p.name] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    manifest_path.write_text(json.dumps({"repo": repo, "revision": revision, "files": hashes, "status": "download_verified"}, indent=2))
    print("Model cache verified", flush=True)

if __name__ == "__main__":
    main()
