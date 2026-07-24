#!/usr/bin/env python3
"""Download benchmark scripts and result artifacts from the training host."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import paramiko


REMOTE_ROOT = PurePosixPath("/volumes/oss5/models/qwen-scaling")
FILES = [
    "README.md",
    "requirements.txt",
    "requirements-full.txt",
    "dataset_utils.py",
    "run_case.sh",
    "run_scaling.sh",
    "run_200step_plan.sh",
    "setup_rocm_container.sh",
    "summarize.py",
    "train_qwen.py",
    "merge_gpu_samples.py",
]
DIRECTORIES = ["configs", "timing"]


def download_file(
    sftp: paramiko.SFTPClient,
    remote: PurePosixPath,
    local: Path,
    manifest: list[dict],
) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(str(remote), str(local))
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    manifest.append({
        "remote": str(remote),
        "local": str(local),
        "size_bytes": local.stat().st_size,
        "sha256": digest,
    })
    print(f"downloaded {remote} -> {local}")


def download_tree(
    sftp: paramiko.SFTPClient,
    remote: PurePosixPath,
    local: Path,
    manifest: list[dict],
) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for entry in sorted(sftp.listdir_attr(str(remote)), key=lambda item: item.filename):
        remote_child = remote / entry.filename
        local_child = local / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_tree(sftp, remote_child, local_child, manifest)
        elif stat.S_ISREG(entry.st_mode):
            download_file(sftp, remote_child, local_child, manifest)


def main() -> None:
    destination = Path(os.environ["LOCAL_DEST"]).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ["SSH_HOST"],
        username=os.environ["SSH_USER"],
        password=os.environ["SSH_PASSWORD"],
        timeout=15,
        banner_timeout=15,
        auth_timeout=15,
    )
    sftp = client.open_sftp()
    manifest: list[dict] = []
    for relative in FILES:
        download_file(
            sftp,
            REMOTE_ROOT / relative,
            destination / "scripts" / relative,
            manifest,
        )
    for relative in DIRECTORIES:
        download_tree(
            sftp,
            REMOTE_ROOT / relative,
            destination / relative,
            manifest,
        )
    sftp.close()
    client.close()

    manifest_path = destination / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
            "remote_host": os.environ["SSH_HOST"],
            "remote_root": str(REMOTE_ROOT),
            "file_count": len(manifest),
            "total_bytes": sum(item["size_bytes"] for item in manifest),
            "files": manifest,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
