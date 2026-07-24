#!/usr/bin/env python3
"""Upload the benchmark files to an SSH host without storing credentials."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import paramiko


FILES = [
    ".gitignore",
    "README.md",
    "requirements.txt",
    "requirements-full.txt",
    "dataset_utils.py",
    "run_case.sh",
    "run_scaling.sh",
    "run_200step_plan.sh",
    "run_8gpu_case.sh",
    "run_8gpu_2000step_plan.sh",
    "merge_gpu_samples.py",
    "setup_rocm_container.sh",
    "summarize.py",
    "train_qwen.py",
    "evaluate_model.py",
    "THREE_SCENARIO_8GPU_2000STEP_GUIDE.md",
    "PARAMETER_GUIDE.md",
    "tools/remote_collect.py",
    "tools/remote_sync.py",
    "configs/zero2_bf16.json",
    "configs/zero3_bf16.json",
]


def mkdirs(sftp: paramiko.SFTPClient, path: PurePosixPath) -> None:
    current = PurePosixPath("/")
    for part in path.parts[1:]:
        current /= part
        try:
            sftp.stat(str(current))
        except FileNotFoundError:
            sftp.mkdir(str(current))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    remote_root = PurePosixPath(os.environ.get(
        "REMOTE_ROOT", "/volumes/oss5/models/qwen-scaling"
    ))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(os.environ["SSH_HOST"], username=os.environ["SSH_USER"],
                   password=os.environ["SSH_PASSWORD"], timeout=15,
                   banner_timeout=15, auth_timeout=15)
    sftp = client.open_sftp()
    mkdirs(sftp, remote_root)
    for relative in FILES:
        remote = remote_root / PurePosixPath(relative)
        mkdirs(sftp, remote.parent)
        sftp.put(str(root / relative), str(remote))
        print(f"uploaded {relative}")
    sftp.chmod(str(remote_root / "run_case.sh"), 0o755)
    sftp.chmod(str(remote_root / "run_scaling.sh"), 0o755)
    sftp.chmod(str(remote_root / "run_200step_plan.sh"), 0o755)
    sftp.chmod(str(remote_root / "run_8gpu_case.sh"), 0o755)
    sftp.chmod(str(remote_root / "run_8gpu_2000step_plan.sh"), 0o755)
    sftp.chmod(str(remote_root / "setup_rocm_container.sh"), 0o755)
    sftp.close()
    client.close()


if __name__ == "__main__":
    main()
