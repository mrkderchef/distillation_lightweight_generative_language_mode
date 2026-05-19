"""
Run the full training pipeline as separate sequential jobs.

This is useful for long unattended runs because each stage gets its own log
file and the runner stops immediately if a stage fails.

Usage:
    python scripts/train_all.py --config config.yaml
    python scripts/train_all.py --config config.yaml --device cuda
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


STAGES = ("teacher", "student_baseline", "distillation")


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_log_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def torch_cuda_available(python_executable):
    if Path(python_executable).resolve() == Path(sys.executable).resolve():
        try:
            import torch
        except ImportError:
            return False
        return torch.cuda.is_available()

    command = [
        python_executable,
        "-c",
        "import torch; print(torch.cuda.is_available())",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0 and result.stdout.strip() == "True"


def run_stage(stage, args, repo_root, log_dir):
    log_path = log_dir / f"{safe_log_timestamp()}_{stage}.log"
    command = [
        args.python,
        "-u",
        str(repo_root / "scripts" / "train.py"),
        "--config",
        str(args.config),
        "--stage",
        stage,
    ]

    if args.device:
        command.extend(["--device", args.device])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[{timestamp()}] Starting stage: {stage}")
    print(f"[{timestamp()}] Log file: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp()}] Command: {' '.join(command)}\n")
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        log_file.write(f"\n[{timestamp()}] Exit code: {return_code}\n")

    if return_code == 0:
        print(f"[{timestamp()}] Finished stage: {stage}")
    else:
        print(f"[{timestamp()}] Stage failed: {stage} (exit code {return_code})")

    return return_code


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Train teacher, baseline student, and distilled student.")
    parser.add_argument("--config", type=Path, default=repo_root / "config.yaml", help="Config file path")
    parser.add_argument("--device", type=str, default=None, help="Device passed to train.py, for example cuda or cpu")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument("--log-dir", type=Path, default=repo_root / "outputs" / "logs", help="Directory for logs")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later stages even if one stage fails",
    )
    args = parser.parse_args()

    if not args.config.is_absolute():
        args.config = repo_root / args.config

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    if args.device and args.device.startswith("cuda") and not torch_cuda_available(args.python):
        print(
            f"[{timestamp()}] GPU training was requested with --device cuda, but this "
            "Python environment's PyTorch install does not expose a CUDA/ROCm GPU backend."
        )
        print(
            f"[{timestamp()}] Re-run without --device to auto-select CPU, use --device cpu, "
            "or install a CUDA/ROCm-enabled PyTorch build first."
        )
        return 1

    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{timestamp()}] Full training run started")
    print(f"[{timestamp()}] Stages: {', '.join(STAGES)}")

    failed = []
    for stage in STAGES:
        return_code = run_stage(stage, args, repo_root, args.log_dir)
        if return_code != 0:
            failed.append(stage)
            if not args.continue_on_error:
                break

    if failed:
        print(f"[{timestamp()}] Training run finished with failures: {', '.join(failed)}")
        return 1

    print(f"[{timestamp()}] Full training run finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
