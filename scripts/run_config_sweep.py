"""
Run several training configurations sequentially.

The default sweep focuses on distillation temperature while keeping the
remaining hyperparameters close to the current best-known config.

Usage:
    python scripts/run_config_sweep.py --config config.yaml --device cuda
    python scripts/run_config_sweep.py --config config.yaml --stage distillation --continue-on-error
"""

import argparse
import copy
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_TEMPERATURES = (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_id_from_temperature(temperature):
    return f"temp_{str(temperature).replace('.', 'p')}"


def load_yaml(path):
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def write_yaml(config, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config, config_file, sort_keys=False)


def build_sweep_config(base_config, temperature, output_dir):
    run_id = run_id_from_temperature(temperature)
    config = copy.deepcopy(base_config)

    config.setdefault("distillation", {})
    config["distillation"]["temperature"] = temperature
    config["distillation"].setdefault("alpha", 0.3)
    config["distillation"].setdefault("lr", 2.0e-4)
    config["distillation"]["save_path"] = str(output_dir / "checkpoints" / f"student_distilled_{run_id}.pt")

    config.setdefault("generation", {})
    config["generation"]["temperature"] = temperature

    return run_id, config


def run_training(config_path, stage, args, repo_root, log_path):
    command = [
        args.python,
        "-u",
        str(repo_root / "scripts" / "train.py"),
        "--config",
        str(config_path),
        "--stage",
        stage,
    ]

    if args.device:
        command.extend(["--device", args.device])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"[{timestamp()}] Command: {' '.join(command)}")
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

    return return_code


def parse_temperatures(raw_temperatures):
    if not raw_temperatures:
        return list(DEFAULT_TEMPERATURES)

    temperatures = []
    for raw_temperature in raw_temperatures:
        temperatures.extend(float(value.strip()) for value in raw_temperature.split(",") if value.strip())
    return temperatures


def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Run a sequential config sweep.")
    parser.add_argument("--config", type=Path, default=repo_root / "config.yaml", help="Base config file")
    parser.add_argument("--stage", choices=["student_baseline", "distillation", "all"], default="distillation")
    parser.add_argument("--device", type=str, default=None, help="Device passed to train.py, for example cuda or cpu")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
    parser.add_argument(
        "--temperatures",
        nargs="*",
        default=None,
        help="Temperatures to test. Accepts space-separated values or comma-separated groups.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs" / "config_sweeps",
        help="Directory for generated configs, checkpoints, and logs",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later runs when one run fails",
    )
    args = parser.parse_args()

    if not args.config.is_absolute():
        args.config = repo_root / args.config
    if not args.output_dir.is_absolute():
        args.output_dir = repo_root / args.output_dir

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    temperatures = parse_temperatures(args.temperatures)
    base_config = load_yaml(args.config)
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = args.output_dir / sweep_id
    config_dir = sweep_dir / "configs"
    log_dir = sweep_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{timestamp()}] Config sweep started")
    print(f"[{timestamp()}] Base config: {args.config}")
    print(f"[{timestamp()}] Stage: {args.stage}")
    print(f"[{timestamp()}] Temperatures: {', '.join(str(t) for t in temperatures)}")
    print(f"[{timestamp()}] Sweep directory: {sweep_dir}")

    failed_runs = []
    for index, temperature in enumerate(temperatures, start=1):
        run_id, run_config = build_sweep_config(base_config, temperature, sweep_dir)
        config_path = config_dir / f"{index:02d}_{run_id}.yaml"
        log_path = log_dir / f"{index:02d}_{run_id}.log"
        write_yaml(run_config, config_path)

        print(f"\n[{timestamp()}] Starting run {index}/{len(temperatures)}: {run_id}")
        return_code = run_training(config_path, args.stage, args, repo_root, log_path)

        if return_code == 0:
            print(f"[{timestamp()}] Finished run {index}/{len(temperatures)}: {run_id}")
        else:
            print(f"[{timestamp()}] Run failed: {run_id} (exit code {return_code})")
            failed_runs.append(run_id)
            if not args.continue_on_error:
                break

    if failed_runs:
        print(f"[{timestamp()}] Sweep finished with failures: {', '.join(failed_runs)}")
        return 1

    print(f"[{timestamp()}] Sweep finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
