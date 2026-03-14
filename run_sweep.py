#!/usr/bin/env python3
"""
run_sweep.py  –  A/B LoRA parameter sweep runner for ai-toolkit
================================================================

Generates a series of training configs from a base config + sweep meta-config,
then runs them sequentially via run.py.

Usage:
    python run_sweep.py my_sweep.yaml
    python run_sweep.py my_sweep.yaml --dry-run   # generate configs only

If interrupted, re-run the same command: already-finished runs are detected
via .done marker files and skipped automatically.

Sweep meta-config format (YAML):

    base_config: "config/my_base.yaml"   # base training config (template)
    sweep_dir: "sweeps/my_sweep"         # root for generated configs and outputs

    # How to combine params:
    #   "product"  → cartesian product of all param lists (all combinations)
    #   "zip"      → parallel lists (same index, same length required)
    mode: "product"

    sweep:
      - param: "train.lr"           # dot-path relative to config.process[0]
        values: [1e-5, 5e-5, 1e-4]

      - param: "network.linear"
        range:                      # integer/float range (from, to inclusive, step)
          from: 8
          to: 32
          step: 8

      - param: "train.batch_size"
        linspace:                   # N evenly-spaced values between from and to
          from: 1
          to: 4
          n: 4

    # Applied to every run before sweep params (sweep params take priority)
    fixed_overrides:
      train.steps: 1000
      save.save_every: 250

    options:
      recover: true       # continue to next run if one fails (default: false)
      dry_run: false      # generate configs only, skip training (default: false)
      log_dir: null       # if set, each run logs to log_dir/<run_name>.log
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import subprocess
import sys
from typing import Any

import yaml


# ─────────────────────────────────────────────────────────────────
# YAML helpers
# ─────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


# ─────────────────────────────────────────────────────────────────
# Nested-dict access via dot-path
# ─────────────────────────────────────────────────────────────────

def _parse_path(dot_path: str) -> list[str]:
    """
    Convert a dot-path string to a list of keys/indices.

    Paths starting with "config." or "meta." are absolute (from YAML root).
    Everything else is relative to config.process[0].

    Examples
    --------
    "train.lr"                     → ["config","process","0","train","lr"]
    "network.linear"               → ["config","process","0","network","linear"]
    "datasets.0.caption_dropout_rate" → ["config","process","0","datasets","0","caption_dropout_rate"]
    "config.name"                  → ["config","name"]
    """
    if dot_path.startswith("config.") or dot_path.startswith("meta."):
        return dot_path.split(".")
    return ["config", "process", "0"] + dot_path.split(".")


def _set(obj: Any, parts: list[str], value: Any) -> None:
    """Set a value in a nested dict/list structure using a pre-parsed path."""
    for p in parts[:-1]:
        obj = obj[int(p)] if isinstance(obj, list) else obj[p]
    last = parts[-1]
    if isinstance(obj, list):
        obj[int(last)] = value
    else:
        obj[last] = value


# ─────────────────────────────────────────────────────────────────
# Value expansion
# ─────────────────────────────────────────────────────────────────

def expand_param(spec: dict) -> list:
    """
    Return a concrete list of values from a sweep param spec.

    Supported specs
    ---------------
    values:   [a, b, c, ...]
    range:    {from: X, to: Y, step: S}    inclusive on both ends
    linspace: {from: X, to: Y, n: N}
    """
    if "values" in spec:
        return list(spec["values"])

    if "range" in spec:
        r = spec["range"]
        start, stop, step = r["from"], r["to"], r["step"]
        if step <= 0:
            raise ValueError(f"range.step must be positive, got {step}")
        result, v = [], start
        # Use a small epsilon to avoid floating-point under-counting at the boundary
        while v <= stop + abs(step) * 1e-9:
            result.append(v if isinstance(start, float) else int(round(v)))
            v += step
        return result

    if "linspace" in spec:
        ls = spec["linspace"]
        start, stop, n = ls["from"], ls["to"], int(ls["n"])
        if n < 1:
            raise ValueError(f"linspace.n must be >= 1, got {n}")
        if n == 1:
            return [start]
        result = [start + (stop - start) * i / (n - 1) for i in range(n)]
        # When both endpoints are integers (e.g. rank sweep 8→32), return ints.
        # NetworkConfig, TrainConfig etc. expect int rank/batch_size and do NOT
        # cast: passing 16.0 where 16 is expected can cause TypeError downstream
        # (e.g. torch.zeros(rank, ...) fails if rank is float).
        if isinstance(start, int) and isinstance(stop, int):
            result = [int(round(v)) for v in result]
        return result

    raise ValueError(
        f"Param spec must contain 'values', 'range', or 'linspace'. Got: {spec}"
    )


# ─────────────────────────────────────────────────────────────────
# Value formatting for run names
# ─────────────────────────────────────────────────────────────────

def fmt(v: Any) -> str:
    """
    Compact, filesystem-safe string for a single value used in run names.

    Examples
    --------
    1e-4   → "1e-4"
    5e-5   → "5e-5"
    1.5e-4 → "1.5e-4"
    0.05   → "0.05"
    32     → "32"
    """
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            # Scientific notation: format then clean up
            s = f"{v:.6e}"               # e.g. "1.000000e-04"
            mantissa, exp_part = s.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")   # "1.000000" → "1"
            exp_val = int(exp_part)       # "-04" → -4; strips sign and leading zeros
            return f"{mantissa}e{exp_val}"   # "1e-4"
        return f"{v:.6g}"                # handles 0.05 → "0.05" cleanly
    return str(v)


# ─────────────────────────────────────────────────────────────────
# Resume helpers (done-marker files)
# ─────────────────────────────────────────────────────────────────

def _done_marker(configs_dir: str, run_name: str) -> str:
    return os.path.join(configs_dir, f"{run_name}.done")


def _mark_done(configs_dir: str, run_name: str) -> None:
    with open(_done_marker(configs_dir, run_name), "w") as f:
        f.write("ok\n")


def _is_done(configs_dir: str, run_name: str) -> bool:
    return os.path.exists(_done_marker(configs_dir, run_name))


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="A/B LoRA sweep runner for ai-toolkit"
    )
    parser.add_argument("sweep_config", help="Path to the sweep meta-config YAML")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and print plan, but do not start training",
    )
    args = parser.parse_args()

    # ── Load sweep meta-config ──────────────────────────────────
    sweep_cfg_path = os.path.abspath(args.sweep_config)
    if not os.path.exists(sweep_cfg_path):
        print(f"ERROR: sweep config not found: {sweep_cfg_path}", file=sys.stderr)
        sys.exit(1)

    sweep_cfg = load_yaml(sweep_cfg_path)
    sweep_root = os.path.dirname(sweep_cfg_path)

    # ── Resolve base config ─────────────────────────────────────
    if "base_config" not in sweep_cfg:
        print("ERROR: sweep config must have a 'base_config' key.", file=sys.stderr)
        sys.exit(1)

    base_config_path = sweep_cfg["base_config"]
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(sweep_root, base_config_path)
    base_config_path = os.path.abspath(base_config_path)

    if not os.path.exists(base_config_path):
        print(f"ERROR: base_config not found: {base_config_path}", file=sys.stderr)
        sys.exit(1)

    base_cfg = load_yaml(base_config_path)

    # ── Sweep output directory ──────────────────────────────────
    sweep_out = sweep_cfg.get("sweep_dir", "sweep_output")
    if not os.path.isabs(sweep_out):
        sweep_out = os.path.join(sweep_root, sweep_out)
    sweep_out = os.path.abspath(sweep_out)

    # ── Read options (guard against null values in YAML) ────────
    mode            = sweep_cfg.get("mode", "product")
    sweep_params    = sweep_cfg.get("sweep") or []        # None-safe
    fixed_overrides = sweep_cfg.get("fixed_overrides") or {}   # None-safe
    options         = sweep_cfg.get("options") or {}      # None-safe
    recover         = bool(options.get("recover", False))
    dry_run         = args.dry_run or bool(options.get("dry_run", False))
    log_dir         = options.get("log_dir") or None

    if log_dir and not os.path.isabs(log_dir):
        log_dir = os.path.abspath(os.path.join(sweep_root, log_dir))

    # ── Expand sweep parameters ─────────────────────────────────
    param_names  = [p["param"] for p in sweep_params]
    param_values = [expand_param(p) for p in sweep_params]

    if not sweep_params:
        combos = [()]
    elif mode == "product":
        combos = list(itertools.product(*param_values))
    elif mode == "zip":
        lengths = [len(v) for v in param_values]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"'zip' mode requires all param lists to have the same length. "
                f"Got: {dict(zip(param_names, lengths))}"
            )
        combos = list(zip(*param_values))
    else:
        raise ValueError(f"Unknown mode: {repr(mode)}. Use 'product' or 'zip'.")

    base_name   = base_cfg["config"]["name"]
    configs_dir = os.path.join(sweep_out, "configs")

    # ── Print plan ──────────────────────────────────────────────
    w = 64
    print("=" * w)
    print(f"  Sweep plan : {len(combos)} run(s)")
    print(f"  Base config: {base_config_path}")
    print(f"  Sweep dir  : {sweep_out}")
    print(f"  Mode       : {mode}")
    if param_names:
        print(f"  Params     : {', '.join(param_names)}")
    if fixed_overrides:
        print(f"  Fixed      : {fixed_overrides}")
    if dry_run:
        print("  [DRY RUN – configs will be written but training skipped]")
    print("=" * w)
    print()
    sys.stdout.flush()

    # ── Generate all configs upfront ────────────────────────────
    generated: list[tuple[str, str]] = []   # (config_path, run_name_fs)

    for i, combo in enumerate(combos):
        run_cfg = copy.deepcopy(base_cfg)

        # Build human-readable suffix from swept values
        if combo:
            suffix = "__".join(
                f"{name.split('.')[-1]}{fmt(val)}"
                for name, val in zip(param_names, combo)
            )
            run_name = f"{base_name}__{suffix}"
        else:
            run_name = base_name

        # Sanitize for filesystem (spaces and slashes are the dangerous ones on Linux)
        run_name_fs = run_name.replace(" ", "_").replace("/", "-")

        # Set config name
        _set(run_cfg, ["config", "name"], run_name_fs)

        # training_folder is the SHARED parent directory for all runs.
        # The toolkit computes save_root = os.path.join(training_folder, config.name),
        # so each run's actual output lands in: sweep_dir/runs/<run_name>/
        # DO NOT include run_name_fs in training_folder — that would cause doubling:
        # e.g.  sweep/runs/my_lora__lr1e-4/my_lora__lr1e-4/  ← wrong
        #        sweep/runs/my_lora__lr1e-4/                  ← correct
        run_cfg["config"]["process"][0]["training_folder"] = os.path.join(sweep_out, "runs")

        # Apply fixed overrides first, then sweep values (sweep wins on conflict)
        for dot_path, value in fixed_overrides.items():
            _set(run_cfg, _parse_path(dot_path), value)

        for name, val in zip(param_names, combo):
            _set(run_cfg, _parse_path(name), val)

        # Write generated config
        config_path = os.path.join(configs_dir, f"{run_name_fs}.yaml")
        save_yaml(run_cfg, config_path)
        generated.append((config_path, run_name_fs))

        vals_str = ", ".join(f"{n}={fmt(v)}" for n, v in zip(param_names, combo))
        n_width = len(str(len(combos)))
        print(f"  [{i+1:>{n_width}}/{len(combos)}]  {run_name_fs}")
        if vals_str:
            print(f"  {' ' * (n_width * 2 + 4)} {vals_str}")
        sys.stdout.flush()

    print()
    print(f"Generated {len(generated)} config(s) in: {configs_dir}")
    sys.stdout.flush()

    if dry_run:
        print("\nDry run complete – skipping training.")
        return

    # ── Run training jobs sequentially ──────────────────────────
    toolkit_dir = os.path.dirname(os.path.abspath(__file__))
    run_script  = os.path.join(toolkit_dir, "run.py")

    completed = 0
    failed    = 0
    skipped   = 0

    for i, (config_path, run_name) in enumerate(generated):
        # Resume support: skip runs that already finished successfully
        if _is_done(configs_dir, run_name):
            print(f"\n  [{i+1}/{len(generated)}] SKIP (already done): {run_name}")
            skipped += 1
            sys.stdout.flush()
            continue

        print()
        print("─" * w)
        print(f"  Training {i+1}/{len(generated)}: {run_name}")
        print("─" * w)
        sys.stdout.flush()

        # NOTE: we deliberately do NOT pass --recover to run.py.
        # run.py's --recover makes it exit 0 even on failure (it only matters when
        # multiple configs are passed to a single run.py call, which we never do).
        # Our own recover logic is handled below by checking the return code.
        cmd = [sys.executable, run_script, config_path]
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            cmd += ["--log", os.path.join(log_dir, f"{run_name}.log")]

        try:
            result = subprocess.run(cmd, cwd=toolkit_dir)
        except KeyboardInterrupt:
            print("\n\n  Interrupted by user. Sweep stopped.")
            print(f"  Progress: {completed} completed, {failed} failed, "
                  f"{skipped} skipped, {len(generated) - completed - failed - skipped - 1} remaining.")
            print("  Re-run the same command to resume from this point.")
            sys.exit(130)   # standard exit code for Ctrl+C

        if result.returncode == 0:
            completed += 1
            _mark_done(configs_dir, run_name)
            print(f"\n  Run {i+1} finished OK.")
        else:
            failed += 1
            print(f"\n  Run {i+1} FAILED (exit code {result.returncode}).")
            if not recover:
                print("  Stopping sweep. Set options.recover: true to continue on failure.")
                print("  Re-run the same command after fixing the issue to resume.")
                break

        sys.stdout.flush()

    print()
    print("=" * w)
    parts = [f"{completed} completed"]
    if skipped:
        parts.append(f"{skipped} skipped (resumed)")
    if failed:
        parts.append(f"{failed} failed")
    print(f"  Sweep done: {', '.join(parts)}")
    print("=" * w)


if __name__ == "__main__":
    main()
