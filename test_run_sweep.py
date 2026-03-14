#!/usr/bin/env python3
"""
test_run_sweep.py – Synthetic unit tests for run_sweep.py
==========================================================

Run with:
    python test_run_sweep.py -v

Tests cover:
    - fmt()               value → run-name string formatting
    - _parse_path()       dot-path → list-of-keys conversion
    - _set()              nested dict/list mutation
    - expand_param()      values / range / linspace expansion
    - YAML round-trip     float/int/unicode survive yaml.dump → yaml.safe_load
    - training_folder     must NOT include run_name (would be doubled by toolkit)
    - Full config gen     base config + overrides produces correct output config
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest

import yaml

# Allow importing from ai-toolkit root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_sweep import fmt, _parse_path, _set, expand_param, load_yaml, save_yaml


# ─────────────────────────────────────────────────────────────────
# fmt()
# ─────────────────────────────────────────────────────────────────

class TestFmt(unittest.TestCase):

    # Scientific notation cases (original bug: rstrip on full string never worked)
    def test_1e_minus_4(self):
        self.assertEqual(fmt(1e-4), "1e-4")

    def test_5e_minus_5(self):
        self.assertEqual(fmt(5e-5), "5e-5")

    def test_1e_minus_5(self):
        self.assertEqual(fmt(1e-5), "1e-5")

    def test_fractional_mantissa(self):
        self.assertEqual(fmt(1.5e-4), "1.5e-4")

    def test_2_5e_minus_4(self):
        self.assertEqual(fmt(2.5e-4), "2.5e-4")

    def test_large_scientific(self):
        # abs(v) >= 1e4 → scientific
        self.assertEqual(fmt(1e5), "1e5")

    def test_large_with_mantissa(self):
        self.assertEqual(fmt(2.5e6), "2.5e6")

    # Decimal cases (abs(v) in [1e-3, 1e4) range)
    def test_decimal_normal(self):
        self.assertEqual(fmt(0.05), "0.05")

    def test_decimal_one(self):
        self.assertEqual(fmt(1.0), "1")

    def test_decimal_half(self):
        self.assertEqual(fmt(0.5), "0.5")

    def test_zero_float(self):
        # 0.0: v != 0 is False, so falls to f"{v:.6g}" = "0"
        self.assertEqual(fmt(0.0), "0")

    # Non-float passthrough
    def test_integer(self):
        self.assertEqual(fmt(32), "32")

    def test_integer_zero(self):
        self.assertEqual(fmt(0), "0")

    def test_string(self):
        self.assertEqual(fmt("adamw8bit"), "adamw8bit")

    def test_bool_true(self):
        self.assertEqual(fmt(True), "True")

    # No trailing zeros in mantissa (the bug that was broken)
    def test_no_trailing_zeros_1e4(self):
        result = fmt(1e-4)
        self.assertNotIn("1.00", result)
        self.assertNotIn("00e", result)

    def test_no_trailing_zeros_5e5(self):
        result = fmt(5e-5)
        self.assertNotIn("5.00", result)

    def test_exponent_not_zero_padded(self):
        # Should be "1e-4" not "1e-04"
        self.assertNotIn("-04", fmt(1e-4))
        self.assertNotIn("-05", fmt(5e-5))


# ─────────────────────────────────────────────────────────────────
# _parse_path()
# ─────────────────────────────────────────────────────────────────

class TestParsePath(unittest.TestCase):

    def test_relative_train_lr(self):
        self.assertEqual(
            _parse_path("train.lr"),
            ["config", "process", "0", "train", "lr"]
        )

    def test_relative_network_linear(self):
        self.assertEqual(
            _parse_path("network.linear"),
            ["config", "process", "0", "network", "linear"]
        )

    def test_relative_batch_size(self):
        self.assertEqual(
            _parse_path("train.batch_size"),
            ["config", "process", "0", "train", "batch_size"]
        )

    def test_relative_dataset_list_access(self):
        # datasets.0.caption_dropout_rate uses list index "0"
        self.assertEqual(
            _parse_path("datasets.0.caption_dropout_rate"),
            ["config", "process", "0", "datasets", "0", "caption_dropout_rate"]
        )

    def test_relative_dataset_second(self):
        self.assertEqual(
            _parse_path("datasets.1.folder_path"),
            ["config", "process", "0", "datasets", "1", "folder_path"]
        )

    def test_absolute_config_name(self):
        self.assertEqual(_parse_path("config.name"), ["config", "name"])

    def test_absolute_config_process(self):
        self.assertEqual(_parse_path("config.process"), ["config", "process"])

    def test_absolute_meta_version(self):
        self.assertEqual(_parse_path("meta.version"), ["meta", "version"])

    def test_absolute_meta_name(self):
        self.assertEqual(_parse_path("meta.name"), ["meta", "name"])

    def test_single_key_relative(self):
        # A single-segment relative path (unusual but should work)
        self.assertEqual(_parse_path("trigger_word"), ["config", "process", "0", "trigger_word"])


# ─────────────────────────────────────────────────────────────────
# _set()
# ─────────────────────────────────────────────────────────────────

class TestSet(unittest.TestCase):

    def _base(self):
        """Minimal config dict mirroring the real toolkit structure."""
        return {
            "job": "extension",
            "config": {
                "name": "base_lora",
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": "/old/path",
                        "trigger_word": "mytoken",
                        "network": {"type": "lora", "linear": 16, "linear_alpha": 16},
                        "train": {
                            "lr": 0.001,
                            "batch_size": 1,
                            "steps": 500,
                            "optimizer": "adamw8bit",
                        },
                        "datasets": [
                            {
                                "folder_path": "/data/images",
                                "caption_dropout_rate": 0.05,
                                "resolution": [512, 768, 1024],
                            }
                        ],
                        "save": {"dtype": "bf16", "save_every": 250},
                    }
                ]
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }

    def test_set_float_lr(self):
        cfg = self._base()
        _set(cfg, _parse_path("train.lr"), 1e-4)
        self.assertAlmostEqual(cfg["config"]["process"][0]["train"]["lr"], 1e-4)

    def test_set_int_rank(self):
        cfg = self._base()
        _set(cfg, _parse_path("network.linear"), 32)
        self.assertEqual(cfg["config"]["process"][0]["network"]["linear"], 32)

    def test_set_int_batch_size(self):
        cfg = self._base()
        _set(cfg, _parse_path("train.batch_size"), 2)
        self.assertEqual(cfg["config"]["process"][0]["train"]["batch_size"], 2)

    def test_set_string_optimizer(self):
        cfg = self._base()
        _set(cfg, _parse_path("train.optimizer"), "prodigy")
        self.assertEqual(cfg["config"]["process"][0]["train"]["optimizer"], "prodigy")

    def test_set_nested_save(self):
        cfg = self._base()
        _set(cfg, _parse_path("save.save_every"), 500)
        self.assertEqual(cfg["config"]["process"][0]["save"]["save_every"], 500)

    def test_set_dataset_dropout(self):
        # datasets is a list; index 0 accessed via "0" string
        cfg = self._base()
        _set(cfg, _parse_path("datasets.0.caption_dropout_rate"), 0.1)
        self.assertAlmostEqual(
            cfg["config"]["process"][0]["datasets"][0]["caption_dropout_rate"], 0.1
        )

    def test_set_config_name_absolute(self):
        cfg = self._base()
        _set(cfg, ["config", "name"], "new_run_name")
        self.assertEqual(cfg["config"]["name"], "new_run_name")

    def test_set_does_not_affect_siblings(self):
        cfg = self._base()
        _set(cfg, _parse_path("train.lr"), 1e-5)
        # batch_size and steps must be untouched
        self.assertEqual(cfg["config"]["process"][0]["train"]["batch_size"], 1)
        self.assertEqual(cfg["config"]["process"][0]["train"]["steps"], 500)

    def test_set_does_not_affect_other_process_keys(self):
        cfg = self._base()
        _set(cfg, _parse_path("network.linear"), 64)
        # linear_alpha must be untouched
        self.assertEqual(cfg["config"]["process"][0]["network"]["linear_alpha"], 16)

    def test_set_training_folder(self):
        cfg = self._base()
        cfg["config"]["process"][0]["training_folder"] = "/sweep/runs"
        self.assertEqual(cfg["config"]["process"][0]["training_folder"], "/sweep/runs")

    def test_set_meta_version(self):
        cfg = self._base()
        _set(cfg, _parse_path("meta.version"), "2.0")
        self.assertEqual(cfg["meta"]["version"], "2.0")

    def test_set_trigger_word(self):
        cfg = self._base()
        _set(cfg, _parse_path("trigger_word"), "newtoken")
        self.assertEqual(cfg["config"]["process"][0]["trigger_word"], "newtoken")


# ─────────────────────────────────────────────────────────────────
# expand_param()
# ─────────────────────────────────────────────────────────────────

class TestExpandParam(unittest.TestCase):

    # --- values ---
    def test_values_floats(self):
        result = expand_param({"values": [1e-5, 5e-5, 1e-4]})
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1e-5)
        self.assertAlmostEqual(result[1], 5e-5)
        self.assertAlmostEqual(result[2], 1e-4)

    def test_values_strings(self):
        result = expand_param({"values": ["adamw8bit", "prodigy"]})
        self.assertEqual(result, ["adamw8bit", "prodigy"])

    def test_values_mixed(self):
        result = expand_param({"values": [1, 2, 4]})
        self.assertEqual(result, [1, 2, 4])

    def test_values_single(self):
        result = expand_param({"values": [42]})
        self.assertEqual(result, [42])

    def test_values_empty(self):
        result = expand_param({"values": []})
        self.assertEqual(result, [])

    # --- range integers ---
    def test_range_int_basic(self):
        result = expand_param({"range": {"from": 8, "to": 32, "step": 8}})
        self.assertEqual(result, [8, 16, 24, 32])

    def test_range_int_all_integers(self):
        result = expand_param({"range": {"from": 8, "to": 32, "step": 8}})
        for v in result:
            self.assertIsInstance(v, int, f"{v!r} should be int")

    def test_range_int_single_value(self):
        result = expand_param({"range": {"from": 16, "to": 16, "step": 8}})
        self.assertEqual(result, [16])

    def test_range_int_step1(self):
        result = expand_param({"range": {"from": 1, "to": 4, "step": 1}})
        self.assertEqual(result, [1, 2, 3, 4])

    # --- range floats ---
    def test_range_float_basic(self):
        result = expand_param({"range": {"from": 0.1, "to": 0.3, "step": 0.1}})
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 0.1)
        self.assertAlmostEqual(result[1], 0.2)
        self.assertAlmostEqual(result[2], 0.3)

    def test_range_float_all_floats(self):
        result = expand_param({"range": {"from": 0.1, "to": 0.3, "step": 0.1}})
        for v in result:
            self.assertIsInstance(v, float, f"{v!r} should be float")

    def test_range_float_boundary_inclusive(self):
        # Classic float arithmetic pitfall: 0.0 + 0.1*3 may not equal 0.3 exactly
        result = expand_param({"range": {"from": 0.0, "to": 0.3, "step": 0.1}})
        self.assertEqual(len(result), 4, f"Expected 4 values [0.0,0.1,0.2,0.3], got {result}")

    def test_range_lr_sweep(self):
        # Realistic LR range: 1e-5 to 1e-3 is not meaningful with step,
        # but let's test a values-based LR scenario via range of floats
        result = expand_param({"range": {"from": 0.0001, "to": 0.001, "step": 0.0003}})
        self.assertGreater(len(result), 0)
        self.assertAlmostEqual(result[0], 0.0001)

    # --- linspace ---
    def test_linspace_basic(self):
        result = expand_param({"linspace": {"from": 0.0, "to": 1.0, "n": 4}})
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.0 / 3)
        self.assertAlmostEqual(result[2], 2.0 / 3)
        self.assertAlmostEqual(result[3], 1.0)

    def test_linspace_n1(self):
        result = expand_param({"linspace": {"from": 0.5, "to": 2.0, "n": 1}})
        self.assertEqual(result, [0.5])

    def test_linspace_n2(self):
        result = expand_param({"linspace": {"from": 0.0, "to": 1.0, "n": 2}})
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 1.0)

    def test_linspace_endpoints(self):
        # First and last values must be exactly from and to
        result = expand_param({"linspace": {"from": 1e-5, "to": 1e-3, "n": 5}})
        self.assertAlmostEqual(result[0], 1e-5, places=12)
        self.assertAlmostEqual(result[-1], 1e-3, places=12)

    # --- error cases ---
    def test_range_step_zero_raises(self):
        with self.assertRaises(ValueError):
            expand_param({"range": {"from": 1, "to": 10, "step": 0}})

    def test_range_step_negative_raises(self):
        with self.assertRaises(ValueError):
            expand_param({"range": {"from": 10, "to": 1, "step": -1}})

    def test_linspace_n0_raises(self):
        with self.assertRaises(ValueError):
            expand_param({"linspace": {"from": 0, "to": 1, "n": 0}})

    def test_unknown_spec_raises(self):
        with self.assertRaises(ValueError):
            expand_param({"unknown_key": [1, 2, 3]})


# ─────────────────────────────────────────────────────────────────
# YAML round-trip
# ─────────────────────────────────────────────────────────────────

class TestYamlRoundTrip(unittest.TestCase):
    """
    Verify values survive  yaml.dump (our code)  →  yaml.safe_load (our load_yaml).

    The toolkit uses its own fixed_loader when reading generated configs, but
    standard yaml.safe_load covers the same float patterns and is available in tests.
    """

    def _roundtrip(self, data: dict) -> dict:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(data, f, allow_unicode=True)
            path = f.name
        try:
            return load_yaml(path)
        finally:
            os.unlink(path)

    def test_float_1e_minus_4(self):
        loaded = self._roundtrip({"lr": 1e-4})
        self.assertAlmostEqual(loaded["lr"], 1e-4, places=10)

    def test_float_1e_minus_5(self):
        loaded = self._roundtrip({"lr": 1e-5})
        self.assertAlmostEqual(loaded["lr"], 1e-5, places=10)

    def test_float_5e_minus_5(self):
        loaded = self._roundtrip({"lr": 5e-5})
        self.assertAlmostEqual(loaded["lr"], 5e-5, places=10)

    def test_float_small_decimal(self):
        loaded = self._roundtrip({"dropout": 0.05})
        self.assertAlmostEqual(loaded["dropout"], 0.05, places=10)

    def test_integer_rank(self):
        loaded = self._roundtrip({"linear": 32})
        self.assertEqual(loaded["linear"], 32)
        self.assertIsInstance(loaded["linear"], int)

    def test_integer_steps(self):
        loaded = self._roundtrip({"steps": 1000})
        self.assertEqual(loaded["steps"], 1000)
        self.assertIsInstance(loaded["steps"], int)

    def test_string_optimizer(self):
        loaded = self._roundtrip({"optimizer": "adamw8bit"})
        self.assertEqual(loaded["optimizer"], "adamw8bit")

    def test_bool_true(self):
        loaded = self._roundtrip({"gradient_checkpointing": True})
        self.assertIs(loaded["gradient_checkpointing"], True)

    def test_unicode_cyrillic_name(self):
        name = "объем_дс_Qwen2512__lr1e-4__linear32"
        loaded = self._roundtrip({"name": name})
        self.assertEqual(loaded["name"], name)

    def test_nested_structure(self):
        data = {
            "config": {
                "name": "test",
                "process": [
                    {
                        "train": {"lr": 1e-4, "steps": 1000},
                        "network": {"linear": 32},
                    }
                ],
            }
        }
        loaded = self._roundtrip(data)
        self.assertAlmostEqual(
            loaded["config"]["process"][0]["train"]["lr"], 1e-4, places=10
        )
        self.assertEqual(loaded["config"]["process"][0]["network"]["linear"], 32)

    def test_resolution_list(self):
        loaded = self._roundtrip({"resolution": [512, 768, 1024]})
        self.assertEqual(loaded["resolution"], [512, 768, 1024])


# ─────────────────────────────────────────────────────────────────
# training_folder path — must NOT be doubled by toolkit
# ─────────────────────────────────────────────────────────────────

class TestTrainingFolderNotDoubled(unittest.TestCase):
    """
    BaseTrainProcess computes:
        self.save_root = os.path.join(self.training_folder, self.name)
    where self.name == config.name == run_name_fs.

    If our script wrote training_folder = sweep/runs/run_name_fs, then:
        save_root = sweep/runs/run_name_fs/run_name_fs   ← WRONG

    If our script writes training_folder = sweep/runs (shared parent), then:
        save_root = sweep/runs/run_name_fs               ← correct
    """

    def _simulate_save_root(self, training_folder: str, run_name: str) -> str:
        """Reproduce the toolkit's path construction."""
        return os.path.join(training_folder, run_name)

    def test_shared_parent_gives_clean_path(self):
        sweep_out = "/home/user/sweeps/my_sweep"
        run_name  = "my_lora__lr1e-4__linear32"

        # What our fixed code writes
        training_folder = os.path.join(sweep_out, "runs")

        save_root = self._simulate_save_root(training_folder, run_name)

        # Use os.path.join for cross-platform comparison (backslash on Windows)
        expected = os.path.join(sweep_out, "runs", run_name)
        self.assertEqual(save_root, expected)

    def test_wrong_per_run_folder_would_double(self):
        """Demonstrate why per-run training_folder is wrong."""
        sweep_out = "/home/user/sweeps/my_sweep"
        run_name  = "my_lora__lr1e-4__linear32"

        # What the OLD (buggy) code wrote
        wrong_training_folder = os.path.join(sweep_out, "runs", run_name)
        wrong_save_root = self._simulate_save_root(wrong_training_folder, run_name)

        self.assertIn(
            run_name + os.sep + run_name,
            wrong_save_root,
            "This test confirms the old code produced doubled paths",
        )

    def test_each_run_gets_unique_save_root(self):
        """Multiple runs sharing the same training_folder still get separate roots."""
        sweep_out = "/home/user/sweeps/my_sweep"
        training_folder = os.path.join(sweep_out, "runs")

        run_names = [
            "my_lora__lr1e-5__linear8",
            "my_lora__lr1e-5__linear16",
            "my_lora__lr1e-4__linear8",
            "my_lora__lr1e-4__linear16",
        ]

        roots = [self._simulate_save_root(training_folder, n) for n in run_names]
        # All unique
        self.assertEqual(len(roots), len(set(roots)))
        # None of them overlap as a prefix of another
        for i, r1 in enumerate(roots):
            for j, r2 in enumerate(roots):
                if i != j:
                    self.assertFalse(
                        r2.startswith(r1 + os.sep),
                        f"{r1!r} is a prefix of {r2!r}",
                    )


# ─────────────────────────────────────────────────────────────────
# Full config generation (end-to-end simulation)
# ─────────────────────────────────────────────────────────────────

class TestFullConfigGeneration(unittest.TestCase):
    """
    Simulate exactly what run_sweep.py does for one run and verify the output.
    """

    BASE_CFG = {
        "job": "extension",
        "config": {
            "name": "test_lora",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": "/original/path",
                    "trigger_word": "mytoken",
                    "network": {"type": "lora", "linear": 16, "linear_alpha": 16},
                    "train": {
                        "lr": 0.001,
                        "batch_size": 1,
                        "steps": 500,
                        "optimizer": "adamw8bit",
                        "dtype": "bf16",
                    },
                    "datasets": [
                        {
                            "folder_path": "/data/images",
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "resolution": [512, 768, 1024],
                            "cache_latents_to_disk": True,
                        }
                    ],
                    "save": {"dtype": "bf16", "save_every": 250, "max_step_saves_to_keep": 4},
                    "model": {
                        "name_or_path": "black-forest-labs/FLUX.1-dev",
                        "is_flux": True,
                        "quantize": True,
                    },
                }
            ],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }

    def _apply(
        self,
        fixed_overrides: dict,
        sweep_params: list[tuple[str, object]],
        run_name: str = "test_lora__lr1e-4__linear32",
        sweep_out: str = "/sweep/my_sweep",
    ) -> dict:
        """Reproduce exactly what run_sweep.py does for one run."""
        run_cfg = copy.deepcopy(self.BASE_CFG)

        # Set config.name
        _set(run_cfg, ["config", "name"], run_name)

        # Set shared training_folder (fixed bug: shared parent, NOT per-run)
        run_cfg["config"]["process"][0]["training_folder"] = os.path.join(sweep_out, "runs")

        # Apply fixed overrides
        for dot_path, value in fixed_overrides.items():
            _set(run_cfg, _parse_path(dot_path), value)

        # Apply sweep params (win over fixed)
        for dot_path, value in sweep_params:
            _set(run_cfg, _parse_path(dot_path), value)

        return run_cfg

    # --- structure preservation ---

    def test_job_key_preserved(self):
        cfg = self._apply({}, [])
        self.assertEqual(cfg["job"], "extension")

    def test_process_type_preserved(self):
        cfg = self._apply({}, [])
        self.assertEqual(cfg["config"]["process"][0]["type"], "sd_trainer")

    def test_trigger_word_preserved(self):
        cfg = self._apply({}, [])
        self.assertEqual(cfg["config"]["process"][0]["trigger_word"], "mytoken")

    def test_dataset_path_preserved(self):
        cfg = self._apply({}, [])
        self.assertEqual(
            cfg["config"]["process"][0]["datasets"][0]["folder_path"],
            "/data/images",
        )

    def test_model_preserved(self):
        cfg = self._apply({}, [])
        self.assertEqual(
            cfg["config"]["process"][0]["model"]["name_or_path"],
            "black-forest-labs/FLUX.1-dev",
        )

    def test_meta_name_placeholder_kept(self):
        # meta.name must stay as "[name]" — the toolkit replaces it at load time
        cfg = self._apply({}, [])
        self.assertEqual(cfg["meta"]["name"], "[name]")

    def test_meta_version_preserved(self):
        cfg = self._apply({}, [])
        self.assertEqual(cfg["meta"]["version"], "1.0")

    # --- config.name ---

    def test_config_name_set(self):
        cfg = self._apply({}, [], run_name="test_lora__lr1e-4__linear32")
        self.assertEqual(cfg["config"]["name"], "test_lora__lr1e-4__linear32")

    # --- training_folder ---

    def test_training_folder_is_shared_runs_dir(self):
        cfg = self._apply({}, [], sweep_out="/sweep/my_sweep")
        tf = cfg["config"]["process"][0]["training_folder"]
        # Use os.path.join for cross-platform comparison
        self.assertEqual(tf, os.path.join("/sweep/my_sweep", "runs"))

    def test_training_folder_does_not_contain_run_name(self):
        run_name = "test_lora__lr1e-4__linear32"
        cfg = self._apply({}, [], run_name=run_name)
        tf = cfg["config"]["process"][0]["training_folder"]
        self.assertNotIn(run_name, tf)

    def test_toolkit_save_root_not_doubled(self):
        """Verify toolkit's path construction produces clean result."""
        run_name = "test_lora__lr1e-4__linear32"
        cfg = self._apply({}, [], run_name=run_name, sweep_out="/sweep/my_sweep")
        tf = cfg["config"]["process"][0]["training_folder"]
        # Simulate BaseTrainProcess: save_root = os.path.join(training_folder, name)
        simulated_name = cfg["config"]["name"]
        save_root = os.path.join(tf, simulated_name)
        self.assertEqual(save_root, os.path.join("/sweep/my_sweep", "runs", "test_lora__lr1e-4__linear32"))

    # --- sweep param application ---

    def test_lr_sweep_applied(self):
        cfg = self._apply({}, [("train.lr", 1e-4)])
        self.assertAlmostEqual(cfg["config"]["process"][0]["train"]["lr"], 1e-4)

    def test_rank_sweep_applied(self):
        cfg = self._apply({}, [("network.linear", 32)])
        self.assertEqual(cfg["config"]["process"][0]["network"]["linear"], 32)

    def test_batch_size_sweep_applied(self):
        cfg = self._apply({}, [("train.batch_size", 2)])
        self.assertEqual(cfg["config"]["process"][0]["train"]["batch_size"], 2)

    def test_dataset_dropout_sweep(self):
        cfg = self._apply({}, [("datasets.0.caption_dropout_rate", 0.1)])
        self.assertAlmostEqual(
            cfg["config"]["process"][0]["datasets"][0]["caption_dropout_rate"], 0.1
        )

    def test_optimizer_sweep_applied(self):
        cfg = self._apply({}, [("train.optimizer", "prodigy")])
        self.assertEqual(cfg["config"]["process"][0]["train"]["optimizer"], "prodigy")

    # --- fixed overrides ---

    def test_fixed_steps_override(self):
        cfg = self._apply({"train.steps": 1000}, [])
        self.assertEqual(cfg["config"]["process"][0]["train"]["steps"], 1000)

    def test_fixed_save_every_override(self):
        cfg = self._apply({"save.save_every": 500}, [])
        self.assertEqual(cfg["config"]["process"][0]["save"]["save_every"], 500)

    # --- sweep wins over fixed when keys conflict ---

    def test_sweep_overrides_fixed_on_same_key(self):
        cfg = self._apply(
            fixed_overrides={"train.lr": 1e-3},
            sweep_params=[("train.lr", 1e-5)],
        )
        # Sweep is applied after fixed, so sweep value must win
        self.assertAlmostEqual(cfg["config"]["process"][0]["train"]["lr"], 1e-5)

    # --- base config is not mutated ---

    def test_base_config_not_mutated(self):
        original_lr = self.BASE_CFG["config"]["process"][0]["train"]["lr"]
        self._apply({}, [("train.lr", 1e-4)])
        self.assertEqual(
            self.BASE_CFG["config"]["process"][0]["train"]["lr"],
            original_lr,
            "Base config must not be mutated (deepcopy required)",
        )

    def test_runs_are_independent(self):
        """Two runs from the same base must not share state."""
        cfg_a = self._apply({}, [("train.lr", 1e-5)], run_name="run_a")
        cfg_b = self._apply({}, [("train.lr", 1e-4)], run_name="run_b")
        self.assertAlmostEqual(cfg_a["config"]["process"][0]["train"]["lr"], 1e-5)
        self.assertAlmostEqual(cfg_b["config"]["process"][0]["train"]["lr"], 1e-4)


# ─────────────────────────────────────────────────────────────────
# YAML file generation (save_yaml + load_yaml round-trip)
# ─────────────────────────────────────────────────────────────────

class TestSaveLoadYaml(unittest.TestCase):
    """Verify save_yaml → load_yaml preserves the full config structure."""

    def test_full_config_round_trip(self):
        original = {
            "job": "extension",
            "config": {
                "name": "sweep__lr1e-4__linear32",
                "process": [
                    {
                        "type": "sd_trainer",
                        "training_folder": "/sweep/runs",
                        "network": {"linear": 32, "linear_alpha": 32},
                        "train": {"lr": 1e-4, "batch_size": 2, "steps": 1000},
                        "datasets": [
                            {
                                "folder_path": "/data",
                                "caption_dropout_rate": 0.05,
                                "resolution": [512, 768, 1024],
                            }
                        ],
                    }
                ],
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "generated.yaml")
            save_yaml(original, path)
            loaded = load_yaml(path)

        proc = loaded["config"]["process"][0]
        self.assertEqual(loaded["job"], "extension")
        self.assertEqual(loaded["config"]["name"], "sweep__lr1e-4__linear32")
        self.assertAlmostEqual(proc["train"]["lr"], 1e-4, places=10)
        self.assertEqual(proc["train"]["batch_size"], 2)
        self.assertIsInstance(proc["train"]["batch_size"], int)
        self.assertEqual(proc["network"]["linear"], 32)
        self.assertIsInstance(proc["network"]["linear"], int)
        self.assertEqual(proc["datasets"][0]["resolution"], [512, 768, 1024])
        self.assertEqual(loaded["meta"]["name"], "[name]")

    def test_save_yaml_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "a", "b", "c", "config.yaml")
            save_yaml({"key": "value"}, nested)
            self.assertTrue(os.path.exists(nested))


# ─────────────────────────────────────────────────────────────────
# linspace integer rounding (new finding: NetworkConfig has no int() cast)
# ─────────────────────────────────────────────────────────────────

class TestLinspaceIntegerRounding(unittest.TestCase):
    """
    NetworkConfig stores self.linear = linear with NO int() conversion.
    Downstream LoRA code calls torch.zeros(rank, ...) etc — a float rank raises
    TypeError.  When both linspace endpoints are ints we must return ints.
    """

    def test_int_endpoints_return_ints(self):
        result = expand_param({"linspace": {"from": 8, "to": 32, "n": 4}})
        for v in result:
            self.assertIsInstance(v, int, f"Expected int, got {type(v).__name__}: {v!r}")

    def test_int_endpoints_correct_values(self):
        result = expand_param({"linspace": {"from": 8, "to": 32, "n": 4}})
        self.assertEqual(result, [8, 16, 24, 32])

    def test_int_endpoints_n2(self):
        result = expand_param({"linspace": {"from": 16, "to": 64, "n": 2}})
        self.assertEqual(result, [16, 64])
        for v in result:
            self.assertIsInstance(v, int)

    def test_float_endpoints_return_floats(self):
        result = expand_param({"linspace": {"from": 1e-5, "to": 1e-3, "n": 3}})
        for v in result:
            self.assertIsInstance(v, float, f"Expected float, got {type(v).__name__}: {v!r}")

    def test_mixed_endpoints_return_floats(self):
        # one int, one float → floats
        result = expand_param({"linspace": {"from": 0, "to": 1.0, "n": 3}})
        # start is int but stop is float; result may be float — no crash is the key requirement
        self.assertEqual(len(result), 3)

    def test_int_batch_size_linspace_safe(self):
        """batch_size swept with linspace must be int so DataLoader accepts it."""
        result = expand_param({"linspace": {"from": 1, "to": 4, "n": 4}})
        self.assertEqual(result, [1, 2, 3, 4])
        for v in result:
            self.assertIsInstance(v, int)

    def test_int_rank_linspace_written_as_int_in_yaml(self):
        """Verify int rank survives yaml.dump → yaml.safe_load as int (not float)."""
        rank = expand_param({"linspace": {"from": 8, "to": 32, "n": 4}})[1]  # 16
        self.assertIsInstance(rank, int)
        cfg = {"network": {"linear": rank}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(cfg, f)
            path = f.name
        try:
            loaded = load_yaml(path)
            self.assertIsInstance(
                loaded["network"]["linear"],
                int,
                "rank must survive YAML round-trip as int",
            )
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────
# preprocess_dataset_raw_config expansion (toolkit behaviour)
# ─────────────────────────────────────────────────────────────────

class TestDatasetResolutionExpansion(unittest.TestCase):
    """
    The toolkit's preprocess_dataset_raw_config() expands a dataset with
    resolution: [512, 768, 1024]  into THREE dataset configs (one per res).
    Our dot-path targets the raw pre-expansion config, so the swept value
    is inherited by all three expanded entries.
    """

    def _simulate_preprocess(self, raw_datasets: list) -> list:
        """Reproduce preprocess_dataset_raw_config logic inline (no toolkit import)."""
        new_config = []
        for dataset in raw_datasets:
            resolution = dataset.get("resolution", 512)
            resolution_list = resolution if isinstance(resolution, list) else [resolution]
            for res in resolution_list:
                dc = dataset.copy()
                dc["resolution"] = res
                new_config.append(dc)
        return new_config

    def test_expansion_count(self):
        raw = [{"folder_path": "/data", "resolution": [512, 768, 1024], "caption_dropout_rate": 0.1}]
        expanded = self._simulate_preprocess(raw)
        self.assertEqual(len(expanded), 3)

    def test_swept_value_propagates_to_all_resolutions(self):
        """A swept caption_dropout_rate on the raw dataset applies to all res variants."""
        raw = [{"folder_path": "/data", "resolution": [512, 768, 1024], "caption_dropout_rate": 0.1}]
        expanded = self._simulate_preprocess(raw)
        for entry in expanded:
            self.assertAlmostEqual(entry["caption_dropout_rate"], 0.1)

    def test_resolutions_split_correctly(self):
        raw = [{"folder_path": "/data", "resolution": [512, 768, 1024]}]
        expanded = self._simulate_preprocess(raw)
        self.assertEqual([e["resolution"] for e in expanded], [512, 768, 1024])

    def test_single_resolution_not_expanded(self):
        raw = [{"folder_path": "/data", "resolution": 512}]
        expanded = self._simulate_preprocess(raw)
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0]["resolution"], 512)

    def test_dotpath_targets_pre_expansion_raw_config(self):
        """
        Our _set on datasets.0.caption_dropout_rate modifies the RAW config entry.
        After the toolkit expands it, the value appears in all resolution variants.
        """
        raw_cfg = {
            "config": {
                "name": "test",
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": "/sweep/runs",
                    "datasets": [{
                        "folder_path": "/data",
                        "resolution": [512, 768, 1024],
                        "caption_dropout_rate": 0.05,
                    }]
                }]
            },
            "meta": {"name": "[name]"}
        }
        # Apply sweep via our mechanism
        _set(raw_cfg, _parse_path("datasets.0.caption_dropout_rate"), 0.15)

        # Confirm raw config is updated
        self.assertAlmostEqual(
            raw_cfg["config"]["process"][0]["datasets"][0]["caption_dropout_rate"], 0.15
        )

        # Simulate toolkit expansion → value must appear in all entries
        raw_datasets = raw_cfg["config"]["process"][0]["datasets"]
        expanded = self._simulate_preprocess(raw_datasets)
        for entry in expanded:
            self.assertAlmostEqual(entry["caption_dropout_rate"], 0.15)


# ─────────────────────────────────────────────────────────────────
# JSON-serializability (preprocess_config does json.dumps on whole config)
# ─────────────────────────────────────────────────────────────────

class TestJsonSerializability(unittest.TestCase):
    """
    toolkit/config.py preprocess_config() does:
        config_string = json.dumps(config)
        config_string = config_string.replace("[name]", name)
        config = json.loads(config_string, object_pairs_hook=OrderedDict)

    Our generated config (after yaml.safe_load) must survive this round-trip
    with all values intact.
    """

    FULL_CFG = {
        "job": "extension",
        "config": {
            "name": "sweep__lr1e-4__linear32",
            "process": [{
                "type": "sd_trainer",
                "training_folder": "/sweep/runs",
                "trigger_word": "mytoken",
                "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
                "train": {
                    "lr": 1e-4,
                    "batch_size": 2,
                    "steps": 1000,
                    "optimizer": "adamw8bit",
                    "dtype": "bf16",
                    "gradient_checkpointing": True,
                },
                "datasets": [{
                    "folder_path": "/data/images",
                    "caption_ext": "txt",
                    "caption_dropout_rate": 0.05,
                    "resolution": [512, 768, 1024],
                    "cache_latents_to_disk": True,
                }],
                "save": {"dtype": "bf16", "save_every": 250},
                "model": {
                    "name_or_path": "black-forest-labs/FLUX.1-dev",
                    "is_flux": True,
                    "quantize": True,
                },
            }]
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }

    def test_json_serializable(self):
        import json
        try:
            s = json.dumps(self.FULL_CFG)
        except (TypeError, ValueError) as e:
            self.fail(f"Config is not JSON-serializable: {e}")
        self.assertIsInstance(s, str)

    def test_json_round_trip_float(self):
        import json
        s = json.dumps(self.FULL_CFG)
        s = s.replace("[name]", "sweep__lr1e-4__linear32")
        from collections import OrderedDict
        loaded = json.loads(s, object_pairs_hook=OrderedDict)
        lr = loaded["config"]["process"][0]["train"]["lr"]
        self.assertAlmostEqual(lr, 1e-4, places=10)

    def test_json_round_trip_int(self):
        import json
        s = json.dumps(self.FULL_CFG)
        s = s.replace("[name]", "sweep__lr1e-4__linear32")
        from collections import OrderedDict
        loaded = json.loads(s, object_pairs_hook=OrderedDict)
        rank = loaded["config"]["process"][0]["network"]["linear"]
        self.assertEqual(rank, 32)

    def test_json_round_trip_bool(self):
        import json
        s = json.dumps(self.FULL_CFG)
        s = s.replace("[name]", "sweep__lr1e-4__linear32")
        from collections import OrderedDict
        loaded = json.loads(s, object_pairs_hook=OrderedDict)
        gc = loaded["config"]["process"][0]["train"]["gradient_checkpointing"]
        self.assertIs(gc, True)

    def test_json_round_trip_name_replaced(self):
        import json
        s = json.dumps(self.FULL_CFG)
        s = s.replace("[name]", "my_run_name")
        from collections import OrderedDict
        loaded = json.loads(s, object_pairs_hook=OrderedDict)
        self.assertEqual(loaded["meta"]["name"], "my_run_name")

    def test_cyrillic_name_json_safe(self):
        """Cyrillic in config.name must survive json.dumps/loads."""
        import json
        cfg = copy.deepcopy(self.FULL_CFG)
        cyrillic_name = "объем_дс_Qwen2512__lr1e-4__linear32"
        cfg["config"]["name"] = cyrillic_name
        s = json.dumps(cfg, ensure_ascii=False)
        from collections import OrderedDict
        loaded = json.loads(s, object_pairs_hook=OrderedDict)
        self.assertEqual(loaded["config"]["name"], cyrillic_name)

    def test_run_name_fs_cannot_contain_name_placeholder(self):
        """run_name_fs must never contain '[name]' — that would cause re-expansion."""
        # Construct plausible run names and verify none contain [name]
        base_names = ["my_lora", "объем_дс_Qwen2512", "flux_test_v1"]
        suffixes = ["lr1e-4__linear32", "lr5e-5__linear16", "lr0.0001__linear8"]
        for base in base_names:
            for suffix in suffixes:
                run_name_fs = f"{base}__{suffix}".replace(" ", "_").replace("/", "-")
                self.assertNotIn("[name]", run_name_fs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
