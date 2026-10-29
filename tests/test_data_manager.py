"""Tests for src/data_manager.py.

Every file operation happens inside a ``tempfile.TemporaryDirectory`` so real
project data is never touched.

Run from the project root with either:
    python -m unittest discover -s tests -v
    python -m pytest tests -v        (if pytest is installed)

All test data below is hand-written fixture data for exercising the code.
It is not real sensor data and makes no claim about banana spoilage.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_manager import (  # noqa: E402
    DataOrigin,
    assess_training_readiness,
    classify_row_origins,
    load_dataset,
    mark_as_synthetic,
    prepare_for_analysis,
    save_dataset,
    separate_synthetic_rows,
    summarize_dataset,
)
from src.data_validator import validate_sensor_data  # noqa: E402

VALID_CSV = (
    "timestamp,sample_id,gas_raw,temperature_c,humidity_percent,source,spoilage_label\n"
    "2026-10-01 09:00:00,S1,310,24.5,61.0,bench_rig_1,fresh\n"
    "2026-10-01 10:00:00,S2,325,25.0,62.5,bench_rig_1,fresh\n"
    "2026-10-01 11:00:00,S3,340,25.2,63.0,bench_rig_1,ripe\n"
)


def make_frame(**overrides) -> pd.DataFrame:
    data = {
        "timestamp": ["2026-10-01 09:00:00", "2026-10-01 10:00:00", "2026-10-01 11:00:00"],
        "sample_id": ["S1", "S2", "S3"],
        "gas_raw": [310, 325, 340],
        "temperature_c": [24.5, 25.0, 25.2],
        "humidity_percent": [61.0, 62.5, 63.0],
        "source": ["bench_rig_1"] * 3,
        "spoilage_label": ["fresh", "fresh", "ripe"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
        return path


class TestLoadDataset(TempDirTestCase):
    def test_load_valid_csv(self):
        result = load_dataset(self.write("ok.csv", VALID_CSV))
        self.assertTrue(result.loaded, result.errors)
        self.assertTrue(result.is_valid, result.validation.format_report())
        self.assertEqual(len(result.data), 3)

    def test_missing_file(self):
        result = load_dataset(self.root / "nope.csv")
        self.assertFalse(result.loaded)
        self.assertIsNone(result.data)
        self.assertIn("File not found", result.errors[0])

    def test_directory_instead_of_file(self):
        result = load_dataset(self.root)
        self.assertFalse(result.loaded)

    def test_empty_file(self):
        result = load_dataset(self.write("empty.csv", ""))
        self.assertFalse(result.loaded)
        self.assertIn("empty", result.errors[0].lower())

    def test_header_only_is_loaded_but_invalid(self):
        result = load_dataset(self.write("header.csv", "timestamp,gas_raw\n"))
        self.assertTrue(result.loaded)
        self.assertFalse(result.is_valid)
        self.assertTrue(result.validation.has_error("empty_dataset"))

    def test_malformed_csv_too_many_fields(self):
        text = "timestamp,gas_raw\n2026-10-01 09:00:00,300\n2026-10-01 10:00:00,310,999,extra\n"
        result = load_dataset(self.write("bad.csv", text))
        self.assertFalse(result.loaded)
        self.assertTrue(any("Malformed CSV" in e and "line" in e for e in result.errors), result.errors)

    def test_malformed_csv_unclosed_quote(self):
        text = 'timestamp,gas_raw\n"2026-10-01 09:00:00,300\n'
        result = load_dataset(self.write("quote.csv", text))
        self.assertFalse(result.is_valid)  # must not crash, must not be accepted

    def test_short_rows_warn_and_become_missing(self):
        text = "timestamp,gas_raw,temperature_c\n2026-10-01 09:00:00,300,25\n2026-10-01 10:00:00\n"
        result = load_dataset(self.write("short.csv", text))
        self.assertTrue(result.loaded)
        self.assertTrue(any("fewer fields" in w for w in result.warnings))
        self.assertTrue(result.validation.has_error("missing_required_value"))
        self.assertEqual(len(result.data), 2)  # row kept, not dropped

    def test_duplicate_header(self):
        result = load_dataset(self.write("dup.csv", "timestamp,gas_raw,gas_raw\n2026-10-01,1,2\n"))
        self.assertFalse(result.loaded)
        self.assertTrue(any("Duplicate column" in e for e in result.errors))

    def test_binary_file(self):
        path = self.root / "binary.csv"
        path.write_bytes(b"\xff\xfe\x00\x81\x82\x83\x00\x00")
        result = load_dataset(path)
        self.assertFalse(result.loaded)

    def test_excel_bom_handled(self):
        result = load_dataset(self.write("bom.csv", VALID_CSV, encoding="utf-8-sig"))
        self.assertTrue(result.is_valid)
        self.assertIn("timestamp", result.data.columns)

    def test_invalid_records_kept_and_reported(self):
        text = VALID_CSV + "2026-10-01 12:00:00,S4,-20,25.0,140,bench_rig_1,ripe\n"
        result = load_dataset(self.write("invalid.csv", text))
        self.assertTrue(result.loaded)
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.data), 4)
        self.assertEqual(result.validation.invalid_rows, (3,))

    def test_non_csv_extension_warns(self):
        result = load_dataset(self.write("data.txt", VALID_CSV))
        self.assertTrue(result.loaded)
        self.assertTrue(any(".csv" in w for w in result.warnings))

    def test_validate_false_skips_validation(self):
        result = load_dataset(self.write("ok.csv", VALID_CSV), validate=False)
        self.assertIsNone(result.validation)
        self.assertTrue(result.is_valid)

    def test_expected_origin_real_flags_synthetic(self):
        text = VALID_CSV.replace("bench_rig_1,ripe", "synthetic_demo,ripe")
        result = load_dataset(self.write("mixed.csv", text), expected_origin="real")
        self.assertTrue(result.validation.has_error("synthetic_rows_in_real_data"))


class TestSaveDataset(TempDirTestCase):
    def test_save_real_data(self):
        dest = self.root / "data" / "raw" / "run1.csv"
        result = save_dataset(make_frame(), dest, origin="real")
        self.assertTrue(result.saved, result.errors)
        self.assertTrue(dest.exists())
        reloaded = load_dataset(dest)
        self.assertTrue(reloaded.is_valid)
        self.assertEqual(len(reloaded.data), 3)

    def test_creates_directories(self):
        dest = self.root / "a" / "b" / "c" / "out.csv"
        self.assertTrue(save_dataset(make_frame(), dest, origin="real").saved)

    def test_create_dirs_false(self):
        dest = self.root / "missing" / "out.csv"
        result = save_dataset(make_frame(), dest, origin="real", create_dirs=False)
        self.assertFalse(result.saved)
        self.assertFalse(dest.exists())

    def test_does_not_overwrite_by_default(self):
        dest = self.write("existing.csv", "original contents\n")
        result = save_dataset(make_frame(), dest, origin="real")
        self.assertFalse(result.saved)
        self.assertTrue(any("already exists" in e for e in result.errors))
        self.assertEqual(dest.read_text(encoding="utf-8"), "original contents\n")

    def test_overwrite_when_explicit(self):
        dest = self.write("existing.csv", "original contents\n")
        result = save_dataset(make_frame(), dest, origin="real", overwrite=True)
        self.assertTrue(result.saved, result.errors)
        self.assertIn("gas_raw", dest.read_text(encoding="utf-8"))
        self.assertEqual([p.name for p in self.root.iterdir()], ["existing.csv"])  # no temp left

    def test_invalid_data_not_saved(self):
        dest = self.root / "bad.csv"
        result = save_dataset(make_frame(humidity_percent=[50, 150, 60]), dest, origin="real")
        self.assertFalse(result.saved)
        self.assertFalse(dest.exists())
        self.assertFalse(result.validation.is_valid)

    def test_allow_invalid_saves_all_rows_unchanged(self):
        dest = self.root / "raw_log.csv"
        df = make_frame(humidity_percent=[50, 150, 60])
        result = save_dataset(df, dest, origin="real", allow_invalid=True)
        self.assertTrue(result.saved)
        self.assertEqual(len(pd.read_csv(dest)), 3)
        self.assertTrue(result.warnings)

    def test_empty_dataframe_not_saved(self):
        result = save_dataset(make_frame().iloc[0:0], self.root / "e.csv", origin="real")
        self.assertFalse(result.saved)

    def test_non_csv_destination_rejected(self):
        result = save_dataset(make_frame(), self.root / "out.xlsx", origin="real")
        self.assertFalse(result.saved)

    def test_destination_is_directory(self):
        (self.root / "folder.csv").mkdir()
        result = save_dataset(make_frame(), self.root / "folder.csv", origin="real")
        self.assertFalse(result.saved)

    def test_origin_is_required_and_checked(self):
        with self.assertRaises(TypeError):
            save_dataset(make_frame(), self.root / "x.csv")  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            save_dataset(make_frame(), self.root / "x.csv", origin="maybe")

    def test_not_a_dataframe(self):
        result = save_dataset([1, 2, 3], self.root / "x.csv", origin="real")  # type: ignore[arg-type]
        self.assertFalse(result.saved)


class TestSyntheticSeparation(TempDirTestCase):
    def synthetic_frame(self) -> pd.DataFrame:
        return mark_as_synthetic(make_frame().drop(columns=["source"]))

    def test_synthetic_not_saved_into_data_raw(self):
        dest = self.root / "data" / "raw" / "demo.csv"
        result = save_dataset(self.synthetic_frame(), dest, origin="synthetic")
        self.assertFalse(result.saved)
        self.assertFalse(dest.exists())
        self.assertTrue(any("data/raw" in e for e in result.errors))

    def test_synthetic_into_raw_case_insensitive(self):
        dest = self.root / "Data" / "RAW" / "demo.csv"
        self.assertFalse(save_dataset(self.synthetic_frame(), dest, origin="synthetic").saved)

    def test_synthetic_into_raw_blocked_even_with_allow_invalid(self):
        dest = self.root / "data" / "raw" / "demo.csv"
        result = save_dataset(self.synthetic_frame(), dest, origin="synthetic", allow_invalid=True)
        self.assertFalse(result.saved)

    def test_synthetic_into_raw_only_with_explicit_override(self):
        dest = self.root / "data" / "raw" / "demo.csv"
        result = save_dataset(self.synthetic_frame(), dest, origin="synthetic",
                              allow_synthetic_in_raw=True)
        self.assertTrue(result.saved, result.errors)

    def test_synthetic_saved_into_demo(self):
        dest = self.root / "data" / "demo" / "synthetic_demo.csv"
        result = save_dataset(self.synthetic_frame(), dest, origin=DataOrigin.SYNTHETIC)
        self.assertTrue(result.saved, result.errors)
        saved = pd.read_csv(dest)
        self.assertTrue((saved["source"] == "synthetic_demo").all())

    def test_unmarked_synthetic_rejected(self):
        dest = self.root / "data" / "demo" / "demo.csv"
        result = save_dataset(make_frame().drop(columns=["source"]), dest, origin="synthetic")
        self.assertFalse(result.saved)

    def test_synthetic_rows_cannot_be_saved_as_real(self):
        df = make_frame(source=["bench_rig_1", "synthetic_demo", "bench_rig_1"])
        dest = self.root / "data" / "processed" / "train.csv"
        for allow in (False, True):
            result = save_dataset(df, dest, origin="real", allow_invalid=allow)
            self.assertFalse(result.saved)
        self.assertFalse(dest.exists())

    def test_mark_as_synthetic_does_not_modify_input(self):
        df = make_frame().drop(columns=["source"])
        marked = mark_as_synthetic(df)
        self.assertNotIn("source", df.columns)
        self.assertTrue((marked["source"] == "synthetic_demo").all())

    def test_mark_as_synthetic_fills_blanks_only(self):
        df = make_frame(source=["simulated", None, ""])
        marked = mark_as_synthetic(df)
        self.assertEqual(marked["source"].tolist(), ["simulated", "synthetic_demo", "synthetic_demo"])

    def test_mark_as_synthetic_refuses_to_relabel_real_rows(self):
        with self.assertRaises(ValueError):
            mark_as_synthetic(make_frame())

    def test_mark_as_synthetic_requires_synthetic_label(self):
        with self.assertRaises(ValueError):
            mark_as_synthetic(make_frame().drop(columns=["source"]), source_label="lab")

    def test_classify_and_separate(self):
        df = make_frame(source=["bench_rig_1", "synthetic_demo", None])
        self.assertEqual(classify_row_origins(df).tolist(), ["declared", "synthetic", "unspecified"])
        non_synth, synth = separate_synthetic_rows(df)
        self.assertEqual(len(non_synth) + len(synth), len(df))
        self.assertEqual(synth["source"].tolist(), ["synthetic_demo"])

    def test_no_source_column_is_unspecified_not_real(self):
        origins = classify_row_origins(make_frame().drop(columns=["source"]))
        self.assertTrue((origins == "unspecified").all())


class TestSummaries(unittest.TestCase):
    def test_summary_basic(self):
        s = summarize_dataset(make_frame())
        self.assertEqual((s.row_count, s.column_count), (3, 7))
        self.assertEqual(s.missing_required_columns, [])
        self.assertEqual(s.sensor_statistics["gas_raw"]["min"], 310.0)
        self.assertEqual(s.sensor_statistics["gas_raw"]["max"], 340.0)
        self.assertEqual(s.timestamp_range["start"], "2026-10-01T09:00:00")
        self.assertEqual(s.timestamp_range["end"], "2026-10-01T11:00:00")
        self.assertEqual(s.source_counts, {"bench_rig_1": 3})
        self.assertEqual(s.label_counts, {"fresh": 2, "ripe": 1})
        self.assertTrue(s.has_labels)

    def test_summary_is_json_serialisable(self):
        json.dumps(summarize_dataset(make_frame()).to_dict())

    def test_summary_empty_dataframe(self):
        s = summarize_dataset(pd.DataFrame())
        self.assertEqual(s.row_count, 0)
        self.assertEqual(s.sensor_statistics, {})
        self.assertIsNone(s.timestamp_range)
        self.assertEqual(s.missing_required_columns, ["timestamp", "gas_raw"])

    def test_summary_columns_no_rows(self):
        s = summarize_dataset(make_frame().iloc[0:0])
        self.assertEqual(s.sensor_statistics["gas_raw"]["count"], 0)
        self.assertIsNone(s.sensor_statistics["gas_raw"]["mean"])
        self.assertIsNone(s.timestamp_range["start"])

    def test_summary_messy_values(self):
        df = make_frame(gas_raw=[300, "abc", None], timestamp=["2026-10-01 09:00", "bad", None])
        s = summarize_dataset(df)
        self.assertEqual(s.sensor_statistics["gas_raw"]["count"], 1)
        self.assertEqual(s.sensor_statistics["gas_raw"]["non_numeric_count"], 1)
        self.assertEqual(s.missing_values["gas_raw"], 1)
        self.assertEqual(s.timestamp_range["invalid_count"], 1)

    def test_summary_without_optional_columns(self):
        s = summarize_dataset(make_frame()[["timestamp", "gas_raw"]])
        self.assertFalse(s.has_labels)
        self.assertIsNone(s.label_counts)
        self.assertIsNone(s.source_counts)
        self.assertEqual(s.origin_counts["unspecified"], 3)

    def test_summary_rejects_non_dataframe(self):
        with self.assertRaises(TypeError):
            summarize_dataset("data.csv")  # type: ignore[arg-type]


class TestReadinessAndPreparation(unittest.TestCase):
    def test_labelled_real_data_structurally_ready(self):
        r = assess_training_readiness(make_frame())
        self.assertTrue(r.suitable_for_exploration)
        self.assertTrue(r.suitable_for_supervised_training, r.blocking_reasons)
        self.assertIn("not a validated", r.disclaimer)

    def test_unlabelled_data_exploration_only(self):
        r = assess_training_readiness(make_frame().drop(columns=["spoilage_label"]))
        self.assertTrue(r.suitable_for_exploration)
        self.assertFalse(r.suitable_for_supervised_training)

    def test_synthetic_rows_block_training(self):
        r = assess_training_readiness(make_frame(source=["bench_rig_1", "synthetic_demo", "bench_rig_1"]))
        self.assertFalse(r.suitable_for_supervised_training)
        self.assertEqual(r.synthetic_row_count, 1)

    def test_single_class_blocks_training(self):
        r = assess_training_readiness(make_frame(spoilage_label=["fresh"] * 3))
        self.assertFalse(r.suitable_for_supervised_training)

    def test_unspecified_origin_is_a_caution(self):
        r = assess_training_readiness(make_frame().drop(columns=["source"]))
        self.assertTrue(r.cautions)
        self.assertEqual(r.unspecified_origin_row_count, 3)

    def test_prepare_for_analysis_converts_types(self):
        df = make_frame(gas_raw=["310", " 325", "340"])
        typed = prepare_for_analysis(df, validate_sensor_data(df))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(typed["timestamp"]))
        self.assertEqual(typed["gas_raw"].tolist(), [310.0, 325.0, 340.0])
        self.assertEqual(df["gas_raw"].tolist(), ["310", " 325", "340"])  # input untouched

    def test_prepare_for_analysis_refuses_invalid(self):
        df = make_frame(gas_raw=[1, "abc", 3])
        with self.assertRaises(ValueError):
            prepare_for_analysis(df, validate_sensor_data(df))


if __name__ == "__main__":
    unittest.main()
