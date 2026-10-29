"""Tests for src/data_validator.py.

Run from the project root with either:
    python -m unittest discover -s tests -v
    python -m pytest tests -v        (if pytest is installed)

All test data below is hand-written fixture data for exercising the code.
It is not real sensor data and makes no claim about banana spoilage.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_validator import (  # noqa: E402
    SCIENTIFIC_DISCLAIMER,
    ValidationConfig,
    ValidationResult,
    check_origin_consistency,
    is_synthetic_source,
    missing_mask,
    parse_timestamps,
    validate_sensor_data,
)


def make_frame(**overrides) -> pd.DataFrame:
    """Return a small structurally valid fixture; keyword args replace columns."""
    data = {
        "timestamp": ["2026-10-01 09:00:00", "2026-10-01 10:00:00", "2026-10-01 11:00:00"],
        "sample_id": ["S1", "S2", "S3"],
        "gas_raw": [310, 325, 340],
        "temperature_c": [24.5, 25.0, 25.2],
        "humidity_percent": [61.0, 62.5, 63.0],
        "source": ["bench_rig_1", "bench_rig_1", "bench_rig_1"],
        "spoilage_label": ["fresh", "fresh", "ripe"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestValidInput(unittest.TestCase):
    def test_valid_dataset_passes(self):
        result = validate_sensor_data(make_frame())
        self.assertTrue(result.is_valid, result.format_report())
        self.assertEqual(result.rows_checked, 3)
        self.assertEqual(result.invalid_row_count, 0)
        self.assertEqual(result.errors, [])

    def test_only_required_columns_is_valid(self):
        df = make_frame()[["timestamp", "gas_raw"]]
        result = validate_sensor_data(df)
        self.assertTrue(result.is_valid, result.format_report())

    def test_input_is_not_modified(self):
        df = make_frame(gas_raw=[1, -5, "abc"])
        before = df.copy()
        validate_sensor_data(df)
        pd.testing.assert_frame_equal(df, before)

    def test_disclaimer_always_present_for_dataframes(self):
        self.assertIn(SCIENTIFIC_DISCLAIMER, validate_sensor_data(make_frame()).notes)

    def test_result_to_dict_is_plain(self):
        d = validate_sensor_data(make_frame(gas_raw=[1, -2, 3])).to_dict()
        self.assertFalse(d["is_valid"])
        self.assertEqual(d["invalid_rows"], [1])
        self.assertIsInstance(d["errors"][0], dict)

    def test_non_default_index_reports_positions(self):
        df = make_frame(gas_raw=[1, -2, 3])
        df.index = [100, 200, 300]
        self.assertEqual(validate_sensor_data(df).invalid_rows, (1,))


class TestStructure(unittest.TestCase):
    def test_not_a_dataframe(self):
        for bad in (None, [1, 2], {"timestamp": []}, "file.csv"):
            result = validate_sensor_data(bad)
            self.assertFalse(result.is_valid)
            self.assertTrue(result.has_error("not_a_dataframe"))
            self.assertIsNone(result.invalid_row_count)

    def test_empty_dataframe(self):
        result = validate_sensor_data(pd.DataFrame())
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_error("empty_dataset"))

    def test_columns_but_no_rows(self):
        result = validate_sensor_data(make_frame().iloc[0:0])
        self.assertTrue(result.has_error("empty_dataset"))
        self.assertEqual(result.rows_checked, 0)

    def test_missing_required_column(self):
        result = validate_sensor_data(make_frame().drop(columns=["gas_raw"]))
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_error("missing_required_column"))
        self.assertTrue(any("gas_raw" in m for m in result.error_messages()))

    def test_missing_column_hint_for_case_mismatch(self):
        df = make_frame().rename(columns={"timestamp": "Timestamp"})
        msgs = validate_sensor_data(df).error_messages()
        self.assertTrue(any("'Timestamp'" in m for m in msgs), msgs)

    def test_optional_columns_not_required(self):
        df = make_frame().drop(columns=["temperature_c", "humidity_percent", "sample_id", "source"])
        self.assertTrue(validate_sensor_data(df).is_valid)

    def test_duplicate_column_names(self):
        df = pd.DataFrame([[1, 2, 3]], columns=["timestamp", "gas_raw", "gas_raw"])
        result = validate_sensor_data(df)
        self.assertTrue(result.has_error("duplicate_columns"))
        self.assertIsNone(result.invalid_rows)

    def test_custom_required_columns(self):
        config = ValidationConfig(required_columns=("timestamp", "gas_raw", "temperature_c"))
        df = make_frame().drop(columns=["temperature_c"])
        self.assertTrue(validate_sensor_data(df, config).has_error("missing_required_column"))

    def test_unknown_columns_noted_not_rejected(self):
        df = make_frame(notes=["a", "b", "c"])
        result = validate_sensor_data(df)
        self.assertTrue(result.is_valid)
        self.assertTrue(any("notes" in n for n in result.notes))


class TestMissingValues(unittest.TestCase):
    def test_missing_required_value_is_error(self):
        result = validate_sensor_data(make_frame(gas_raw=[300, None, 320]))
        self.assertTrue(result.has_error("missing_required_value"))
        self.assertEqual(result.invalid_rows, (1,))

    def test_blank_string_counts_as_missing(self):
        result = validate_sensor_data(make_frame(timestamp=["2026-10-01 09:00", "   ", "2026-10-01 11:00"]))
        self.assertTrue(result.has_error("missing_required_value"))

    def test_missing_optional_value_is_warning(self):
        result = validate_sensor_data(make_frame(temperature_c=[24.0, None, 25.0]))
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warning("missing_optional_value"))

    def test_missing_mask_helper(self):
        mask = missing_mask(pd.Series(["a", "", " ", None, float("nan"), 0]))
        self.assertEqual(mask.tolist(), [False, True, True, True, True, False])


class TestTimestamps(unittest.TestCase):
    def test_invalid_timestamp_format(self):
        df = make_frame(timestamp=["2026-10-01 09:00:00", "01/10/2026 10:00", "not a date"])
        result = validate_sensor_data(df)
        self.assertTrue(result.has_error("invalid_timestamp"))
        self.assertEqual(result.invalid_rows, (1, 2))

    def test_custom_timestamp_format(self):
        df = make_frame(timestamp=["01/10/2026 09:00", "01/10/2026 10:00", "01/10/2026 11:00"])
        config = ValidationConfig(timestamp_format="%d/%m/%Y %H:%M")
        self.assertTrue(validate_sensor_data(df, config).is_valid)

    def test_iso_with_t_and_offset_accepted(self):
        df = make_frame(timestamp=["2026-10-01T09:00:00+05:30", "2026-10-01T10:00:00+05:30",
                                   "2026-10-01T11:00:00+05:30"])
        self.assertTrue(validate_sensor_data(df).is_valid)

    def test_mixed_offsets_warn(self):
        df = make_frame(timestamp=["2026-10-01T09:00:00+05:30", "2026-10-01T10:00:00+00:00",
                                   "2026-10-01T11:00:00+05:30"])
        result = validate_sensor_data(df)
        self.assertTrue(result.is_valid, result.format_report())
        self.assertTrue(result.has_warning("mixed_timezones"))

    def test_datetime_dtype_accepted(self):
        df = make_frame(timestamp=pd.to_datetime(["2026-10-01 09:00", "2026-10-01 10:00", "2026-10-01 11:00"]))
        self.assertTrue(validate_sensor_data(df).is_valid)

    def test_repeated_timestamps_warn_not_error(self):
        df = make_frame(timestamp=["2026-10-01 09:00:00"] * 3)
        result = validate_sensor_data(df)
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warning("repeated_timestamp"))

    def test_same_sample_same_timestamp_warns_when_ids_repeat(self):
        df = make_frame(sample_id=["B1", "B1", "B2"],
                        timestamp=["2026-10-01 09:00", "2026-10-01 09:00", "2026-10-01 09:00"])
        result = validate_sensor_data(df, ValidationConfig(sample_id_unique=False))
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warning("duplicate_sample_timestamp"))

    def test_clock_problems_warn(self):
        df = make_frame(timestamp=["1970-01-01 00:00:05", "2099-01-01 00:00:00", "2026-10-01 11:00:00"])
        result = validate_sensor_data(df)
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warning("implausibly_old_timestamp"))
        self.assertTrue(result.has_warning("future_timestamp"))

    def test_parse_timestamps_keeps_length(self):
        parsed = parse_timestamps(pd.Series(["2026-10-01 09:00", "bad", None]))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(int(parsed.isna().sum()), 2)


class TestSensorValues(unittest.TestCase):
    def test_non_numeric_sensor_readings(self):
        df = make_frame(gas_raw=[300, "high", 320], humidity_percent=[60, "wet", 61])
        result = validate_sensor_data(df)
        codes = [(e.code, e.column) for e in result.errors]
        self.assertIn(("non_numeric_value", "gas_raw"), codes)
        self.assertIn(("non_numeric_value", "humidity_percent"), codes)
        self.assertEqual(result.invalid_rows, (1,))

    def test_numeric_strings_with_spaces_accepted(self):
        self.assertTrue(validate_sensor_data(make_frame(gas_raw=[" 300 ", "310", 320])).is_valid)

    def test_infinite_values_rejected(self):
        result = validate_sensor_data(make_frame(gas_raw=[300, float("inf"), 320]))
        self.assertTrue(result.has_error("non_finite_value"))

    def test_negative_gas_reading(self):
        result = validate_sensor_data(make_frame(gas_raw=[300, -1, 320]))
        self.assertTrue(result.has_error("negative_gas_reading"))
        self.assertEqual(result.invalid_rows, (1,))

    def test_zero_gas_reading_allowed(self):
        self.assertTrue(validate_sensor_data(make_frame(gas_raw=[0, 1, 2])).is_valid)

    def test_gas_max_only_when_configured(self):
        df = make_frame(gas_raw=[300, 5000, 320])
        self.assertTrue(validate_sensor_data(df).is_valid)
        result = validate_sensor_data(df, ValidationConfig(gas_raw_max=4095))
        self.assertTrue(result.has_error("gas_reading_above_max"))

    def test_humidity_below_zero(self):
        result = validate_sensor_data(make_frame(humidity_percent=[60, -0.1, 61]))
        self.assertTrue(result.has_error("humidity_out_of_range"))

    def test_humidity_above_hundred(self):
        result = validate_sensor_data(make_frame(humidity_percent=[60, 100.5, 61]))
        self.assertTrue(result.has_error("humidity_out_of_range"))

    def test_humidity_boundaries_valid(self):
        self.assertTrue(validate_sensor_data(make_frame(humidity_percent=[0, 100, 50])).is_valid)

    def test_impossible_temperature(self):
        result = validate_sensor_data(make_frame(temperature_c=[25, 300, -300]))
        self.assertTrue(result.has_error("temperature_out_of_range"))
        self.assertEqual(result.invalid_rows, (1, 2))

    def test_unusual_temperature_is_warning(self):
        result = validate_sensor_data(make_frame(temperature_c=[25, 60, -5]))
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warning("temperature_unusual"))


class TestIdentifiersAndDuplicates(unittest.TestCase):
    def test_duplicate_sample_ids(self):
        result = validate_sensor_data(make_frame(sample_id=["S1", "S1", "S2"]))
        self.assertTrue(result.has_error("duplicate_sample_id"))
        self.assertEqual(result.invalid_rows, (0, 1))

    def test_sample_ids_trimmed_before_comparison(self):
        result = validate_sensor_data(make_frame(sample_id=["S1", " S1", "S2"]))
        self.assertTrue(result.has_error("duplicate_sample_id"))

    def test_duplicate_ids_allowed_when_configured(self):
        df = make_frame(sample_id=["B1", "B1", "B1"])
        self.assertTrue(validate_sensor_data(df, ValidationConfig(sample_id_unique=False)).is_valid)

    def test_exact_duplicate_rows_warn(self):
        df = make_frame().drop(columns=["sample_id"])
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        result = validate_sensor_data(df)
        self.assertTrue(result.has_warning("duplicate_row"))
        self.assertEqual(result.rows_checked, 4)  # nothing removed


class TestLabels(unittest.TestCase):
    def test_no_label_column_still_valid_for_exploration(self):
        result = validate_sensor_data(make_frame().drop(columns=["spoilage_label"]))
        self.assertTrue(result.is_valid)
        self.assertTrue(any("not for supervised" in n for n in result.notes))

    def test_missing_labels_are_warnings(self):
        result = validate_sensor_data(make_frame(spoilage_label=["fresh", None, "ripe"]))
        self.assertTrue(result.is_valid)
        self.assertTrue(result.has_warning("missing_label"))

    def test_allowed_labels_enforced(self):
        config = ValidationConfig(allowed_labels=frozenset({"fresh", "ripe", "spoiled"}))
        result = validate_sensor_data(make_frame(spoilage_label=["Fresh", "rotten", "ripe"]), config)
        self.assertTrue(result.has_error("unknown_label"))
        self.assertEqual(result.invalid_rows, (1,))

    def test_inconsistent_label_spelling_warns(self):
        result = validate_sensor_data(make_frame(spoilage_label=["fresh", "Fresh", "ripe"]))
        self.assertTrue(result.has_warning("inconsistent_label_spelling"))

    def test_single_class_warns(self):
        result = validate_sensor_data(make_frame(spoilage_label=["fresh"] * 3))
        self.assertTrue(result.has_warning("single_label_class"))

    def test_labels_never_claimed_scientifically_valid(self):
        notes = " ".join(validate_sensor_data(make_frame()).notes)
        self.assertIn("not been verified", notes)


class TestOrigin(unittest.TestCase):
    def test_synthetic_markers(self):
        for value in ("synthetic", "Synthetic_Demo", " demo ", "synthetic-v2", "SIMULATED"):
            self.assertTrue(is_synthetic_source(value), value)
        for value in ("bench_rig_1", "", None, float("nan"), "lab"):
            self.assertFalse(is_synthetic_source(value), value)

    def test_mixed_origin_warns(self):
        result = validate_sensor_data(make_frame(source=["bench_rig_1", "synthetic_demo", "bench_rig_1"]))
        self.assertTrue(result.has_warning("mixed_origin"))

    def test_expected_real_rejects_synthetic_rows(self):
        df = make_frame(source=["bench_rig_1", "synthetic_demo", "bench_rig_1"])
        result = validate_sensor_data(df, expected_origin="real")
        self.assertTrue(result.has_error("synthetic_rows_in_real_data"))
        self.assertEqual(result.invalid_rows, (1,))

    def test_expected_synthetic_requires_marking(self):
        df = make_frame().drop(columns=["source"])
        self.assertTrue(validate_sensor_data(df, expected_origin="synthetic").has_error("synthetic_not_marked"))
        df2 = make_frame(source=["synthetic_demo", None, "synthetic_demo"])
        issues = check_origin_consistency(df2, "synthetic")
        self.assertEqual(issues[0].rows, (1,))

    def test_invalid_expected_origin_raises(self):
        with self.assertRaises(ValueError):
            validate_sensor_data(make_frame(), expected_origin="fake")


class TestResultObject(unittest.TestCase):
    def test_format_report_mentions_status(self):
        report = validate_sensor_data(make_frame(gas_raw=[1, -1, 2])).format_report()
        self.assertIn("INVALID", report)
        self.assertIn("negative", report.lower())

    def test_row_list_truncated_in_message(self):
        df = make_frame(gas_raw=[-1, -1, -1])
        msg = validate_sensor_data(df, ValidationConfig(max_rows_in_message=1)).error_messages()[0]
        self.assertIn("and 2 more", msg)

    def test_is_valid_reflects_errors(self):
        self.assertFalse(ValidationResult(rows_checked=1, errors=[object()]).is_valid)  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
