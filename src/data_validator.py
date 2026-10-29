"""Structural validation for Banana Spoilage AI sensor datasets.

This module checks that a pandas DataFrame is *structurally* fit to be accepted
by the system: the right columns exist, values have the right types, sensor
readings sit inside physically/technically plausible ranges, and identifiers
are not duplicated.

What this module does NOT do
----------------------------
Structural validity is not scientific validity. A dataset can pass every check
here and still contain spoilage labels that are wrong, subjective, or not
reproducible. The MQ-135 is a broad gas/vapour sensor, not a banana-specific
spoilage sensor, and this module makes no claim that gas readings indicate
spoilage. See ``SCIENTIFIC_DISCLAIMER``.

Invalid records are never removed or altered. Every problem is *reported*;
deciding what to do about it is left to the caller.

Expected DataFrame schema
-------------------------
=================  ========  ==================================================
Column             Required  Meaning / rule
=================  ========  ==================================================
timestamp          yes       Measurement time. ISO 8601 text by default
                             (e.g. ``2026-10-28 14:30:00``), or datetime dtype.
gas_raw            yes       Raw MQ-135 reading (e.g. ADC counts). Numeric,
                             finite, >= 0. No upper limit unless configured.
sample_id          no        Record identifier. Unique per row by default
                             (see ``ValidationConfig.sample_id_unique``).
temperature_c      no        Degrees Celsius. Numeric. Outside -40..80 is an
                             error; outside 0..50 is a warning (configurable).
humidity_percent   no        Relative humidity. Numeric, 0..100 inclusive.
source             no        Data origin, e.g. ``sensor_rig_1`` or
                             ``synthetic_demo``. Values in
                             ``SYNTHETIC_SOURCE_MARKERS`` (or starting with
                             "synthetic") mark a row as synthetic.
spoilage_label     no        Human-assigned label. Checked structurally only.
=================  ========  ==================================================

Columns not listed above are allowed and reported as a note.

Public interface
----------------
``validate_sensor_data(df, config=None, *, expected_origin=None) -> ValidationResult``
    Main entry point. Never raises for bad *data*; problems are returned.
``check_origin_consistency(df, expected_origin) -> list[ValidationIssue]``
    Real-vs-synthetic consistency check (also used by ``data_manager``).
``parse_timestamps(series, timestamp_format="ISO8601") -> pandas.Series``
    Parse a timestamp column the same way the validator does.
``missing_mask(series) -> pandas.Series``
    Boolean mask of missing values (NaN/None/NaT or blank text).
``is_synthetic_source(value) -> bool``
    Whether a ``source`` value marks a row as synthetic.
``ValidationConfig``, ``ValidationResult``, ``ValidationIssue``
    Configuration and result types (documented on each class).
"""

from __future__ import annotations

import warnings
from datetime import timedelta
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

__all__ = [
    "TIMESTAMP_COLUMN",
    "SAMPLE_ID_COLUMN",
    "GAS_COLUMN",
    "TEMPERATURE_COLUMN",
    "HUMIDITY_COLUMN",
    "SOURCE_COLUMN",
    "LABEL_COLUMN",
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "KNOWN_COLUMNS",
    "SENSOR_COLUMNS",
    "SYNTHETIC_SOURCE_MARKERS",
    "SCIENTIFIC_DISCLAIMER",
    "Origin",
    "ValidationConfig",
    "ValidationIssue",
    "ValidationResult",
    "validate_sensor_data",
    "check_origin_consistency",
    "parse_timestamps",
    "missing_mask",
    "is_synthetic_source",
]

# --------------------------------------------------------------------------- #
# Schema constants
# --------------------------------------------------------------------------- #
TIMESTAMP_COLUMN = "timestamp"
SAMPLE_ID_COLUMN = "sample_id"
GAS_COLUMN = "gas_raw"
TEMPERATURE_COLUMN = "temperature_c"
HUMIDITY_COLUMN = "humidity_percent"
SOURCE_COLUMN = "source"
LABEL_COLUMN = "spoilage_label"

#: Columns every dataset must contain.
REQUIRED_COLUMNS: tuple[str, ...] = (TIMESTAMP_COLUMN, GAS_COLUMN)
#: Columns that are checked when present but may be absent.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    SAMPLE_ID_COLUMN,
    TEMPERATURE_COLUMN,
    HUMIDITY_COLUMN,
    SOURCE_COLUMN,
    LABEL_COLUMN,
)
KNOWN_COLUMNS: tuple[str, ...] = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
#: Numeric sensor columns.
SENSOR_COLUMNS: tuple[str, ...] = (GAS_COLUMN, TEMPERATURE_COLUMN, HUMIDITY_COLUMN)

#: ``source`` values (case-insensitive, trimmed) that mark a row as synthetic.
#: Any value starting with "synthetic" is also treated as synthetic.
SYNTHETIC_SOURCE_MARKERS: frozenset[str] = frozenset(
    {"synthetic", "synthetic_demo", "demo", "simulated", "generated"}
)

HUMIDITY_RANGE: tuple[float, float] = (0.0, 100.0)

SCIENTIFIC_DISCLAIMER = (
    "Structural validation only: passing these checks does not mean spoilage "
    "labels are correct or that gas readings indicate banana spoilage. The "
    "MQ-135 responds to many gases and vapours and is not a validated "
    "banana-spoilage sensor."
)

Origin = Literal["real", "synthetic"]
Severity = Literal["error", "warning"]


def is_synthetic_source(value: Any) -> bool:
    """Return True if a ``source`` cell value marks its row as synthetic.

    Args:
        value: A single cell from the ``source`` column.

    Returns:
        True when the trimmed, lower-cased value is in
        ``SYNTHETIC_SOURCE_MARKERS`` or starts with ``"synthetic"``.
        Missing values return False (missing is *unspecified*, not real).
    """
    if not isinstance(value, str):
        if value is None or pd.isna(value):
            return False
        value = str(value)
    normalized = value.strip().lower()
    return normalized in SYNTHETIC_SOURCE_MARKERS or normalized.startswith("synthetic")


# --------------------------------------------------------------------------- #
# Configuration and result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationConfig:
    """Tunable validation rules. All fields have sensible defaults.

    Attributes:
        required_columns: Columns that must be present.
        timestamp_format: Format passed to ``pandas.to_datetime``. ``"ISO8601"``
            (default) accepts e.g. ``2026-10-28 14:30:00`` and
            ``2026-10-28T14:30:00+05:30``. Use a strftime pattern such as
            ``"%d/%m/%Y %H:%M"`` for other formats, or ``None`` to let pandas
            guess (lenient; may accept ambiguous dates).
        temperature_error_range: Inclusive (min, max) in Celsius. Values outside
            are errors. Default -40..80 matches the range of common hobby
            sensors (e.g. DHT22); values outside it almost certainly indicate a
            sensor or logging fault.
        temperature_warning_range: Inclusive (min, max). Values inside the error
            range but outside this one produce a warning only.
        gas_raw_max: Optional inclusive upper bound for ``gas_raw`` (e.g. 1023
            for a 10-bit ADC, 4095 for 12-bit). ``None`` disables the check,
            because the correct bound depends on hardware we have not seen.
        sample_id_unique: If True, each ``sample_id`` must appear once (one
            record per row). Set False if ``sample_id`` identifies a banana
            measured repeatedly; then repeated (sample_id, timestamp) pairs
            are warned about instead.
        allowed_labels: Optional set of permitted ``spoilage_label`` values
            (compared case-insensitively after trimming). ``None`` means any
            non-blank label is structurally acceptable.
        max_rows_in_message: How many row positions to list in a message.
    """

    required_columns: tuple[str, ...] = REQUIRED_COLUMNS
    timestamp_format: str | None = "ISO8601"
    temperature_error_range: tuple[float, float] = (-40.0, 80.0)
    temperature_warning_range: tuple[float, float] = (0.0, 50.0)
    gas_raw_max: float | None = None
    sample_id_unique: bool = True
    allowed_labels: frozenset[str] | None = None
    max_rows_in_message: int = 10


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found during validation.

    Attributes:
        severity: ``"error"`` (dataset is rejected) or ``"warning"`` (accepted,
            but worth a look).
        code: Stable machine-readable identifier, e.g. ``"humidity_out_of_range"``.
        message: Human-readable explanation suitable for showing to a user.
        column: Column concerned, if any.
        rows: 0-based row *positions* affected (not index labels). Empty for
            dataset-level issues.
    """

    severity: Severity
    code: str
    message: str
    column: str | None = None
    rows: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "column": self.column,
            "rows": list(self.rows),
        }


@dataclass
class ValidationResult:
    """Outcome of ``validate_sensor_data``.

    Attributes:
        rows_checked: Number of rows examined (0 if input was not a DataFrame).
        errors: Issues that make the dataset invalid.
        warnings: Issues that do not block acceptance.
        notes: Informational messages (e.g. label column absent, disclaimer).
        invalid_rows: Sorted 0-based positions of rows with at least one
            row-level error, or ``None`` if this could not be determined (input
            not a DataFrame, or duplicate column names). Dataset-level errors
            such as a missing required column are not attributed to rows.

    Properties:
        is_valid: True when there are no errors.
        invalid_row_count: ``len(invalid_rows)`` or ``None``.
    """

    rows_checked: int
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    invalid_rows: tuple[int, ...] | None = ()

    @property
    def is_valid(self) -> bool:
        """True when no errors were found."""
        return not self.errors

    @property
    def invalid_row_count(self) -> int | None:
        """Number of rows with row-level errors, or None if undeterminable."""
        return None if self.invalid_rows is None else len(self.invalid_rows)

    def error_messages(self) -> list[str]:
        """Return error messages as plain strings."""
        return [issue.message for issue in self.errors]

    def warning_messages(self) -> list[str]:
        """Return warning messages as plain strings."""
        return [issue.message for issue in self.warnings]

    def has_error(self, code: str) -> bool:
        """Return True if an error with the given ``code`` was reported."""
        return any(issue.code == code for issue in self.errors)

    def has_warning(self, code: str) -> bool:
        """Return True if a warning with the given ``code`` was reported."""
        return any(issue.code == code for issue in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary of the full result."""
        return {
            "is_valid": self.is_valid,
            "rows_checked": self.rows_checked,
            "invalid_row_count": self.invalid_row_count,
            "invalid_rows": None if self.invalid_rows is None else list(self.invalid_rows),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "notes": list(self.notes),
        }

    def format_report(self) -> str:
        """Return a short multi-line, human-readable report."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [f"Validation: {status} ({self.rows_checked} rows checked)"]
        if self.invalid_row_count is not None:
            lines.append(f"Rows with errors: {self.invalid_row_count}")
        for title, items in (("Errors", self.error_messages()), ("Warnings", self.warning_messages())):
            if items:
                lines.append(f"{title}:")
                lines.extend(f"  - {msg}" for msg in items)
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def missing_mask(series: pd.Series) -> pd.Series:
    """Return a boolean Series that is True where a value is missing.

    Missing means NaN/None/NaT *or* a string that is empty after trimming.

    Args:
        series: Any pandas Series.

    Returns:
        Boolean Series aligned with ``series``.
    """
    blank = series.map(lambda v: isinstance(v, str) and not v.strip())
    return (series.isna() | blank.astype(bool)).astype(bool)


def _positions(mask: pd.Series) -> tuple[int, ...]:
    """Return 0-based positions where a boolean mask is True."""
    return tuple(int(i) for i in mask.to_numpy(dtype=bool).nonzero()[0])


def _describe_rows(rows: tuple[int, ...], limit: int) -> str:
    """Format row positions for messages, e.g. ``"rows 0, 4 (and 2 more)"``."""
    shown = ", ".join(str(r) for r in rows[:limit])
    extra = len(rows) - limit
    label = "row" if len(rows) == 1 else "rows"
    return f"{label} {shown}" + (f" (and {extra} more)" if extra > 0 else "")


def _examples(series: pd.Series, mask: pd.Series, limit: int = 3) -> str:
    """Return a few example values (repr, truncated) where ``mask`` is True."""
    values = series[mask.to_numpy(dtype=bool)].head(limit).tolist()
    return ", ".join(repr(v)[:30] for v in values)


def _to_numeric(series: pd.Series) -> pd.Series:
    """Coerce to float; unparseable values become NaN. Strings are trimmed."""
    cleaned = series.map(lambda v: v.strip() if isinstance(v, str) else v)
    return pd.to_numeric(cleaned, errors="coerce").astype(float)


def _parse_timestamps_with_flag(
    series: pd.Series, timestamp_format: str | None
) -> tuple[pd.Series, bool]:
    """Parse timestamps; return (parsed, converted_to_utc).

    ``converted_to_utc`` is True when values had mixed UTC offsets (or mixed
    naive/aware values) and had to be normalised to UTC to compare them.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, False

    text = series.map(
        lambda v: None if (not isinstance(v, str) and pd.isna(v)) else str(v).strip() or None
    )
    kwargs: dict[str, Any] = {"errors": "coerce"}
    if timestamp_format:
        kwargs["format"] = timestamp_format

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(text, **kwargs)
            if pd.api.types.is_datetime64_any_dtype(parsed):
                return parsed, False
        except (ValueError, TypeError):
            pass
        # Mixed offsets / naive+aware mix: compare everything in UTC.
        return pd.to_datetime(text, utc=True, **kwargs), True


def parse_timestamps(series: pd.Series, timestamp_format: str | None = "ISO8601") -> pd.Series:
    """Parse a timestamp column exactly as the validator does.

    Args:
        series: Column of timestamp text (or an already-datetime Series).
        timestamp_format: See ``ValidationConfig.timestamp_format``.

    Returns:
        Datetime Series aligned with the input. Missing or unparseable values
        become ``NaT``; nothing is dropped. Mixed time-zone offsets are
        converted to UTC.
    """
    parsed, _ = _parse_timestamps_with_flag(series, timestamp_format)
    return parsed


class _Collector:
    """Accumulates issues, notes and invalid row positions during validation."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.errors: list[ValidationIssue] = []
        self.warnings: list[ValidationIssue] = []
        self.notes: list[str] = []
        self.invalid_rows: set[int] = set()

    def add(
        self,
        severity: Severity,
        code: str,
        message: str,
        column: str | None = None,
        rows: tuple[int, ...] = (),
    ) -> None:
        if rows:
            message = f"{message} ({_describe_rows(rows, self.limit)})."
        issue = ValidationIssue(severity, code, message, column, rows)
        if severity == "error":
            self.errors.append(issue)
            self.invalid_rows.update(rows)
        else:
            self.warnings.append(issue)

    def result(self, rows_checked: int, rows_determinable: bool = True) -> ValidationResult:
        return ValidationResult(
            rows_checked=rows_checked,
            errors=self.errors,
            warnings=self.warnings,
            notes=self.notes,
            invalid_rows=tuple(sorted(self.invalid_rows)) if rows_determinable else None,
        )


# --------------------------------------------------------------------------- #
# Individual checks (each one small and independent)
# --------------------------------------------------------------------------- #
def _check_column_structure(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> bool:
    """Check duplicate/missing/unknown columns. Return False if checking must stop."""
    duplicated = sorted({str(c) for c in df.columns[df.columns.duplicated()]})
    if duplicated:
        out.add("error", "duplicate_columns",
                f"Duplicate column names: {', '.join(duplicated)}. Each column must appear once.")
        return False

    present = {str(c) for c in df.columns}
    lookup = {str(c).strip().lower(): str(c) for c in df.columns}
    for column in config.required_columns:
        if column in present:
            continue
        near = lookup.get(column.lower())
        hint = f" A column named {near!r} exists; check spelling, case or spaces." if near else ""
        out.add("error", "missing_required_column",
                f"Required column {column!r} is missing.{hint}", column=column)

    unknown = sorted(present - set(KNOWN_COLUMNS) - set(config.required_columns))
    if unknown:
        out.notes.append(f"Extra columns not checked by the validator: {', '.join(unknown)}.")
    return True


def _check_missing_values(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """Required columns: missing values are errors. Other known columns: warnings."""
    for column in df.columns:
        column = str(column)
        if column not in KNOWN_COLUMNS and column not in config.required_columns:
            continue
        rows = _positions(missing_mask(df[column]))
        if not rows:
            continue
        if column in config.required_columns:
            out.add("error", "missing_required_value",
                    f"Column {column!r} has {len(rows)} missing value(s)", column, rows)
        elif column == LABEL_COLUMN:
            out.add("warning", "missing_label",
                    f"{len(rows)} row(s) have no spoilage label and cannot be used for "
                    "supervised training", column, rows)
        else:
            out.add("warning", "missing_optional_value",
                    f"Optional column {column!r} has {len(rows)} missing value(s)", column, rows)


def _check_timestamps(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """Check timestamp format, implausible dates and repeated timestamps."""
    if TIMESTAMP_COLUMN not in df.columns:
        return
    raw = df[TIMESTAMP_COLUMN]
    parsed, converted = _parse_timestamps_with_flag(raw, config.timestamp_format)
    present = ~missing_mask(raw)

    bad = present & parsed.isna()
    if bad.any():
        fmt = config.timestamp_format or "auto-detected"
        out.add("error", "invalid_timestamp",
                f"{int(bad.sum())} timestamp(s) could not be parsed with format {fmt!r}; "
                f"examples: {_examples(raw, bad)}", TIMESTAMP_COLUMN, _positions(bad))

    if converted:
        out.add("warning", "mixed_timezones",
                "Timestamps mix time-zone offsets (or mix zoned and un-zoned values); "
                "they were compared in UTC.", TIMESTAMP_COLUMN)

    valid = parsed.notna()
    if not valid.any():
        return
    tz = getattr(parsed.dt, "tz", None)
    now = pd.Timestamp.now(tz=tz)
    future = valid & (parsed > now + timedelta(days=1))
    if future.any():
        out.add("warning", "future_timestamp",
                "Timestamp(s) lie in the future; the logger clock may be wrong",
                TIMESTAMP_COLUMN, _positions(future))
    old = valid & (parsed < pd.Timestamp("2000-01-01", tz=tz))
    if old.any():
        out.add("warning", "implausibly_old_timestamp",
                "Timestamp(s) are before 2000; the logger clock may have been reset",
                TIMESTAMP_COLUMN, _positions(old))

    _check_repeated_timestamps(df, parsed, valid, config, out)


def _check_repeated_timestamps(
    df: pd.DataFrame, parsed: pd.Series, valid: pd.Series,
    config: ValidationConfig, out: _Collector,
) -> None:
    """Warn about repeated timestamps without treating them as invalid."""
    repeated = valid & parsed.duplicated(keep=False)
    if not repeated.any():
        return
    if SAMPLE_ID_COLUMN in df.columns and not config.sample_id_unique:
        keys = pd.DataFrame({"id": df[SAMPLE_ID_COLUMN].astype(str).str.strip(), "t": parsed})
        same_sample = valid & keys.duplicated(keep=False)
        if same_sample.any():
            out.add("warning", "duplicate_sample_timestamp",
                    "The same sample_id has more than one reading at the same timestamp; "
                    "this may be double logging", TIMESTAMP_COLUMN, _positions(same_sample))
        return
    out.add("warning", "repeated_timestamp",
            f"{int(repeated.sum())} row(s) share a timestamp with another row. This is "
            "normal if several samples were measured at once, but may also indicate "
            "repeated logging", TIMESTAMP_COLUMN, _positions(repeated))


def _numeric_values(df: pd.DataFrame, column: str, out: _Collector) -> pd.Series | None:
    """Return numeric values for a sensor column, reporting non-numeric entries.

    Returns None when the column is absent. Non-numeric and non-finite entries
    are reported as errors and appear as NaN in the returned Series.
    """
    if column not in df.columns:
        return None
    raw = df[column]
    numeric = _to_numeric(raw)
    present = ~missing_mask(raw)

    non_numeric = present & numeric.isna()
    if non_numeric.any():
        out.add("error", "non_numeric_value",
                f"Column {column!r} has {int(non_numeric.sum())} non-numeric value(s); "
                f"examples: {_examples(raw, non_numeric)}", column, _positions(non_numeric))

    infinite = numeric.isin([float("inf"), float("-inf")])
    if infinite.any():
        out.add("error", "non_finite_value",
                f"Column {column!r} contains infinite value(s)", column, _positions(infinite))
    return numeric.where(~infinite)


def _check_gas(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """gas_raw must be numeric, finite and non-negative (and <= gas_raw_max if set)."""
    values = _numeric_values(df, GAS_COLUMN, out)
    if values is None:
        return
    negative = values < 0
    if negative.any():
        out.add("error", "negative_gas_reading",
                "Raw gas readings cannot be negative", GAS_COLUMN, _positions(negative))
    if config.gas_raw_max is not None:
        too_high = values > config.gas_raw_max
        if too_high.any():
            out.add("error", "gas_reading_above_max",
                    f"Raw gas readings exceed the configured maximum {config.gas_raw_max:g}",
                    GAS_COLUMN, _positions(too_high))


def _check_temperature(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """Temperatures outside the error range are errors; outside the warning range, warnings."""
    values = _numeric_values(df, TEMPERATURE_COLUMN, out)
    if values is None:
        return
    low, high = config.temperature_error_range
    impossible = (values < low) | (values > high)
    if impossible.any():
        out.add("error", "temperature_out_of_range",
                f"Temperatures outside {low:g}..{high:g} °C are treated as sensor or "
                "logging faults", TEMPERATURE_COLUMN, _positions(impossible))
    w_low, w_high = config.temperature_warning_range
    unusual = ~impossible & ((values < w_low) | (values > w_high))
    if unusual.any():
        out.add("warning", "temperature_unusual",
                f"Temperatures outside {w_low:g}..{w_high:g} °C are unusual for banana storage",
                TEMPERATURE_COLUMN, _positions(unusual))


def _check_humidity(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """Relative humidity must lie within 0..100 % inclusive."""
    values = _numeric_values(df, HUMIDITY_COLUMN, out)
    if values is None:
        return
    low, high = HUMIDITY_RANGE
    outside = (values < low) | (values > high)
    if outside.any():
        out.add("error", "humidity_out_of_range",
                "Relative humidity must be between 0 and 100 %", HUMIDITY_COLUMN,
                _positions(outside))


def _check_sample_ids(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """Report duplicate sample IDs (all occurrences) when IDs must be unique."""
    if SAMPLE_ID_COLUMN not in df.columns or not config.sample_id_unique:
        return
    raw = df[SAMPLE_ID_COLUMN]
    present = ~missing_mask(raw)
    ids = raw.map(lambda v: str(v).strip())
    duplicated = present & ids.where(present).duplicated(keep=False)
    if duplicated.any():
        distinct = ids[duplicated.to_numpy()].nunique()
        out.add("error", "duplicate_sample_id",
                f"{distinct} sample_id value(s) appear more than once; all occurrences are "
                "listed because it is not possible to tell which one is correct",
                SAMPLE_ID_COLUMN, _positions(duplicated))


def _check_duplicate_rows(df: pd.DataFrame, out: _Collector) -> None:
    """Warn about rows that exactly repeat an earlier row."""
    repeats = df.astype(str).duplicated(keep="first")
    if repeats.any():
        out.add("warning", "duplicate_row",
                "Row(s) exactly repeat an earlier row; possible double logging",
                rows=_positions(repeats))


def _normalise_label(value: Any) -> str:
    return str(value).strip().casefold()


def _check_labels(df: pd.DataFrame, config: ValidationConfig, out: _Collector) -> None:
    """Structural label checks. Label *correctness* is never assessed."""
    if LABEL_COLUMN not in df.columns:
        out.notes.append(
            "No 'spoilage_label' column: the dataset can be used for sensor-data "
            "exploration but not for supervised model training."
        )
        return
    raw = df[LABEL_COLUMN]
    present = ~missing_mask(raw)
    labels = raw[present.to_numpy()]
    normalised = labels.map(_normalise_label)

    if config.allowed_labels is not None:
        allowed = {_normalise_label(v) for v in config.allowed_labels}
        unknown = present & raw.map(_normalise_label).map(lambda v: v not in allowed)
        if unknown.any():
            out.add("error", "unknown_label",
                    f"Labels not in the allowed set {sorted(config.allowed_labels)}; "
                    f"examples: {_examples(raw, unknown)}", LABEL_COLUMN, _positions(unknown))
    else:
        variants = labels.map(lambda v: str(v).strip()).groupby(normalised.to_numpy()).nunique()
        if (variants > 1).any() or (labels.map(str) != labels.map(lambda v: str(v).strip())).any():
            out.add("warning", "inconsistent_label_spelling",
                    "Some labels differ only in case or surrounding spaces "
                    "(e.g. 'Fresh' vs 'fresh '); consider standardising them.", LABEL_COLUMN)

    classes = sorted(normalised.unique())
    if len(classes) == 1:
        out.add("warning", "single_label_class",
                f"Only one label class is present ({classes[0]!r}); a classifier cannot be "
                "trained on a single class.", LABEL_COLUMN)
    if classes:
        out.notes.append(f"Label classes found: {', '.join(classes)}.")
    out.notes.append(
        "Spoilage labels were checked for structure only; whether they reflect "
        "actual banana spoilage has not been verified."
    )


def _check_source(df: pd.DataFrame, out: _Collector) -> None:
    """Report the real/synthetic make-up of the dataset."""
    if SOURCE_COLUMN not in df.columns:
        out.notes.append(
            "No 'source' column: whether rows are real or synthetic cannot be "
            "determined from the data itself."
        )
        return
    raw = df[SOURCE_COLUMN]
    synthetic = raw.map(is_synthetic_source).astype(bool)
    declared = ~missing_mask(raw) & ~synthetic
    if synthetic.any() and declared.any():
        out.add("warning", "mixed_origin",
                f"Dataset mixes {int(synthetic.sum())} synthetic row(s) with "
                f"{int(declared.sum())} non-synthetic row(s). Synthetic rows must not be "
                "used as real observations.", SOURCE_COLUMN)
    elif synthetic.all():
        out.notes.append("All rows are marked as synthetic demo data.")


def check_origin_consistency(df: pd.DataFrame, expected_origin: Origin | None) -> list[ValidationIssue]:
    """Check that a dataset's ``source`` column matches its declared origin.

    Rules:
        * ``"synthetic"``: a ``source`` column must exist and *every* row must
          carry a synthetic marker (see ``is_synthetic_source``), so synthetic
          records stay identifiable after they leave this program.
        * ``"real"``: no row may carry a synthetic marker. A missing ``source``
          column is allowed (origin is then simply undeclared).
        * ``None``: no check.

    Args:
        df: Dataset to check.
        expected_origin: ``"real"``, ``"synthetic"`` or ``None``.

    Returns:
        List of error-severity issues (empty when consistent).

    Raises:
        ValueError: If ``expected_origin`` is not one of the allowed values.
    """
    if expected_origin is None:
        return []
    expected = str(expected_origin)
    if expected not in ("real", "synthetic"):
        raise ValueError(f"expected_origin must be 'real', 'synthetic' or None, got {expected_origin!r}")

    if SOURCE_COLUMN not in df.columns:
        if expected == "synthetic":
            return [ValidationIssue(
                "error", "synthetic_not_marked",
                "Synthetic data must have a 'source' column marking every row as synthetic "
                "(use data_manager.mark_as_synthetic).", SOURCE_COLUMN)]
        return []

    synthetic = df[SOURCE_COLUMN].map(is_synthetic_source).astype(bool)
    if expected == "synthetic":
        rows = _positions(~synthetic)
        if rows:
            return [ValidationIssue(
                "error", "synthetic_not_marked",
                f"{len(rows)} row(s) of a synthetic dataset are not marked as synthetic in "
                f"'source' ({_describe_rows(rows, 10)}).", SOURCE_COLUMN, rows)]
        return []

    rows = _positions(synthetic)
    if rows:
        return [ValidationIssue(
            "error", "synthetic_rows_in_real_data",
            f"{len(rows)} row(s) are marked as synthetic but the dataset is declared real "
            f"({_describe_rows(rows, 10)}). Synthetic records must not be mixed into real data.",
            SOURCE_COLUMN, rows)]
    return []


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def validate_sensor_data(
    df: Any,
    config: ValidationConfig | None = None,
    *,
    expected_origin: Origin | None = None,
) -> ValidationResult:
    """Validate a banana-monitoring dataset without modifying it.

    Args:
        df: The dataset. Anything other than a ``pandas.DataFrame`` produces an
            invalid result rather than an exception.
        config: Validation rules; defaults to ``ValidationConfig()``.
        expected_origin: Optional ``"real"`` or ``"synthetic"``; if given, the
            ``source`` column is checked for consistency
            (see ``check_origin_consistency``).

    Returns:
        A ``ValidationResult``. ``result.is_valid`` is True when no errors were
        found. The input DataFrame is never changed and no rows are removed.

    Raises:
        ValueError: Only for an invalid ``expected_origin`` argument
            (a programming error, not a data problem).
    """
    config = config or ValidationConfig()
    out = _Collector(config.max_rows_in_message)

    if not isinstance(df, pd.DataFrame):
        out.add("error", "not_a_dataframe",
                f"Expected a pandas DataFrame, got {type(df).__name__}.")
        return out.result(rows_checked=0, rows_determinable=False)

    data = df.reset_index(drop=True)  # positions are reported, so index is irrelevant
    if not _check_column_structure(data, config, out):
        return out.result(rows_checked=len(data), rows_determinable=False)

    if len(data.columns) == 0 or len(data) == 0:
        out.add("error", "empty_dataset", "The dataset is empty (no rows or no columns).")
        return out.result(rows_checked=0)

    _check_missing_values(data, config, out)
    _check_timestamps(data, config, out)
    _check_gas(data, config, out)
    _check_temperature(data, config, out)
    _check_humidity(data, config, out)
    _check_sample_ids(data, config, out)
    _check_duplicate_rows(data, out)
    _check_labels(data, config, out)
    _check_source(data, out)
    for issue in check_origin_consistency(data, expected_origin):
        out.errors.append(issue)
        out.invalid_rows.update(issue.rows)

    out.notes.append(SCIENTIFIC_DISCLAIMER)
    return out.result(rows_checked=len(data))
