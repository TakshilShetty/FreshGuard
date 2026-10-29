"""Loading, saving, summarising and separating Banana Spoilage AI datasets.

All functions are UI-independent (no Streamlit) and report problems through
result objects instead of raising for bad files or bad data. Exceptions are
raised only for programming errors (wrong argument types/values), and each
function documents which ones.

Real vs synthetic data
----------------------
* **Real** data: measurements from actual sensors or genuine documented sources.
* **Synthetic** data: artificially generated values used to demonstrate the UI.

Rules enforced here:

1. Every save must declare ``origin="real"`` or ``origin="synthetic"``; there is
   no default.
2. Synthetic datasets must mark *every* row as synthetic in a ``source`` column
   (``mark_as_synthetic`` does this), so records stay identifiable anywhere.
3. A dataset saved as real may not contain any synthetic-marked row. This check
   cannot be bypassed with ``allow_invalid``.
4. Synthetic data is never written under a ``data/raw`` folder unless the caller
   explicitly passes ``allow_synthetic_in_raw=True``.
5. Nothing here merges datasets or relabels real rows automatically.
6. Rows with no ``source`` value are reported as *unspecified*, never assumed
   to be real.

Public interface
----------------
``load_dataset(path, ...) -> LoadResult``
``save_dataset(df, destination, *, origin, ...) -> SaveResult``
``summarize_dataset(df, ...) -> DatasetSummary``
``classify_row_origins(df) -> pandas.Series``
``separate_synthetic_rows(df) -> tuple[DataFrame, DataFrame]``
``mark_as_synthetic(df, source_label="synthetic_demo") -> DataFrame``
``assess_training_readiness(df, ...) -> TrainingReadiness``
``prepare_for_analysis(df, validation, ...) -> DataFrame``
"""

from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd

try:  # works both as ``src.data_manager`` and as a top-level ``data_manager``
    from .data_validator import (
        GAS_COLUMN, HUMIDITY_COLUMN, LABEL_COLUMN, SCIENTIFIC_DISCLAIMER,
        SENSOR_COLUMNS, SOURCE_COLUMN, TEMPERATURE_COLUMN, TIMESTAMP_COLUMN,
        ValidationConfig, ValidationResult, check_origin_consistency, is_synthetic_source,
        missing_mask, parse_timestamps, validate_sensor_data,
    )
except ImportError:  # pragma: no cover - fallback for flat imports
    from data_validator import (  # type: ignore[no-redef]
        GAS_COLUMN, HUMIDITY_COLUMN, LABEL_COLUMN, SCIENTIFIC_DISCLAIMER,
        SENSOR_COLUMNS, SOURCE_COLUMN, TEMPERATURE_COLUMN, TIMESTAMP_COLUMN,
        ValidationConfig, ValidationResult, check_origin_consistency, is_synthetic_source,
        missing_mask, parse_timestamps, validate_sensor_data,
    )

__all__ = [
    "DataOrigin",
    "DEFAULT_SYNTHETIC_SOURCE",
    "LoadResult",
    "SaveResult",
    "DatasetSummary",
    "TrainingReadiness",
    "load_dataset",
    "save_dataset",
    "summarize_dataset",
    "classify_row_origins",
    "separate_synthetic_rows",
    "mark_as_synthetic",
    "assess_training_readiness",
    "prepare_for_analysis",
]


class DataOrigin(StrEnum):
    """Declared origin of a dataset. Plain strings ``"real"``/``"synthetic"`` also work."""

    REAL = "real"
    SYNTHETIC = "synthetic"


#: ``source`` value written by ``mark_as_synthetic``.
DEFAULT_SYNTHETIC_SOURCE = "synthetic_demo"

ROW_ORIGIN_SYNTHETIC = "synthetic"
ROW_ORIGIN_DECLARED = "declared"
ROW_ORIGIN_UNSPECIFIED = "unspecified"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class LoadResult:
    """Outcome of ``load_dataset``.

    Attributes:
        path: The path that was requested.
        data: The DataFrame exactly as read (no rows dropped, no values changed,
            pandas' default type inference), or ``None`` if the file could not
            be read.
        validation: Validation result, or ``None`` if not validated (file
            unreadable or ``validate=False``).
        errors: File-level problems (missing file, malformed CSV, ...).
        warnings: File-level warnings (unexpected extension, short rows, ...).

    Properties:
        loaded: True if the file was read into a DataFrame.
        is_valid: True if loaded and validation (when run) found no errors.
    """

    path: Path
    data: pd.DataFrame | None = None
    validation: ValidationResult | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        """True if the file was read successfully."""
        return self.data is not None and not self.errors

    @property
    def is_valid(self) -> bool:
        """True if loaded and the dataset passed validation (or was not validated)."""
        return self.loaded and (self.validation is None or self.validation.is_valid)

    def all_errors(self) -> list[str]:
        """File-level errors followed by validation error messages."""
        extra = self.validation.error_messages() if self.validation else []
        return [*self.errors, *extra]

    def all_warnings(self) -> list[str]:
        """File-level warnings followed by validation warning messages."""
        extra = self.validation.warning_messages() if self.validation else []
        return [*self.warnings, *extra]


@dataclass
class SaveResult:
    """Outcome of ``save_dataset``.

    Attributes:
        path: Destination that was requested.
        saved: True only if the file was written.
        errors: Reasons the file was not written.
        warnings: Non-blocking concerns.
        validation: Validation result, if validation ran.
    """

    path: Path
    saved: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation: ValidationResult | None = None


@dataclass
class DatasetSummary:
    """Descriptive summary of a dataset (see ``summarize_dataset``).

    Attributes:
        row_count, column_count, columns: Basic shape.
        missing_required_columns: Required columns that are absent.
        missing_values: Missing-value count for every column.
        sensor_statistics: For each sensor column present, a dict with
            ``count`` (finite numeric values), ``non_numeric_count``, ``mean``,
            ``std``, ``min``, ``median``, ``max`` (``None`` when count is 0).
        timestamp_range: ``{"start", "end", "valid_count", "invalid_count"}``
            with ISO-8601 strings, or ``None`` if there is no timestamp column.
            ``start``/``end`` are ``None`` when no timestamp parses.
        source_counts: Count per ``source`` value, or ``None`` if no column.
        origin_counts: Rows per origin class: ``synthetic``, ``declared``
            (non-synthetic source given) and ``unspecified`` (no source).
        has_labels: Whether a ``spoilage_label`` column exists.
        label_counts: Count per label value, or ``None`` if no label column.
    """

    row_count: int
    column_count: int
    columns: list[str]
    missing_required_columns: list[str]
    missing_values: dict[str, int]
    sensor_statistics: dict[str, dict[str, float | int | None]]
    timestamp_range: dict[str, Any] | None
    source_counts: dict[str, int] | None
    origin_counts: dict[str, int]
    has_labels: bool
    label_counts: dict[str, int] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return dict(self.__dict__)


@dataclass
class TrainingReadiness:
    """Whether a dataset is structurally usable for exploration / training.

    Attributes:
        suitable_for_exploration: Dataset passed structural validation.
        suitable_for_supervised_training: Also has >= 2 label classes, at least
            one labelled row and no synthetic rows.
        blocking_reasons: Why supervised training is not suitable.
        cautions: Non-blocking concerns (e.g. undeclared origin).
        labeled_row_count, synthetic_row_count, unspecified_origin_row_count:
            Supporting counts.
        label_counts: Count per label value among labelled rows.
        disclaimer: Reminder that this is not scientific validation.
    """

    suitable_for_exploration: bool
    suitable_for_supervised_training: bool
    blocking_reasons: list[str]
    cautions: list[str]
    labeled_row_count: int
    synthetic_row_count: int
    unspecified_origin_row_count: int
    label_counts: dict[str, int]
    disclaimer: str = SCIENTIFIC_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        return dict(self.__dict__)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _inspect_csv_structure(path: Path, encoding: str) -> tuple[list[str], list[str]]:
    """Scan a CSV with the ``csv`` module; return (errors, warnings).

    Detects duplicate header names, rows with more fields than the header
    (error) and rows with fewer fields (warning: pandas fills them with NaN,
    which validation then reports). Line numbers are 1-based file lines.
    """
    errors: list[str] = []
    warns: list[str] = []
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return ["The file has no header row."], warns
        names = [h.strip() for h in header]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            errors.append(f"Duplicate column names in header: {', '.join(dupes)}.")
        too_many: list[int] = []
        too_few: list[int] = []
        for row in reader:
            if not row:  # blank line; pandas skips these
                continue
            if len(row) > len(header):
                too_many.append(reader.line_num)
            elif len(row) < len(header):
                too_few.append(reader.line_num)
    if too_many:
        errors.append(
            f"{len(too_many)} line(s) have more fields than the header "
            f"(lines {', '.join(map(str, too_many[:10]))}{' ...' if len(too_many) > 10 else ''})."
        )
    if too_few:
        warns.append(
            f"{len(too_few)} line(s) have fewer fields than the header; the missing "
            f"fields are treated as missing values (lines "
            f"{', '.join(map(str, too_few[:10]))}{' ...' if len(too_few) > 10 else ''})."
        )
    return errors, warns


def load_dataset(
    path: str | os.PathLike[str],
    *,
    validate: bool = True,
    config: ValidationConfig | None = None,
    expected_origin: DataOrigin | str | None = None,
    encoding: str = "utf-8-sig",
) -> LoadResult:
    """Load a CSV file into a DataFrame and (by default) validate it.

    Never raises for file or data problems: check ``result.loaded`` and
    ``result.is_valid``. The DataFrame is returned even when validation fails,
    so the UI can show the problem rows; nothing is removed or altered.

    Args:
        path: CSV file path.
        validate: Run ``validate_sensor_data`` on the loaded data.
        config: Validation rules (defaults to ``ValidationConfig()``).
        expected_origin: Optional ``"real"``/``"synthetic"`` origin check.
        encoding: File encoding. The default ``"utf-8-sig"`` also handles the
            byte-order mark that Excel adds to UTF-8 CSVs.

    Returns:
        ``LoadResult``.

    Raises:
        ValueError: Only for an invalid ``expected_origin`` value.
    """
    file_path = Path(path).expanduser()
    result = LoadResult(path=file_path)
    origin = _normalise_origin(expected_origin) if expected_origin is not None else None

    if not file_path.exists():
        result.errors.append(f"File not found: {file_path}")
        return result
    if not file_path.is_file():
        result.errors.append(f"Not a file: {file_path}")
        return result
    if file_path.suffix.lower() != ".csv":
        result.warnings.append(f"File does not have a .csv extension: {file_path.name}")

    try:
        if file_path.stat().st_size == 0:
            result.errors.append(f"The file is empty: {file_path.name}")
            return result
        struct_errors, struct_warnings = _inspect_csv_structure(file_path, encoding)
        result.warnings.extend(struct_warnings)
        if struct_errors:
            result.errors.extend(f"Malformed CSV: {msg}" for msg in struct_errors)
            return result
        data = pd.read_csv(file_path, encoding=encoding, index_col=False)
    except UnicodeDecodeError:
        result.errors.append(
            f"Could not decode {file_path.name} as {encoding}; is it a text CSV file?"
        )
        return result
    except (pd.errors.ParserError, pd.errors.EmptyDataError, csv.Error, ValueError) as exc:
        result.errors.append(f"Malformed CSV ({file_path.name}): {exc}")
        return result
    except OSError as exc:
        result.errors.append(f"Could not read {file_path}: {exc}")
        return result

    result.data = data
    if validate:
        result.validation = validate_sensor_data(data, config, expected_origin=origin)
    return result


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def _normalise_origin(origin: DataOrigin | str) -> str:
    """Return ``"real"`` or ``"synthetic"``; raise ValueError otherwise."""
    try:
        return DataOrigin(str(origin).strip().lower()).value
    except ValueError:
        raise ValueError(f"origin must be 'real' or 'synthetic', got {origin!r}") from None


def _is_under_data_subdir(path: Path, subdir: str) -> bool:
    """True if ``path`` lies inside a ``data/<subdir>`` folder (case-insensitive)."""
    parts = [p.lower() for p in path.resolve().parts]
    return any(parts[i] == "data" and parts[i + 1] == subdir for i in range(len(parts) - 1))


def _write_csv(df: pd.DataFrame, destination: Path, overwrite: bool) -> None:
    """Write CSV without overwriting (exclusive create) or atomically replace."""
    if not overwrite:
        with destination.open("x", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
        return
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            df.to_csv(handle, index=False)
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _pre_save_checks(
    df: pd.DataFrame, destination: Path, origin: str, allow_synthetic_in_raw: bool,
    result: SaveResult,
) -> None:
    """Origin and destination checks that ``allow_invalid`` can never bypass."""
    if destination.suffix.lower() != ".csv":
        result.errors.append(f"Destination must be a .csv file: {destination.name}")
    if origin == DataOrigin.SYNTHETIC:
        if _is_under_data_subdir(destination, "raw") and not allow_synthetic_in_raw:
            result.errors.append(
                "Refusing to save synthetic demo data inside data/raw/. Save it under "
                "data/demo/ instead (or pass allow_synthetic_in_raw=True deliberately)."
            )
        name = destination.name.lower()
        if "demo" not in name and "synthetic" not in name:
            result.warnings.append(
                "Synthetic dataset file name does not contain 'demo' or 'synthetic'; "
                "consider naming it so it is obviously synthetic."
            )
    elif _is_under_data_subdir(destination, "demo"):
        result.warnings.append("Saving real data inside data/demo/, which is meant for synthetic data.")
    result.errors.extend(issue.message for issue in check_origin_consistency(df, origin))


def save_dataset(
    df: pd.DataFrame,
    destination: str | os.PathLike[str],
    *,
    origin: DataOrigin | str,
    overwrite: bool = False,
    create_dirs: bool = True,
    validate: bool = True,
    allow_invalid: bool = False,
    allow_synthetic_in_raw: bool = False,
    config: ValidationConfig | None = None,
) -> SaveResult:
    """Save a dataset as CSV to an explicitly chosen destination.

    Nothing is written unless every check passes. The DataFrame is written as
    given (no rows dropped, no index column).

    Args:
        df: Dataset to save.
        destination: Target ``.csv`` path. There is no default location.
        origin: ``"real"`` or ``"synthetic"`` (required). Synthetic data must have
            every row marked in ``source`` (see ``mark_as_synthetic``); real data
            must contain no synthetic-marked rows.
        overwrite: If False (default) an existing file is never replaced. If
            True, the file is replaced atomically (temp file + rename).
        create_dirs: Create missing parent directories.
        validate: Run structural validation before saving.
        allow_invalid: Save even if validation finds errors (e.g. archiving an
            unmodified raw sensor log). Origin and destination rules still apply.
        allow_synthetic_in_raw: Permit synthetic data under ``data/raw/``.
        config: Validation rules.

    Returns:
        ``SaveResult``; ``saved`` is True only if the file was written.

    Raises:
        ValueError: If ``origin`` is not ``"real"`` or ``"synthetic"``.
    """
    origin_value = _normalise_origin(origin)
    dest = Path(destination).expanduser()
    result = SaveResult(path=dest)

    if not isinstance(df, pd.DataFrame):
        result.errors.append(f"Expected a pandas DataFrame, got {type(df).__name__}.")
        return result

    _pre_save_checks(df, dest, origin_value, allow_synthetic_in_raw, result)

    if validate:
        result.validation = validate_sensor_data(df, config)
        if not result.validation.is_valid:
            if allow_invalid:
                result.warnings.append("Saved despite validation errors (allow_invalid=True).")
            else:
                result.errors.append("Dataset failed validation; see result.validation for details.")

    if dest.is_dir():
        result.errors.append(f"Destination is a directory, not a file: {dest}")
    elif dest.exists() and not overwrite:
        result.errors.append(f"File already exists and overwrite=False: {dest}")
    if not dest.parent.exists() and not create_dirs:
        result.errors.append(f"Destination folder does not exist: {dest.parent}")

    if result.errors:
        return result

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        _write_csv(df, dest, overwrite)
    except FileExistsError:
        result.errors.append(f"File already exists and overwrite=False: {dest}")
        return result
    except OSError as exc:
        result.errors.append(f"Could not write {dest}: {exc}")
        return result

    result.saved = True
    return result


# --------------------------------------------------------------------------- #
# Real / synthetic separation
# --------------------------------------------------------------------------- #
def classify_row_origins(df: pd.DataFrame) -> pd.Series:
    """Classify each row as ``"synthetic"``, ``"declared"`` or ``"unspecified"``.

    ``declared`` means a non-synthetic ``source`` value is present. It is *not*
    proof the row is real; it only means it was not marked synthetic.

    Args:
        df: Any DataFrame.

    Returns:
        String Series aligned with ``df.index``. Without a ``source`` column
        every row is ``"unspecified"``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
    """
    _require_dataframe(df)
    if SOURCE_COLUMN not in df.columns:
        return pd.Series(ROW_ORIGIN_UNSPECIFIED, index=df.index, dtype=object)
    source = df[SOURCE_COLUMN]
    synthetic = source.map(is_synthetic_source).astype(bool)
    missing = missing_mask(source)
    labels = pd.Series(ROW_ORIGIN_DECLARED, index=df.index, dtype=object)
    labels[missing.to_numpy()] = ROW_ORIGIN_UNSPECIFIED
    labels[synthetic.to_numpy()] = ROW_ORIGIN_SYNTHETIC
    return labels


def separate_synthetic_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into (non-synthetic rows, synthetic rows).

    No rows are lost: every row ends up in exactly one of the two copies.
    Non-synthetic includes rows with an unspecified source; use
    ``classify_row_origins`` if that distinction matters.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
    """
    origins = classify_row_origins(df)
    synthetic = (origins == ROW_ORIGIN_SYNTHETIC).to_numpy()
    return df.loc[~synthetic].copy(), df.loc[synthetic].copy()


def mark_as_synthetic(df: pd.DataFrame, source_label: str = DEFAULT_SYNTHETIC_SOURCE) -> pd.DataFrame:
    """Return a copy with every row marked synthetic in the ``source`` column.

    Rows already marked synthetic keep their value; blank ``source`` values are
    filled. The input is not modified.

    Args:
        df: Generated/demo dataset.
        source_label: Marker to write; must itself be recognised as synthetic.

    Returns:
        A new DataFrame.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``source_label`` is not a synthetic marker, or if any row
            already declares a non-synthetic source (that row may be real data
            and is not silently relabelled).
    """
    _require_dataframe(df)
    if not is_synthetic_source(source_label):
        raise ValueError(f"source_label {source_label!r} is not recognised as a synthetic marker.")
    marked = df.copy()
    if SOURCE_COLUMN not in marked.columns:
        marked[SOURCE_COLUMN] = source_label
        return marked
    origins = classify_row_origins(marked)
    declared = (origins == ROW_ORIGIN_DECLARED).to_numpy()
    if declared.any():
        raise ValueError(
            f"{int(declared.sum())} row(s) already declare a non-synthetic source; refusing to "
            "relabel possible real data as synthetic."
        )
    unspecified = (origins == ROW_ORIGIN_UNSPECIFIED).to_numpy()
    marked[SOURCE_COLUMN] = marked[SOURCE_COLUMN].astype(object)
    marked.loc[unspecified, SOURCE_COLUMN] = source_label
    return marked


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def _require_dataframe(df: Any) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}.")


def _to_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _sensor_statistics(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    raw = df[column]
    cleaned = raw.map(lambda v: v.strip() if isinstance(v, str) else v)
    numeric = pd.to_numeric(cleaned, errors="coerce").astype(float)
    non_numeric = int((~missing_mask(raw) & numeric.isna()).sum())
    finite = numeric[numeric.abs() != float("inf")].dropna()
    has = not finite.empty
    return {
        "count": int(finite.size),
        "non_numeric_count": non_numeric,
        "mean": _to_float(finite.mean()) if has else None,
        "std": _to_float(finite.std()) if finite.size > 1 else None,
        "min": _to_float(finite.min()) if has else None,
        "median": _to_float(finite.median()) if has else None,
        "max": _to_float(finite.max()) if has else None,
    }


def _timestamp_range(df: pd.DataFrame, timestamp_format: str | None) -> dict[str, Any] | None:
    if TIMESTAMP_COLUMN not in df.columns:
        return None
    raw = df[TIMESTAMP_COLUMN]
    parsed = parse_timestamps(raw, timestamp_format)
    valid = parsed.dropna()
    present = int((~missing_mask(raw)).sum())
    return {
        "start": valid.min().isoformat() if not valid.empty else None,
        "end": valid.max().isoformat() if not valid.empty else None,
        "valid_count": int(valid.size),
        "invalid_count": present - int(valid.size),
    }


def _value_counts(series: pd.Series) -> dict[str, int]:
    present = series[~missing_mask(series).to_numpy()]
    return {str(k): int(v) for k, v in present.map(lambda v: str(v).strip()).value_counts().items()}


def summarize_dataset(df: pd.DataFrame, *, config: ValidationConfig | None = None) -> DatasetSummary:
    """Describe a dataset. Safe for empty data and missing columns.

    Statistics are computed over finite numeric values *including* values the
    validator would reject (e.g. humidity 120). Validate first if that matters.
    Summarising does not modify the DataFrame.

    Args:
        df: Dataset to summarise.
        config: Used for ``required_columns`` and ``timestamp_format``.

    Returns:
        ``DatasetSummary`` (call ``.to_dict()`` for a plain dict).

    Raises:
        TypeError: If ``df`` is not a DataFrame.
    """
    _require_dataframe(df)
    config = config or ValidationConfig()
    columns = [str(c) for c in df.columns]
    present = set(columns)
    unique_cols = not df.columns.duplicated().any()

    def col_ok(name: str) -> bool:
        return unique_cols and name in present

    has_labels = col_ok(LABEL_COLUMN)
    return DatasetSummary(
        row_count=int(len(df)),
        column_count=len(columns),
        columns=columns,
        missing_required_columns=[c for c in config.required_columns if c not in present],
        missing_values={str(c): int(missing_mask(df.iloc[:, i]).sum()) for i, c in enumerate(df.columns)},
        sensor_statistics={c: _sensor_statistics(df, c) for c in SENSOR_COLUMNS if col_ok(c)},
        timestamp_range=_timestamp_range(df, config.timestamp_format) if col_ok(TIMESTAMP_COLUMN) else None,
        source_counts=_value_counts(df[SOURCE_COLUMN]) if col_ok(SOURCE_COLUMN) else None,
        origin_counts=_origin_counts(df) if unique_cols else {},
        has_labels=has_labels,
        label_counts=_value_counts(df[LABEL_COLUMN]) if has_labels else None,
    )


def _origin_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = classify_row_origins(df).value_counts()
    return {
        key: int(counts.get(key, 0))
        for key in (ROW_ORIGIN_SYNTHETIC, ROW_ORIGIN_DECLARED, ROW_ORIGIN_UNSPECIFIED)
    }


# --------------------------------------------------------------------------- #
# Readiness and analysis preparation
# --------------------------------------------------------------------------- #
def assess_training_readiness(
    df: pd.DataFrame,
    validation: ValidationResult | None = None,
    *,
    config: ValidationConfig | None = None,
) -> TrainingReadiness:
    """Report whether a dataset is structurally usable for exploration and training.

    This is a structural gate, not scientific validation: it cannot tell
    whether labels reflect real spoilage.

    Args:
        df: Dataset to assess.
        validation: Existing result for ``df``; computed if omitted.
        config: Validation rules used when computing validation.

    Returns:
        ``TrainingReadiness``.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
    """
    _require_dataframe(df)
    validation = validation or validate_sensor_data(df, config)
    blocking: list[str] = []
    cautions: list[str] = []

    if not validation.is_valid:
        blocking.append("Dataset failed structural validation.")

    origins = classify_row_origins(df) if not df.columns.duplicated().any() else pd.Series(dtype=object)
    synthetic_count = int((origins == ROW_ORIGIN_SYNTHETIC).sum())
    unspecified_count = int((origins == ROW_ORIGIN_UNSPECIFIED).sum())
    if synthetic_count:
        blocking.append(f"{synthetic_count} row(s) are synthetic demo data and must not be used for training.")
    if unspecified_count:
        cautions.append(
            f"{unspecified_count} row(s) have no declared source; confirm they are real measurements."
        )

    label_counts: dict[str, int] = {}
    if LABEL_COLUMN not in df.columns or df.columns.duplicated().any():
        blocking.append("No 'spoilage_label' column; suitable for exploration only.")
    else:
        label_counts = _value_counts(df[LABEL_COLUMN])
        if not label_counts:
            blocking.append("The 'spoilage_label' column contains no labels.")
        elif len(label_counts) < 2:
            blocking.append("At least two label classes are needed for supervised classification.")
        unlabeled = int(missing_mask(df[LABEL_COLUMN]).sum())
        if unlabeled and label_counts:
            cautions.append(f"{unlabeled} row(s) are unlabelled and would be excluded from training.")

    exploration = validation.is_valid
    return TrainingReadiness(
        suitable_for_exploration=exploration,
        suitable_for_supervised_training=exploration and not blocking,
        blocking_reasons=blocking,
        cautions=cautions,
        labeled_row_count=sum(label_counts.values()),
        synthetic_row_count=synthetic_count,
        unspecified_origin_row_count=unspecified_count,
        label_counts=label_counts,
    )


def prepare_for_analysis(
    df: pd.DataFrame,
    validation: ValidationResult,
    *,
    config: ValidationConfig | None = None,
) -> pd.DataFrame:
    """Return a typed copy (datetime timestamps, float sensor columns) for plotting/analysis.

    Only allowed for datasets that passed validation, so type conversion
    cannot silently turn bad values into NaN/NaT. The input is not modified and
    no rows are dropped.

    Args:
        df: Dataset that was validated.
        validation: The ``ValidationResult`` for ``df``.
        config: Must match the config used for validation (timestamp format).

    Returns:
        New DataFrame with converted columns.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If ``validation`` reports errors.
    """
    _require_dataframe(df)
    if not validation.is_valid:
        raise ValueError("Cannot prepare a dataset that failed validation; fix the errors first.")
    config = config or ValidationConfig()
    typed = df.copy()
    if TIMESTAMP_COLUMN in typed.columns:
        typed[TIMESTAMP_COLUMN] = parse_timestamps(typed[TIMESTAMP_COLUMN], config.timestamp_format)
    for column in (GAS_COLUMN, TEMPERATURE_COLUMN, HUMIDITY_COLUMN):
        if column in typed.columns:
            cleaned = typed[column].map(lambda v: v.strip() if isinstance(v, str) else v)
            typed[column] = pd.to_numeric(cleaned, errors="coerce").astype(float)
    return typed
