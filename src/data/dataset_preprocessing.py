# src/data/dataset_preprocessing.py
"""
Dataset-specific preprocessing for the 21 real credit datasets.

PROVENANCE: copied from the sibling TabPFNCredit project
(`1. TabPFN/TabPFNCredit/src/data/dataset_preprocessing.py`) where these
per-dataset recipes were developed and validated. Only the imports and the
staging environment-variable name were changed. Do NOT rewrite the recipes here:
they encode which columns are IDs, which leak the target, and which need log or
clip transforms, per dataset. That knowledge is not recoverable from the CSVs.

If a recipe needs fixing, fix it in BOTH projects or they will silently disagree
and their results will stop being comparable.

This module handles the critical first stage of the data pipeline:
loading raw data and transforming it into a clean, standardized format.

=============================================================================
PIPELINE OVERVIEW
=============================================================================

    Raw CSV/Parquet
          │
          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  1. LOAD RAW FILE                                                   │
    │     • Try .csv first, then .parquet                                 │
    │     • Handle MultiIndex from parquet files                          │
    └─────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  2. DATASET-SPECIFIC PREPROCESSING                                  │
    │     • Drop ID columns (no predictive value)                         │
    │     • Drop date columns (temporal leakage risk)                     │
    │     • Drop leakage columns (derived from target)                    │
    │     • Encode categorical values (text → numeric codes)              │
    │     • Handle special values (e.g., "Unknown" → 0)                   │
    │     • Apply dataset-specific transformations (log, clipping)        │
    │     • Identify target, categorical, and numerical columns           │
    └─────────────────────────────────────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  3. GENERAL POST-PROCESSING (Applied to ALL datasets)               │
    │     • LGD target clipping to [0, 1] (domain constraint)             │
    │     • Numeric value sanitization (inf → NaN, clip extremes)         │
    └─────────────────────────────────────────────────────────────────────┘
          │
          ▼
    (df, target_col, cat_cols, num_cols)
    
=============================================================================
NOTE: This module does NOT perform operations that depend on dataset 
statistics (PCA, outlier removal, constant column removal). Those operations 
are handled AFTER train/test splitting to prevent data leakage.
=============================================================================
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np

from src.utils.paths import find_raw_path

logger = logging.getLogger(__name__)
pd.set_option("future.no_silent_downcasting", True)


# =============================================================================
# FILE LOADING UTILITIES
# =============================================================================

# src/data/dataset_preprocessing.py

# At the top, replace the import:
# OLD:
# from ..utils.config_reader import load_config

# NEW: Don't import load_config at all, use default paths instead

# =============================================================================
# FILE LOADING UTILITIES
# =============================================================================

def _get_data_directory(task: str, config: Optional[dict] = None) -> Path:
    """
    Get the data directory path for a specific task.
    
    Uses default path structure: project_root/data/raw/{task}
    Config parameter kept for backward compatibility but not used.
    
    Parameters
    ----------
    task : str
        Either 'pd' (Probability of Default) or 'lgd' (Loss Given Default)
    config : dict, optional
        Deprecated. Kept for backward compatibility.
        
    Returns
    -------
    Path
        Absolute path to the task's data directory
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Default path structure: project_root/data/raw/{task}/
    if task == 'pd':
        return project_root / 'data' / 'raw' / 'pd'
    elif task == 'lgd':
        return project_root / 'data' / 'raw' / 'lgd'
    else:
        raise ValueError(f"Invalid task '{task}'. Must be 'pd' or 'lgd'.")


#: Raw CSVs that have NO header row -- their first line is already an
#: observation. pandas defaults to ``header=0`` and would silently consume it:
#: German Credit was evaluated on 999 of its 1000 records that way, and the loss
#: was invisible because the resulting column names ("A11", "6", "A34", ...) look
#: plausible and pandas de-duplicates the repeated ones into "4.1"/"1.1".
#:
#: A headerless file is identifiable without knowing the dataset: its first line
#: has the same numeric/text profile as the lines after it, whereas a real header
#: is all text above numeric data. Every other raw file here passes that test as
#: "has header", so this set is deliberately explicit rather than heuristic --
#: guessing per load would risk dropping a genuine header on a new dataset.
_HEADERLESS_RAW = frozenset({"0008.german"})


def _unmangle_first_row(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Undo pandas duplicate-column suffixes in a headerless file's first row.

    These raw CSVs were themselves produced by loading the ORIGINAL headerless
    file with the default ``header=0`` and writing it back out: the consumed
    record became the header, and pandas had already renamed its repeated values
    ("4" -> "4.1", "1" -> "1.1"). Reading with ``header=None`` recovers the row
    but keeps those corrupted fields -- for German Credit that put a third value
    of 1.1 into a binary target, which is worse than the missing row it fixes.

    A mangled field is identified by pandas' own rule: a non-integral value whose
    integer part appears elsewhere in the SAME row (that is why pandas had to
    suffix it). On a pristine headerless file nothing matches and this is a
    no-op, so restoring a clean copy of the raw data does not conflict with it.
    """
    if df.empty:
        return df

    def _as_number(value):
        """float(value) or None. Handles numpy scalars, which are NOT Python
        ints -- ``isinstance(np.int64(6), int)`` is False, and testing for that
        silently emptied the comparison set and made this a no-op."""
        if isinstance(value, bool):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    row = df.iloc[0]
    numbers = [_as_number(v) for v in row]
    integral = {int(f) for f in numbers if f is not None and f.is_integer()}
    repaired: dict = {}
    for (col, _value), number in zip(row.items(), numbers):
        if number is None or number.is_integer():
            continue
        stem = int(number)                  # 4.1 -> 4, 1.1 -> 1
        if stem in integral:
            repaired[col] = stem
    for col, value in repaired.items():
        df.loc[df.index[0], col] = value
    if repaired:
        logger.info(
            f"{name}: repaired {len(repaired)} pandas dupe-suffixed value(s) in "
            f"the recovered first record ({', '.join(map(str, repaired.values()))})"
        )
    return df


def _load_dataset_file(dataset_path: Path) -> pd.DataFrame:
    """
    Load a dataset from CSV or Parquet format.

    Attempts to load .csv first, then falls back to .parquet.
    Handles MultiIndex from parquet files by flattening.
    Files listed in :data:`_HEADERLESS_RAW` are read with ``header=None`` so
    their first observation is not mistaken for a header, then passed through
    :func:`_unmangle_first_row` to repair that record's suffixed values.
    """
    csv_path = Path(str(dataset_path) + '.csv')
    parquet_path = Path(str(dataset_path) + '.parquet')

    if csv_path.exists():
        headerless = dataset_path.name in _HEADERLESS_RAW
        df = pd.read_csv(csv_path, low_memory=False,
                         header=None if headerless else 'infer')
        if headerless:
            df = _unmangle_first_row(df, csv_path.name)
            # Self-check: every line of a headerless file must survive as a row.
            # Cheap, and it would have caught the original bug immediately.
            with csv_path.open(encoding='utf-8', errors='ignore') as fh:
                n_lines = sum(1 for line in fh if line.strip())
            if len(df) != n_lines:
                logger.warning(
                    f"{csv_path.name}: read {len(df)} rows from {n_lines} "
                    f"non-empty lines of a headerless file"
                )
        logger.info(f"Loaded CSV: {csv_path.name}, shape = {df.shape}")
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded Parquet: {parquet_path.name}, shape = {df.shape}")
        
        if isinstance(df.index, pd.MultiIndex):
            logger.info(f"Resetting MultiIndex with {df.index.nlevels} levels")
            df = df.reset_index(drop=True)
    else:
        raise FileNotFoundError(
            f"Dataset file not found. Checked:\n"
            f"  - {csv_path}\n"
            f"  - {parquet_path}"
        )
    
    return df


# =============================================================================
# GENERAL PREPROCESSING UTILITIES (NON-LEAKY)
# =============================================================================

def _clip_lgd_target(
    df: pd.DataFrame, 
    target_col: str,
    lower: float = 0.0,
    upper: float = 1.0
) -> Tuple[pd.DataFrame, int]:
    """
    Clip LGD target values to valid range [0, 1].
    
    LGD represents fraction of exposure lost - must be between 0 and 1.
    Values outside this range indicate data quality issues.
    
    This is a DOMAIN CONSTRAINT, not a statistical operation, so it's
    safe to apply before splitting (no leakage).
    """
    df = df.copy()
    
    original = df[target_col]
    n_below = (original < lower).sum()
    n_above = (original > upper).sum()
    n_clipped = n_below + n_above
    
    df[target_col] = df[target_col].clip(lower=lower, upper=upper)
    
    if n_clipped > 0:
        logger.info(
            f"LGD target clipping: {n_clipped} values clipped to [{lower}, {upper}] "
            f"({n_below} below, {n_above} above)"
        )
    
    return df, n_clipped


def _sanitize_numeric_values(
    df: pd.DataFrame, 
    num_cols: List[str],
    clip_threshold: float = 1e10
) -> pd.DataFrame:
    """
    Sanitize numeric columns to prevent numerical overflow.
    
    Operations:
    1. Replace infinity values (±inf) with NaN
    2. Clip extreme values to ±clip_threshold
    
    This is DATA CLEANING, not statistical modeling, so it's safe to 
    apply before splitting (no leakage).
    """
    df = df.copy()
    
    n_inf_replaced = 0
    n_clipped = 0
    
    for col in num_cols:
        if col not in df.columns:
            continue
            
        inf_mask = np.isinf(df[col])
        n_inf_replaced += inf_mask.sum()
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        extreme_mask = (df[col].abs() > clip_threshold) & df[col].notna()
        n_clipped += extreme_mask.sum()
        df[col] = df[col].clip(lower=-clip_threshold, upper=clip_threshold)
    
    if n_inf_replaced > 0 or n_clipped > 0:
        logger.info(
            f"Sanitized: {n_inf_replaced} inf→NaN, {n_clipped} clipped to ±{clip_threshold:.0e}"
        )
    
    return df


# =============================================================================
# MAIN PREPROCESSING FUNCTION
# =============================================================================

def preprocess_dataset_specific(
    task: str, 
    dataset: str, 
    raw_dir: Path = None,
    config: dict = None  # Keep parameter but mark as deprecated
) -> Tuple[pd.DataFrame, str, List[str], List[str]]:
    """
    Load and preprocess a specific dataset by name and task.
    
    This function performs ONLY dataset-specific cleaning and transformations
    that are safe to apply before train/test splitting (no leakage).
    
    Statistical operations (PCA, outlier removal, constant column removal)
    are NOT performed here - they happen after splitting in data_feeder.py.
    
    Parameters
    ----------
    task : str
        Task type: 'pd' or 'lgd'
    dataset : str
        Dataset identifier, e.g., '0013.hmeq', '0001.heloc'
    raw_dir : Path, optional
        Override for data directory. If None, uses default: data/raw/{task}
    config : dict, optional
        Deprecated. Kept for backward compatibility.
        
    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe
    target_col : str
        Name of target column
    cat_cols : list[str]
        List of categorical column names
    num_cols : list[str]
        List of numerical column names
    """
    
    # =========================================================================
    # STEP 1: Load raw data file
    # =========================================================================
    
    if raw_dir is None:
        # Locate the raw file: repo-local data/raw first, then project storage.
        dataset_path = find_raw_path(task, dataset)
        if dataset_path is None:
            # Report EVERY root that was searched. Falling back to the repo-local
            # default made the error name a single path, which reads as "only the
            # repo was checked" and sent a cluster debugging session chasing the
            # wrong thing -- the file was on project storage but the roots were
            # never listed.
            import os
            from src.utils.paths import raw_task_dirs, staging_root
            searched = "\n".join(
                f"  - {d / dataset}.csv / .parquet    (dir exists: {d.is_dir()})"
                for d in raw_task_dirs(task)
            )
            raise FileNotFoundError(
                f"Raw dataset {task}/{dataset} not found in any data root.\n"
                f"{searched}\n"
                f"  staging_root() = {staging_root()}  "
                f"(CREDITICL_STAGING_ROOT={os.environ.get('CREDITICL_STAGING_ROOT')!r})\n"
                f"Place the file in one of the directories above, or set "
                f"CREDITICL_STAGING_ROOT to the project-storage root that holds it."
            )
    else:
        dataset_path = raw_dir / task / dataset
    
    df = _load_dataset_file(dataset_path)
    initial_shape = df.shape
    logger.info(f"Loaded {dataset} ({task}), shape = {initial_shape}")

    # =========================================================================
    # STEP 2: Dataset-specific preprocessing
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # PROBABILITY OF DEFAULT (PD) DATASETS
    # -------------------------------------------------------------------------
    
    if task == "pd":

        if dataset == "0001.gmsc":
            target_col = "SeriousDlqin2yrs"
            if target_col not in df.columns:
                raise ValueError(f"Expected target '{target_col}' not in columns.")
            
            # Apply dataset-specific transformations
            if "RevolvingUtilizationOfUnsecuredLines" in df.columns:
                df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1)
            
            if "NumberOfTime30-59DaysPastDueNotWorse" in df.columns:
                df["NumberOfTime30-59DaysPastDueNotWorse"] = df["NumberOfTime30-59DaysPastDueNotWorse"].clip(0, 5)
            
            if "DebtRatio" in df.columns:
                df["DebtRatio"] = np.log1p(df["DebtRatio"].clip(lower=0))
            
            if "MonthlyIncome" in df.columns:
                df["MonthlyIncome"] = np.log1p(df["MonthlyIncome"].clip(lower=0))
            
            if "NumberOfTimes90DaysLate" in df.columns:
                df["NumberOfTimes90DaysLate"] = df["NumberOfTimes90DaysLate"].clip(0, 5)
            
            if "NumberOfTime60-89DaysPastDueNotWorse" in df.columns:
                df["NumberOfTime60-89DaysPastDueNotWorse"] = df["NumberOfTime60-89DaysPastDueNotWorse"].clip(0, 5)
            
            logger.info("Applied gmsc-specific transformations: capping and log transforms")
            
            cat_cols: List[str] = []
            num_cols = [c for c in df.columns if c != target_col]

        elif dataset == "0002.taiwan_creditcard":
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])
            if "SEX" in df.columns:
                df["SEX"] = df["SEX"].replace({2: 1, 1: 0, "2": 1, "1": 0})
            
            if "LIMIT_BAL" in df.columns:
                df["LIMIT_BAL"] = np.log1p(df["LIMIT_BAL"].clip(lower=0))
            
            if "BILL_AMT1" in df.columns:
                df["BILL_AMT1"] = np.log1p(df["BILL_AMT1"].clip(lower=0))
            
            logger.info("Applied taiwan_creditcard-specific transformations: log transforms")
            
            target_col = "default.payment.next.month"
            if target_col not in df.columns:
                raise ValueError(f"Expected target '{target_col}' not in columns.")
            cat_cols: List[str] = []
            num_cols = [c for c in df.columns if c != target_col]
        
        elif dataset == "0003.vehicle_loan":
            drop_cols = [
                "UniqueID", "branch_id", "supplier_id",
                "Current_pincode_ID", "Employee_code_ID", "MobileNo_Avl_Flag",
            ]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            def _to_year(two_digit: str) -> int:
                s = str(two_digit)
                digits = "".join([ch for ch in s if ch.isdigit()])
                if len(digits) >= 2:
                    yy = int(digits[-2:])
                else:
                    yy = int(digits) if digits else 0
                return 2000 + yy if 0 <= yy < 20 else 1900 + yy

            if "Date.of.Birth" in df.columns:
                df["Date.of.Birth"] = df["Date.of.Birth"].apply(_to_year)
            if "DisbursalDate" in df.columns:
                df["DisbursalDate"] = df["DisbursalDate"].apply(_to_year)

            if {"Date.of.Birth", "DisbursalDate"}.issubset(df.columns):
                df["Age"] = df["DisbursalDate"] - df["Date.of.Birth"]
                df = df.drop(columns=["DisbursalDate", "Date.of.Birth"])

            if "PERFORM_CNS.SCORE.DESCRIPTION" in df.columns:
                df.replace(
                    {
                        "PERFORM_CNS.SCORE.DESCRIPTION": {
                            "C-Very Low Risk": "Very Low Risk",
                            "A-Very Low Risk": "Very Low Risk",
                            "D-Very Low Risk": "Very Low Risk",
                            "B-Very Low Risk": "Very Low Risk",
                            "M-Very High Risk": "Very High Risk",
                            "L-Very High Risk": "Very High Risk",
                            "F-Low Risk": "Low Risk",
                            "E-Low Risk": "Low Risk",
                            "G-Low Risk": "Low Risk",
                            "H-Medium Risk": "Medium Risk",
                            "I-Medium Risk": "Medium Risk",
                            "J-High Risk": "High Risk",
                            "K-High Risk": "High Risk",
                        }
                    },
                    inplace=True,
                )
                risk_map = {
                    "No Bureau History Available": -1,
                    "Not Scored: No Activity seen on the customer (Inactive)": -1,
                    "Not Scored: Sufficient History Not Available": -1,
                    "Not Scored: No Updates available in last 36 months": -1,
                    "Not Scored: Only a Guarantor": -1,
                    "Not Scored: More than 50 active Accounts found": -1,
                    "Not Scored: Not Enough Info available on the customer": -1,
                    "Very Low Risk": 4,
                    "Low Risk": 3,
                    "Medium Risk": 2,
                    "High Risk": 1,
                    "Very High Risk": 0,
                }
                df["PERFORM_CNS.SCORE.DESCRIPTION"] = df["PERFORM_CNS.SCORE.DESCRIPTION"].map(risk_map)

            def _months(s: str) -> int:
                if pd.isna(s):
                    return np.nan
                parts = str(s).split()
                years, months = 0, 0
                for i, tok in enumerate(parts):
                    if "yr" in tok:
                        try:
                            years = int(parts[i - 1])
                        except Exception:
                            pass
                    if "mon" in tok:
                        try:
                            months = int(parts[i - 1])
                        except Exception:
                            pass
                return years * 12 + months

            if "AVERAGE.ACCT.AGE" in df.columns:
                df["AVERAGE.ACCT.AGE"] = df["AVERAGE.ACCT.AGE"].apply(_months)
            if "CREDIT.HISTORY.LENGTH" in df.columns:
                df["CREDIT.HISTORY.LENGTH"] = df["CREDIT.HISTORY.LENGTH"].apply(_months)

            target_col = "loan_default"
            if target_col not in df.columns:
                raise ValueError(f"Expected target '{target_col}' not in columns.")

            explicit_cat = ["manufacturer_id", "Employment.Type", "State_ID"]
            explicit_num = [
                "disbursed_amount", "asset_cost", "ltv", "Aadhar_flag", "PAN_flag",
                "VoterID_flag", "Driving_flag", "Passport_flag", "PERFORM_CNS.SCORE",
                "PERFORM_CNS.SCORE.DESCRIPTION", "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS",
                "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT",
                "PRI.DISBURSED.AMOUNT", "SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS",
                "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE", "SEC.SANCTIONED.AMOUNT",
                "SEC.DISBURSED.AMOUNT", "PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT",
                "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
                "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH", "NO.OF_INQUIRIES", "Age",
            ]
            cat_cols = [c for c in explicit_cat if c in df.columns and c != target_col]
            num_cols = [c for c in explicit_num if c in df.columns and c != target_col]

            remaining_num = (df.select_dtypes(include=["number"]).columns
                        .drop([target_col] + num_cols, errors="ignore").tolist())
            num_cols = num_cols + remaining_num

            if "disbursed_amount" in df.columns:
                df["disbursed_amount"] = np.log1p(df["disbursed_amount"].clip(lower=0))
            
            if "asset_cost" in df.columns:
                df["asset_cost"] = np.log1p(df["asset_cost"].clip(lower=0))
            
            logger.info("Applied vehicle_loan-specific transformations: log transforms")

            cat_cols = [c for c in cat_cols if c != target_col]
            num_cols = [c for c in num_cols if c != target_col]

        elif dataset == "0004.lendingclub":
            target_col = "not.fully.paid"
            if target_col not in df.columns:
                raise ValueError("Expected 'not.fully.paid' in LendingClub dataset.")

            if "revol.bal" in df.columns:
                df["revol.bal"] = np.log1p(df["revol.bal"].clip(lower=0))
            
            logger.info("Applied lendingclub-specific transformations: log transform")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0005.myhom":
            if "loan_id" in df.columns:
                df = df.drop(columns=["loan_id"], errors="ignore")

            target_col = "loan_default"
            if target_col not in df.columns:
                raise ValueError("Expected 'loan_default' in myhom dataset.")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0006.hackerearth":
            drop_cols = ["member_id", "batch_enrolled", "emp_title", "pymnt_plan",
                        "desc", "title", "zip_code"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

            if "emp_length" in df.columns:
                df["emp_length"] = df["emp_length"].replace("< 1 year", 0)
                df["emp_length"] = df["emp_length"].astype(str).str.replace(" years", "", regex=False)
                df["emp_length"] = df["emp_length"].astype(str).str.replace(" year", "", regex=False)
                df["emp_length"] = df["emp_length"].replace("10+", 11)
                df["emp_length"] = pd.to_numeric(df["emp_length"], errors="coerce")

            if "last_week_pay" in df.columns:
                df["last_week_pay"] = df["last_week_pay"].astype(str).str.replace("th week", "", regex=False)
                df["last_week_pay"] = df["last_week_pay"].replace("NA", np.nan)
                df["last_week_pay"] = pd.to_numeric(df["last_week_pay"], errors="coerce")

            target_col = "loan_status"
            if target_col not in df.columns:
                raise ValueError("Expected 'loan_status' in hackerearth dataset.")
            df = df.dropna(subset=[target_col])

            potential_cat_cols = [
                "addr_state", "home_ownership", "verification_status",
                "purpose", "application_type", "grade", "sub_grade", "initial_list_status"
            ]
            cat_cols = [c for c in potential_cat_cols if c in df.columns]

            for c in cat_cols:
                df[c] = df[c].astype("category")
                df[c] = df[c].cat.codes

            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()
            num_cols = [c for c in num_cols if c not in cat_cols]

        elif dataset == "0007.cobranded":
            df = df.replace(["na", "missing"], np.nan)

            if "application_key" in df.columns:
                df = df.drop(columns=["application_key"], errors="ignore")

            if "mvar47" in df.columns:
                df["mvar47"] = df["mvar47"].replace({"C": 1, "L": 0})

            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except Exception:
                    pass

            for i in range(1, 16):
                col_name = f"mvar{i}"
                if col_name in df.columns:
                    df[col_name] = np.log1p(df[col_name].clip(lower=0))
            
            logger.info("Applied cobranded-specific transformations: log transforms for mvar1-mvar15")

            target_col = "default_ind"
            if target_col not in df.columns:
                raise ValueError("Expected 'default_ind' in cobranded dataset.")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0008.german":
            if df.columns[0] == 0 or "Unnamed: 0" in df.columns:
                n_cols = df.shape[1]
                df.columns = [f"feature_{i}" for i in range(1, n_cols)] + ["target"]
            else:
                if "target" not in df.columns:
                    n_cols = len(df.columns)
                    df.columns = [f"feature_{i}" for i in range(1, n_cols)] + ["target"]

            target_col = "target"
            df = df.dropna(subset=[target_col])
            df[target_col] = df[target_col].replace({1: 0, 2: 1})

            cat_cols = [
                "feature_1", "feature_3", "feature_4", "feature_6", "feature_7",
                "feature_9", "feature_10", "feature_12", "feature_14", "feature_15",
                "feature_17", "feature_19", "feature_20"
            ]
            num_cols = [
                "feature_2", "feature_5", "feature_8", "feature_11",
                "feature_13", "feature_16", "feature_18"
            ]

            cat_cols = [c for c in cat_cols if c in df.columns]
            num_cols = [c for c in num_cols if c in df.columns]

        elif dataset == "0009.bank_status":
            df = df.dropna(how="all").reset_index(drop=True)
            df = df.drop(columns=[c for c in ["Loan ID", "Customer ID"] if c in df.columns], errors="ignore")

            if "Loan Status" in df.columns:
                df["Loan Status"] = df["Loan Status"].replace({"Fully Paid": 0, "Charged Off": 1})
                df["Loan Status"] = pd.to_numeric(df["Loan Status"], errors="coerce")

            if "Term" in df.columns:
                df["Term"] = df["Term"].replace({"Short Term": 0, "Long Term": 1})
                df["Term"] = pd.to_numeric(df["Term"], errors="coerce")

            if "Home Ownership" in df.columns:
                home_map = {"Own Home": 0, "Home Mortgage": 1, "HaveMortgage": 1, "Rent": 2}
                df["Home Ownership"] = df["Home Ownership"].replace(home_map)
                df["Home Ownership"] = pd.to_numeric(df["Home Ownership"], errors="coerce")

            if "Purpose" in df.columns:
                purpose_map = {
                    "Debt Consolidation": 0, "Debt Consolidation Loan": 0,
                    "Home Improvements": 1, "Home Improvement": 1,
                    "Buy House": 2, "Buy a Car": 3, "major_purchase": 4,
                    "Business Loan": 5, "small_business": 5,
                    "Take a Trip": 6, "Vacation": 6, "Other": 7, "other": 7
                }
                df["Purpose"] = df["Purpose"].replace(purpose_map)
                df["Purpose"] = pd.to_numeric(df["Purpose"], errors="coerce")

            if "Years in current job" in df.columns:
                df["Years in current job"] = df["Years in current job"].replace("< 1 year", 0)
                df["Years in current job"] = (df["Years in current job"].astype(str)
                                            .str.replace(" years", "", regex=False))
                df["Years in current job"] = (df["Years in current job"].astype(str)
                                            .str.replace(" year", "", regex=False))
                df["Years in current job"] = df["Years in current job"].replace("10+", 11)
                df["Years in current job"] = pd.to_numeric(df["Years in current job"], errors="coerce")

            target_col = "Loan Status"
            df = df.dropna(subset=[target_col])

            if "Credit Score" in df.columns:
                df["Credit Score"] = df["Credit Score"].clip(0, 1000)
            
            logger.info("Applied bank_status-specific transformations: capping Credit Score")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0010.thomas":
            target_col = "BAD"
            if target_col not in df.columns:
                raise ValueError("Expected 'BAD' in Thomas dataset.")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0011.loan_default":
            df = df.apply(pd.to_numeric, errors="coerce")
            if "id" in df.columns:
                df = df.drop(columns=["id"], errors="ignore")

            df = df.loc[:, (df != df.iloc[0]).any()]

            target_col = "loss"
            if target_col not in df.columns:
                raise ValueError("Expected 'loss' in loan_default dataset.")

            df[target_col] = np.where(df[target_col] == 0, 0, 1)

            cat_cols = []
            num_cols = [c for c in df.columns if c != target_col]

        elif dataset == "0012.home_credit":
            if "SK_ID_CURR" in df.columns:
                df = df.drop(columns=["SK_ID_CURR"], errors="ignore")

            target_col = "TARGET"
            if target_col not in df.columns:
                raise ValueError("Expected 'TARGET' in Home Credit dataset.")

            cat_cols = [
                "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
                "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
                "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"
            ]
            cat_cols = [c for c in cat_cols if c in df.columns]
            num_cols = [c for c in df.columns if c not in cat_cols + [target_col]]

        elif dataset == "0013.hmeq":
            target_col = "BAD"
            if target_col not in df.columns:
                raise ValueError("Expected 'BAD' in HMEQ dataset.")

            cat_cols = ["REASON", "JOB"]
            num_cols = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG",
                        "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]

            cat_cols = [c for c in cat_cols if c in df.columns]
            num_cols = [c for c in num_cols if c in df.columns]

        elif dataset == "0014.algorithmwatch":
            target_col = "arrears"
            if target_col not in df.columns:
                raise ValueError("Expected 'arrears' in AlgorithmWatch dataset.")
            
            df[target_col] = df[target_col].astype(int)
            
            cat_cols = []
            num_cols = [c for c in df.columns if c != target_col]

        else:
            raise ValueError(f"No preprocessing routine defined for PD dataset: {dataset}")

    # -------------------------------------------------------------------------
    # LOSS GIVEN DEFAULT (LGD) DATASETS
    # -------------------------------------------------------------------------
    
    elif task == "lgd":

        if dataset == "0001.heloc":
            # DefPrinBal is intentionally NOT in this drop list: the
            # log-transform block below operates on it, which is the
            # intended handling for that column.
            df = df.drop(columns=["REC", "DLGD_Econ", "PrinBal", "PayOff",
                                "DefPayOff", "ObsDT", "DefDT"], errors="ignore")

            if "LienPos" in df.columns:
                df["LienPos"] = df["LienPos"].replace({"Unknow": 0, "First": 1, "Second": 2})
                df = df.infer_objects(copy=False)

            if "Age" in df.columns:
                n_clipped = ((df["Age"] < 0) | (df["Age"] > 20)).sum()
                if n_clipped > 0:
                    logger.info(f"Clipping {n_clipped} Age values to [0, 20]")
                df["Age"] = df["Age"].clip(lower=0, upper=20)

            if "AvailAmt" in df.columns:
                n_negative = (df["AvailAmt"] < 0).sum()
                if n_negative > 0:
                    logger.info(f"Clipping {n_negative} negative AvailAmt values to 0")
                    df["AvailAmt"] = df["AvailAmt"].clip(lower=0)
                logger.info("Transforming AvailAmt to log scale using log1p")
                df["AvailAmt"] = np.log1p(df["AvailAmt"])

            if "DefPrinBal" in df.columns:
                n_negative = (df["DefPrinBal"] < 0).sum()
                if n_negative > 0:
                    logger.info(f"Clipping {n_negative} negative DefPrinBal values to 0")
                    df["DefPrinBal"] = df["DefPrinBal"].clip(lower=0)
                logger.info("Transforming DefPrinBal to log scale using log1p")
                df["DefPrinBal"] = np.log1p(df["DefPrinBal"])

            target_col = "LGD_ACTG"
            if target_col not in df.columns:
                raise ValueError("Expected 'LGD_ACTG' in HELOC dataset.")

            # Remove duplicate rows (fixes data leakage)
            original_size = len(df)
            feature_cols = [c for c in df.columns if c != target_col]
            df = df.drop_duplicates(subset=feature_cols, keep='first')
            n_removed = original_size - len(df)
            if n_removed > 0:
                logger.info(f"Removed {n_removed} duplicate rows ({100*n_removed/original_size:.2f}%) from {dataset}")
            
            cat_cols = []
            num_cols = [c for c in ["PortNum", "AvailAmt", "LTV", "LienPos", "Age",
                                    "CurrEquifax", "Utilization", "DefPrinBal", "PD_Rnd"]
                        if c in df.columns]

        elif dataset == "0002.loss2":
            drop_cols = [
                "_ELGDnum1", "_ELGDnum2", "id1", "Alltel_Client",
                "REO_Appraisal_Date", "Origination_Date", "date_vintage_year",
                "date_vintage_year_month", "Servicing_Loss", "lr1", "lss_rt",
                "_Loss_Amount", "lss_amt","Investor_Category","_Proceeds",
                "_Net_sales_Proceeds","_reo_sales_price","_SellingCosts",
                "REO_Sales_Price"
            ]
            df = df.drop(columns=drop_cols, errors="ignore")

            target_col = "_ELGD"
            if target_col not in df.columns:
                raise ValueError("Expected '_ELGD' in loss2 dataset.")

            df = df.dropna(subset=[target_col])

            # Remove duplicate rows
            original_size = len(df)
            feature_cols = [c for c in df.columns if c != target_col]
            df = df.drop_duplicates(subset=feature_cols, keep='first')
            n_removed = original_size - len(df)
            if n_removed > 0:
                logger.info(f"Removed {n_removed} duplicate rows ({100*n_removed/original_size:.2f}%) from {dataset}")
            
            log_transform_cols = [
                "UPB_At_Resolution", "Unpaid_Interest", "Total_Debt",
                "REO_Appraisal_Amount", "Original_UPB", "amount_funder",
                "amount_appraised", "amount_funded",
                "_reo_sales_price", "_SellingCosts", "_adv_interest1M",
                "_adv_interest", "_ELAO", "_Accruad_int", "_EAD",
                "_Net_Sales_Proceeds", "_Miclaimbal", "_Mirecovery", "_Proceeds"
            ]
            for col in log_transform_cols:
                if col in df.columns:
                    df[col] = np.log1p(df[col].clip(lower=0))
            
            logger.info(f"Applied loss2-specific transformations: log transforms for {len([c for c in log_transform_cols if c in df.columns])} columns")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0003.axa":
            df = df.drop(columns=["Recovery_rate", "y_logistic", "lnrr",
                                "Y_probit", "event"], errors="ignore")

            target_col = "lgd_time"
            if target_col not in df.columns:
                raise ValueError("Expected 'lgd_time' in axa dataset.")

            cat_cols = []
            num_cols = [c for c in ["LTV", "purpose1"] if c in df.columns]

        elif dataset == "0004.base_model":
            drop_cols = [
                'DEAL_DocUNID', 'DEAL_MainID', 'DEAL_FacilityIdentifier', 'DEAL_StarWebIdentifier',
                'DFLT_MainID', 'DFLT_SPM', 'DFLT_DAI', 'DFLT_BDR', 'DFLT_LegalEntityName',
                'DFLT_StarWeb_PCRU', 'DFLT_ClientNAE', 'DFLT_ParentSPM', 'DFLT_ParentSIREN',
                'DFLT_ParentDAI', 'DFLT_ParentLegalEntityName', 'DFLT_ParentPCRU', 'DFLT_ParentNAE',
                'DFLT_subject', 'FCLT_DealUNID', 'FCLT_BCEIdentifier', 'FCLT_Identifier',
                'FCLT_BookingUnit', 'fclt_docunid',
                'DEAL_TransactionStartDate', 'DEAL_TransactionEndDate', 'DEAL_DateComposed',
                'DEAL_LastUpDate', 'DFLT_SGDefaultDate', 'DFLT_PublicDefaultDate',
                'DFLT_EndDefaultDate', 'DFLT_SGRatingDate', 'DFLT_RatingDate1YPD',
                'DFLT_ParentDefaultDateIf', 'DFLT_ParentSGRatingDate', 'DFLT_ParentRatingDate1YPD',
                'DFLT_DateComposed', 'DFLT_LastUpdate', 'DATE_DECLAR_CT', 'FCLT_StartDate',
                'FCLT_EndDate', 'FCLT_DefaultDate', 'FCLT_DateComposed', 'FCLT_LastUpdate', 'date',
                'FCLT_CommentsOnLimit', 'FCLT_subject', 'DEAL_GoverningLawRecovery',
                'DEAL_PFRU', 'DEAL_subject',
                'lgd_cat_15', 'lgd_cat_10', 'lgd_cat_5', 'LGD_log', 'LGD_deF',
                'LGD_norm', 'sortie', 'RecAssoFlag',
                'DEAL_ConstructionEndDate', 'DEAL_ConstructionStartDate', 'DEAL_AverageRents',
                'DEAL_ExpectedVacancyRate', 'DEAL_StrikeLESSEEOption', 'DEAL_StatusUpDate',
                'DEAL_DeleteDate', 'DFLT_DeleteStatus', 'DFLT_DeletedDate', 'DFLT_JRIRating',
                'DFLT_ParentJRIRating', 'FCLT_DeleteStatus', 'FCLT_IrrevocableLocOffshore',
                'FCLT_DeleteDate', 'flag_eps', 'flag_fcltcurrency', 'fac_ss_commcov',
                'flag_pme', 'Flag_specifique', 'flag_specperi', 'Categorie_AV'
            ]
            df = df.drop(columns=drop_cols, errors="ignore")

            target_col = "LGD_brute"
            if target_col not in df.columns:
                raise ValueError("Expected 'LGD_brute' in base_model dataset.")
            df = df.dropna(subset=[target_col])

            missing_ratio = df.isnull().mean()
            high_missing_cols = missing_ratio[missing_ratio > 0.8].index.tolist()
            if high_missing_cols:
                df = df.drop(columns=high_missing_cols, errors="ignore")
                logger.info(f"Dropped {len(high_missing_cols)} columns with >80% missing")

            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0005.base_modelisation":
            df = df.drop(columns=["Ident_cliej_spm", "ID_CONC_ORIGIN_CDL",
                                "id_crc", "id_unique"], errors="ignore")

            leakage_cols = [
                "lgd_5_sscout_ligne", "lgd_corr", "lgd_defaut_nt", "lgd_3class",
                "lgd_2class", "lgd_log", "lgd_t", "lgd_1log", "logit_lgd",
                "Dt_entree_defaut", "Dt_sortie_defaut", "flag_defaut_moins_1an",
                "auto_av_defaut", "util_av_defaut", "defaut_clos", "defaut_clos_4nonclos",
                "duree_1A_av_defaut", "util_av_defaut_tot", "auto_av_defaut_tot",
                "defaut_M1Y", "defaut_P1Y"
            ]
            df = df.drop(columns=leakage_cols, errors="ignore")

            target_col = "lgd_defaut"
            if target_col not in df.columns:
                raise ValueError("Expected 'lgd_defaut' in base_modelisation dataset.")

            df = df.dropna(subset=[target_col])
            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0006.lgd_freddie":
            if "loan_id" in df.columns:
                df = df.drop(columns=["loan_id"], errors="ignore")
            
            target_col = "lgd"
            if target_col not in df.columns:
                raise ValueError("Expected 'lgd' in lgd_freddie dataset.")
            
            df = df.dropna(subset=[target_col])
            
            if "number_of_units" in df.columns:
                df["number_of_units"] = df["number_of_units"].clip(0, 5)
            
            if "dti" in df.columns:
                df["dti"] = df["dti"].clip(0, 100)
            
            logger.info("Applied lgd_freddie-specific transformations: capping number_of_units and dti")
            
            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        elif dataset == "0007.lgd_lendingclub":
            drop_cols = ["addr_state", "purpose", "id"]
            df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            
            target_col = "lgd"
            if target_col not in df.columns:
                raise ValueError("Expected 'lgd' in lgd_lendingclub dataset.")
            
            df = df.dropna(subset=[target_col])
            
            if "months_since_earliest" in df.columns:
                df["months_since_earliest"] = df["months_since_earliest"].clip(0, 500)
            
            if "annual_inc" in df.columns:
                df["annual_inc"] = np.log1p(df["annual_inc"].clip(lower=0))
            
            logger.info("Applied lgd_lendingclub-specific transformations: capping months_since_earliest and log transform annual_inc")
            
            cat_cols = (df.select_dtypes(include=["object", "category"])
                    .drop(columns=[target_col], errors="ignore").columns.tolist())
            num_cols = df.select_dtypes(include=["number"]).columns.drop(target_col, errors="ignore").tolist()

        else:
            raise ValueError(f"No preprocessing routine defined for LGD dataset: {dataset}")

    else:
        raise ValueError("Task must be either 'pd' or 'lgd'")

    # =========================================================================
    # STEP 3: General post-processing (NON-LEAKY operations only)
    # =========================================================================
    
    # 3a. LGD target clipping to [0, 1] (domain constraint - no leakage)
    if task == "lgd":
        df, n_clipped = _clip_lgd_target(df, target_col, lower=0.0, upper=1.0)
    
    # 3b. Sanitize numeric values (data cleaning - no leakage)
    df = _sanitize_numeric_values(df, num_cols)
    
    # Log final summary
    final_shape = df.shape
    logger.info(
        f"Dataset-specific preprocessing complete: {final_shape[0]} samples, "
        f"{len(num_cols)} num, {len(cat_cols)} cat"
    )

    return df, target_col, cat_cols, num_cols