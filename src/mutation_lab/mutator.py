"""
AuditHub Mutation Lab - Data Mutation Lab
==========================================

Intentionally injects anomalies, noise, and distribution shifts into datasets for adversarial testing.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetMutator:
    """Injects various simulated anomalies and shifts into tabular datasets.

    All mutations are seed-driven to ensure full reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize the DatasetMutator with a random seed.

        Parameters
        ----------
        seed : int
            Random seed for numpy operations.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.debug("DatasetMutator initialized with seed %d", seed)

    def _reset_rng(self) -> None:
        """Reset the random number generator using the instance seed."""
        self.rng = np.random.default_rng(self.seed)

    def inject_missing_values(self, df: pd.DataFrame, fraction: float, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Inject missing values (NaN) into specified columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        fraction : float
            Fraction of values to replace with NaN (0.0 to 1.0).
        columns : list[str] | None
            Columns to mutate. If None, applies to all columns.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        cols = columns or df_mut.columns.tolist()
        num_rows = len(df_mut)

        for col in cols:
            if col not in df_mut.columns:
                continue
            indices = self.rng.choice(df_mut.index.values, size=int(np.ceil(num_rows * fraction)), replace=False)
            df_mut.loc[indices, col] = np.nan

        logger.info("Injected missing values (fraction=%.2f) in columns %s", fraction, cols)
        return df_mut

    def inject_duplicates(self, df: pd.DataFrame, fraction: float) -> pd.DataFrame:
        """Inject duplicate rows by duplicating a subset of rows.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        fraction : float
            Fraction of rows to duplicate (0.0 to 1.0).

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        num_rows = len(df_mut)
        if num_rows == 0:
            return df_mut

        dup_size = int(num_rows * fraction)
        if dup_size == 0:
            return df_mut

        dup_indices = self.rng.choice(num_rows, size=dup_size, replace=True)
        duplicates = df_mut.iloc[dup_indices]
        
        # Append duplicates and shuffle
        df_mut = pd.concat([df_mut, duplicates], ignore_index=True)
        df_mut = df_mut.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        logger.info("Injected %d duplicate rows (fraction=%.2f)", dup_size, fraction)
        return df_mut

    def inject_outliers(self, df: pd.DataFrame, fraction: float, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Inject extreme value outliers into numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        fraction : float
            Fraction of column values to corrupt as outliers.
        columns : list[str] | None
            Numerical columns to mutate. If None, targets all numeric columns.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        cols = columns or [c for c in df_mut.columns if pd.api.types.is_numeric_dtype(df_mut[c].dtype)]
        num_rows = len(df_mut)

        for col in cols:
            if col not in df_mut.columns:
                continue
            series = df_mut[col]
            std = series.std()
            mean = series.mean()
            if pd.isna(std) or std == 0:
                std = 1.0
                mean = 0.0

            indices = self.rng.choice(df_mut.index.values, size=int(np.ceil(num_rows * fraction)), replace=False)
            # Add or subtract 5-10 standard deviations
            outliers = mean + self.rng.choice([-1.0, 1.0], size=len(indices)) * self.rng.uniform(5.0, 10.0, size=len(indices)) * std
            # Widen to float first: an integer column (including the nullable
            # Int64 used for integer columns with gaps) rejects fractional values.
            df_mut[col] = df_mut[col].astype("float64")
            df_mut.loc[indices, col] = outliers

        logger.info("Injected outliers (fraction=%.2f) in columns %s", fraction, cols)
        return df_mut

    def inject_gaussian_noise(self, df: pd.DataFrame, noise_level: float, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Inject Gaussian noise into numerical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        noise_level : float
            Standard deviation scalar (fraction of standard deviation).
        columns : list[str] | None
            Numerical columns to mutate.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        cols = columns or [c for c in df_mut.columns if pd.api.types.is_numeric_dtype(df_mut[c].dtype)]

        for col in cols:
            if col not in df_mut.columns:
                continue
            series = df_mut[col]
            std = series.std()
            if pd.isna(std) or std == 0:
                std = 1.0
            
            noise = self.rng.normal(0, noise_level * std, size=len(df_mut))
            # Noise is fractional, so widen integer columns before adding it.
            df_mut[col] = series.astype("float64") + noise

        logger.info("Injected Gaussian noise (level=%.2f) in columns %s", noise_level, cols)
        return df_mut

    def remove_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Remove features/columns from the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list[str]
            Column names to remove.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        cols_to_drop = [c for c in columns if c in df_mut.columns]
        df_mut = df_mut.drop(columns=cols_to_drop)
        logger.info("Dropped columns: %s", cols_to_drop)
        return df_mut

    def shuffle_column(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Randomly shuffle (permute) the values of specific columns.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        columns : list[str]
            Columns to shuffle.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        for col in columns:
            if col in df_mut.columns:
                vals = df_mut[col].values.copy()
                self.rng.shuffle(vals)
                df_mut[col] = vals
        logger.info("Shuffled values in columns %s", columns)
        return df_mut

    def inject_distribution_shift(
        self,
        df: pd.DataFrame,
        mean_offset_ratio: float = 1.0,
        variance_scale: float = 1.0,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Shift numerical feature distributions.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        mean_offset_ratio : float
            Offset to add, scaled by std dev (e.g. 1.0 = add 1 std dev).
        variance_scale : float
            Variance scaling factor (e.g. 2.0 = double std dev).
        columns : list[str] | None
            Columns to shift.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        cols = columns or [c for c in df_mut.columns if pd.api.types.is_numeric_dtype(df_mut[c].dtype)]

        for col in cols:
            if col not in df_mut.columns:
                continue
            series = df_mut[col]
            mean = series.mean()
            std = series.std()
            if pd.isna(std) or std == 0:
                std = 1.0

            # Apply shift: (x - mean) * scale + mean + offset
            offset = mean_offset_ratio * std
            df_mut[col] = (series - mean) * variance_scale + mean + offset

        logger.info("Injected distribution shift (offset=%.2f, scale=%.2f) in columns %s", mean_offset_ratio, variance_scale, cols)
        return df_mut

    def flip_labels(self, df: pd.DataFrame, target_column: str, fraction: float) -> pd.DataFrame:
        """Flip/randomize class labels in a categorical or boolean target column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target_column : str
            Target classification column.
        fraction : float
            Fraction of rows to flip.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        if target_column not in df_mut.columns:
            return df_mut

        series = df_mut[target_column]
        unique_classes = series.dropna().unique().tolist()
        if len(unique_classes) <= 1:
            return df_mut

        num_rows = len(df_mut)
        flip_indices = self.rng.choice(df_mut.index.values, size=int(np.ceil(num_rows * fraction)), replace=False)

        for idx in flip_indices:
            current_val = df_mut.loc[idx, target_column]
            # Pick a different class
            other_classes = [c for c in unique_classes if c != current_val]
            if other_classes:
                df_mut.loc[idx, target_column] = self.rng.choice(other_classes)

        logger.info("Flipped labels (fraction=%.2f) in target column '%s'", fraction, target_column)
        return df_mut

    def inject_class_imbalance(self, df: pd.DataFrame, target_column: str, majority_class: Any, minority_class: Any, ratio: float = 0.95) -> pd.DataFrame:
        """Resample the dataset to induce severe class imbalance.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target_column : str
            Target class column.
        majority_class : Any
            The class value to make majority.
        minority_class : Any
            The class value to make minority.
        ratio : float
            Desired fraction of majority class records in the output.

        Returns
        -------
        pd.DataFrame
            Resampled DataFrame with induced imbalance.
        """
        if target_column not in df.columns:
            return df

        df_maj = df[df[target_column] == majority_class]
        df_min = df[df[target_column] == minority_class]
        df_others = df[~df[target_column].isin([majority_class, minority_class])]

        if df_maj.empty or df_min.empty:
            return df

        # We keep all majority records and downsample minority records
        # maj_count / (maj_count + min_count) = ratio
        # => min_count = maj_count * (1 - ratio) / ratio
        maj_count = len(df_maj)
        target_min_count = int(maj_count * (1.0 - ratio) / ratio)
        target_min_count = max(1, min(target_min_count, len(df_min)))

        df_min_downsampled = df_min.sample(n=target_min_count, random_state=self.seed)
        
        # Combine
        df_mut = pd.concat([df_maj, df_min_downsampled, df_others], ignore_index=True)
        df_mut = df_mut.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        logger.info("Resampled class imbalance: %s class is now %.2f%% of binary subset.", majority_class, ratio * 100)
        return df_mut

    def inject_random_corruption(self, df: pd.DataFrame, fraction: float, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Inject random string/character noise into cells.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        fraction : float
            Fraction of cells to corrupt.
        columns : list[str] | None
            Columns to mutate.

        Returns
        -------
        pd.DataFrame
            Mutated DataFrame.
        """
        df_mut = df.copy()
        cols = columns or df_mut.columns.tolist()
        num_rows = len(df_mut)

        for col in cols:
            if col not in df_mut.columns:
                continue
            indices = self.rng.choice(df_mut.index.values, size=int(np.ceil(num_rows * fraction)), replace=False)
            for idx in indices:
                corrupted_str = "".join(self.rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789!@#$"), size=5))
                df_mut.loc[idx, col] = corrupted_str

        logger.info("Injected random alphanumeric corruption (fraction=%.2f) in columns %s", fraction, cols)
        return df_mut
