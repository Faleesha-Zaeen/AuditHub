"""
End-to-end integration tests for the ingestion pipeline.

Tests the full flow:
  DatasetLoader → DatasetAnalyzer → DatasetFingerprint → DatasetRegistry

Verifies that all components work together correctly.
"""

import json
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

from src.ingestion.dataset_analyzer import DatasetAnalyzer
from src.ingestion.dataset_fingerprint import FingerprintGenerator
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.exceptions import CorruptedDatasetError
from src.utils.dataset_registry import DatasetRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def csv_dataset(tmp_path: Path) -> Path:
    """Create a small CSV dataset for integration testing."""
    path = tmp_path / "iris.csv"
    pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0],
        "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6],
        "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4],
        "species": ["setosa", "setosa", "setosa", "setosa", "setosa"],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def classification_dataset(tmp_path: Path) -> Path:
    """Create a binary classification dataset."""
    path = tmp_path / "binary_classification.csv"
    pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0, 1, 0, 1, 0],
    }).to_csv(path, index=False)
    return path


@pytest.fixture
def registry(tmp_path: Path) -> Generator[DatasetRegistry, None, None]:
    """Create a temporary DatasetRegistry."""
    db_path = tmp_path / "test_integration.db"
    reg = DatasetRegistry(db_path=str(db_path))
    yield reg
    reg.close()
    # Clean up SQLite file
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        pass


# ============================================================================
# Test: Full pipeline — Load → Analyze → Fingerprint → Registry
# ============================================================================


class TestFullIngestionPipeline:
    """End-to-end ingestion pipeline tests."""

    def test_load_analyze_flow(self, csv_dataset: Path):
        """Load a CSV, then analyze it."""
        loader = DatasetLoader()
        df, metadata = loader.load(csv_dataset)

        analyzer = DatasetAnalyzer(generate_fingerprint=True)
        summary = analyzer.analyze(df, source_filename=csv_dataset.name)

        # Load result
        assert len(df) == 5
        assert len(df.columns) == 4

        # Metadata result
        assert metadata.filename == "iris.csv"
        assert metadata.shape == (5, 4)
        assert metadata.checksum is not None
        assert len(metadata.checksum) == 64

        # Analysis result
        assert summary.general["num_rows"] == 5
        assert summary.general["num_columns"] == 4
        assert len(summary.column_stats) == 4
        assert summary.fingerprint is not None

    def test_load_to_registry_full(self, csv_dataset: Path, registry: DatasetRegistry):
        """Load dataset, analyze it, and register metadata in the registry."""
        loader = DatasetLoader()
        df, metadata = loader.load(csv_dataset)

        # Register in registry
        loader.register_with_registry(metadata, registry=registry)

        # Verify it's in the registry
        retrieved = registry.get(metadata.dataset_id)
        assert retrieved is not None
        assert retrieved.filename == "iris.csv"
        assert retrieved.shape == (5, 4)
        assert retrieved.checksum == metadata.checksum

    def test_load_analyze_register(
        self,
        classification_dataset: Path,
        registry: DatasetRegistry,
    ):
        """Full pipeline: load, analyze, fingerprint, register."""
        loader = DatasetLoader()
        df, metadata = loader.load(classification_dataset)

        analyzer = DatasetAnalyzer(generate_fingerprint=True)
        summary = analyzer.analyze(df, source_filename=classification_dataset.name)

        # Register with fingerprint in extra_metadata
        fp_dict = summary.fingerprint.to_dict() if summary.fingerprint else {}
        loader.register_with_registry(
            metadata,
            registry=registry,
            fingerprint=fp_dict,
            file_path=str(classification_dataset),
        )

        # Verify registry entry
        retrieved = registry.get(metadata.dataset_id)
        assert retrieved is not None
        assert retrieved.filename == "binary_classification.csv"

        # Verify fingerprint stored in extra_metadata
        extra = retrieved.extra_metadata or {}
        assert "fingerprint" in extra
        assert extra["fingerprint"]["is_complete"] is True
        assert "file_path" in extra

    def test_load_json_and_register(self, tmp_path: Path, registry: DatasetRegistry):
        """Load a JSON dataset and register it."""
        import json

        path = tmp_path / "data.json"
        data = [{"id": i, "val": i * 10} for i in range(5)]
        path.write_text(json.dumps(data))

        loader = DatasetLoader()
        df, metadata = loader.load(path)
        loader.register_with_registry(metadata, registry=registry)

        retrieved = registry.get(metadata.dataset_id)
        assert retrieved is not None
        assert retrieved.source_format == ".json"
        assert retrieved.shape == (5, 2)

    def test_load_excel_and_register(self, tmp_path: Path, registry: DatasetRegistry):
        """Load an Excel dataset and register it."""
        path = tmp_path / "data.xlsx"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_excel(path, index=False)

        loader = DatasetLoader()
        df, metadata = loader.load(path)
        loader.register_with_registry(metadata, registry=registry)

        retrieved = registry.get(metadata.dataset_id)
        assert retrieved is not None
        assert retrieved.source_format == ".xlsx"
        assert retrieved.num_rows == 2

    def test_multiple_datasets_in_registry(
        self,
        csv_dataset: Path,
        classification_dataset: Path,
        registry: DatasetRegistry,
    ):
        """Multiple datasets can be registered and listed."""
        loader = DatasetLoader()

        # Load and register first dataset
        _, meta1 = loader.load(csv_dataset)
        loader.register_with_registry(meta1, registry=registry)

        # Load and register second dataset
        _, meta2 = loader.load(classification_dataset)
        loader.register_with_registry(meta2, registry=registry)

        # Both should appear in list_all
        all_ds = registry.list_all()
        assert len(all_ds) == 2
        filenames = [d.filename for d in all_ds]
        assert "iris.csv" in filenames
        assert "binary_classification.csv" in filenames

    def test_checksum_consistency(self, tmp_path: Path):
        """Same file loaded twice should produce same checksum."""
        path = tmp_path / "repeat.csv"
        pd.DataFrame({"x": [1, 2, 3]}).to_csv(path, index=False)

        loader = DatasetLoader()
        _, meta1 = loader.load(path)
        _, meta2 = loader.load(path)
        assert meta1.checksum == meta2.checksum

    def test_fingerprint_deterministic(self, csv_dataset: Path):
        """Same file should produce same fingerprint hashes."""
        loader = DatasetLoader()
        df1, _ = loader.load(csv_dataset)
        df2, _ = loader.load(csv_dataset)

        fp_gen = FingerprintGenerator()
        fp1 = fp_gen.from_dataframe(df1)
        fp2 = fp_gen.from_dataframe(df2)

        assert fp1.schema_hash == fp2.schema_hash
        assert fp1.column_hash == fp2.column_hash
        assert fp1.content_hash == fp2.content_hash
        assert fp1.shape_hash == fp2.shape_hash

    def test_analyzer_target_detection_integration(self, classification_dataset: Path):
        """Analyzer should detect classification target in pipeline."""
        loader = DatasetLoader()
        df, _ = loader.load(classification_dataset)

        analyzer = DatasetAnalyzer()
        summary = analyzer.analyze(df)

        assert summary.dataset_type["type"] == "classification"
        assert "target" in [t["name"] for t in summary.potential_targets]

    def test_register_with_backup(
        self,
        csv_dataset: Path,
        registry: DatasetRegistry,
    ):
        """Register should persist across registry instances (same DB)."""
        loader = DatasetLoader()
        _, metadata = loader.load(csv_dataset)
        loader.register_with_registry(metadata, registry=registry)

        # Create a new registry pointing to same DB
        db_path = registry._db_path
        registry2 = DatasetRegistry(db_path=db_path)
        retrieved = registry2.get(metadata.dataset_id)
        assert retrieved is not None
        assert retrieved.filename == "iris.csv"
        registry2.close()

    def test_registry_count_after_ingestion(
        self,
        csv_dataset: Path,
        classification_dataset: Path,
        registry: DatasetRegistry,
    ):
        """Registry count should reflect ingested datasets."""
        loader = DatasetLoader()

        assert registry.count() == 0

        _, meta1 = loader.load(csv_dataset)
        loader.register_with_registry(meta1, registry=registry)
        assert registry.count() == 1

        _, meta2 = loader.load(classification_dataset)
        loader.register_with_registry(meta2, registry=registry)
        assert registry.count() == 2

    def test_fingerprint_to_dict_in_extra_metadata(self, csv_dataset: Path):
        """Fingerprint to_dict output should be JSON-serializable."""
        loader = DatasetLoader()
        df, metadata = loader.load(csv_dataset)

        analyzer = DatasetAnalyzer(generate_fingerprint=True)
        summary = analyzer.analyze(df)

        fp_dict = summary.fingerprint.to_dict()
        # Verify it's JSON-serializable
        json_str = json.dumps(fp_dict)
        assert isinstance(json_str, str)

        # Verify key fields
        assert "fingerprint_id" in fp_dict
        assert "schema_hash" in fp_dict
        assert "content_hash" in fp_dict

    def test_loader_with_custom_kwargs(self, tmp_path: Path):
        """DatasetLoader should accept custom kwargs."""
        path = tmp_path / "custom.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(path, index=False)

        loader = DatasetLoader(dtype={"a": "float64"})
        df, _ = loader.load(path)
        assert df["a"].dtype == "float64"


class TestIngestionErrorHandling:
    """Error handling across the pipeline."""

    def test_load_nonexistent_then_analyze_fails_gracefully(self, tmp_path: Path):
        """Loading a nonexistent file should raise before analysis."""
        loader = DatasetLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "does_not_exist.csv")

    def test_load_corrupted_then_no_analysis(self, tmp_path: Path):
        """Corrupted file should raise CorruptedDatasetError."""
        path = tmp_path / "corrupt.csv"
        # Unterminated quote causes ParserError from the C parser
        path.write_text('a,b\n"unterminated,2\n3,4\n')

        loader = DatasetLoader()
        with pytest.raises(CorruptedDatasetError):
            loader.load(path)

    def test_empty_dataset_analyzer(self):
        """Analyzer should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        analyzer = DatasetAnalyzer()
        summary = analyzer.analyze(df)
        assert summary.general["num_rows"] == 0
        assert summary.general["num_columns"] == 0
        assert summary.fingerprint is not None  # Should still generate fingerprint
