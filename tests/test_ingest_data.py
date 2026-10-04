import zipfile

import pandas as pd
import pytest

from src.ingest_data import DataIngestorFactory, ZipDataIngestor


def _make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_zip_ingestor_reads_single_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "data.zip"
    _make_zip(archive, {"houses.csv": "a,b\n1,2\n3,4\n"})

    df = ZipDataIngestor().ingest(str(archive))

    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_zip_ingestor_rejects_non_zip(tmp_path):
    with pytest.raises(ValueError):
        ZipDataIngestor().ingest(str(tmp_path / "data.csv"))


def test_zip_ingestor_without_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive = tmp_path / "data.zip"
    _make_zip(archive, {"readme.txt": "no data"})

    with pytest.raises(FileNotFoundError):
        ZipDataIngestor().ingest(str(archive))


def test_factory():
    assert isinstance(DataIngestorFactory.get_data_ingestor(".zip"), ZipDataIngestor)
    with pytest.raises(ValueError):
        DataIngestorFactory.get_data_ingestor(".json")


def test_bundled_dataset_shape(tmp_path, monkeypatch):
    archive = __import__("os").path.abspath("data/archive.zip")
    monkeypatch.chdir(tmp_path)
    df = ZipDataIngestor().ingest(archive)
    assert df.shape == (2930, 82)
    assert "SalePrice" in df.columns
    assert isinstance(df, pd.DataFrame)
