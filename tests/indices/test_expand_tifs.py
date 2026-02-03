from pathlib import Path

from shelterbelts.indices import expand_tifs as expand_module
from shelterbelts.indices.expand_tifs import expand_tifs


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("test")


def test_expand_tifs_filters_and_calls(monkeypatch, tmp_path):
    """expand_tifs should only process *.tif files and skip merged files."""
    folder_to_expand = tmp_path / "tiles"
    folder_merged = tmp_path / "merged"
    outdir = tmp_path / "out"

    _touch(folder_to_expand / "a.tif")
    _touch(folder_to_expand / "b.tif")
    _touch(folder_to_expand / "c_merged.tif")
    (folder_to_expand / "uint8_predicted").mkdir(parents=True)

    calls = []

    def fake_expand_tif(filename, folder_merged_arg, sub_outdir, gpkg):
        calls.append((Path(filename).name, Path(sub_outdir)))

    monkeypatch.setattr(expand_module, "expand_tif", fake_expand_tif)

    expand_tifs(
        folder_to_expand=str(folder_to_expand),
        folder_merged=str(folder_merged),
        outdir=str(outdir),
        limit=None,
        gpkg="fake.gpkg",
    )

    processed = {name for name, _ in calls}
    assert processed == {"a.tif", "b.tif"}
    assert (outdir / folder_to_expand.stem).exists()


def test_expand_tifs_respects_limit(monkeypatch, tmp_path):
    """expand_tifs should respect the limit parameter."""
    folder_to_expand = tmp_path / "tiles"
    folder_merged = tmp_path / "merged"
    outdir = tmp_path / "out"

    _touch(folder_to_expand / "a.tif")
    _touch(folder_to_expand / "b.tif")
    _touch(folder_to_expand / "c.tif")

    calls = []

    def fake_expand_tif(filename, folder_merged_arg, sub_outdir, gpkg):
        calls.append(Path(filename).name)

    monkeypatch.setattr(expand_module, "expand_tif", fake_expand_tif)

    expand_tifs(
        folder_to_expand=str(folder_to_expand),
        folder_merged=str(folder_merged),
        outdir=str(outdir),
        limit=2,
        gpkg="fake.gpkg",
    )

    assert len(calls) == 2
