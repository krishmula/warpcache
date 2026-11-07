import json
from pathlib import Path

from scripts.figures import FIGURES, load_cases, main

BENCH = Path(__file__).resolve().parents[1] / "results" / "bench"


def test_main_writes_every_figure(tmp_path):
    main(["--bench", str(BENCH), "--out", str(tmp_path)])
    for name, _draw in FIGURES:
        for ext in ("pdf", "png"):
            path = tmp_path / f"{name}.{ext}"
            assert path.stat().st_size > 0, path
    assert len(list(tmp_path.iterdir())) == 10


def test_newest_file_wins_each_case_key(tmp_path):
    line = {"kernel": "path_a", "dtype": "fp16", "batch": 1, "history": 4096, "chunks": 1}
    # The newer file is written first, so creation order cannot pick the winner.
    (tmp_path / "ubuntu-20261107-020000.jsonl").write_text(json.dumps({**line, "median_us": 90.0}) + "\n")
    (tmp_path / "ubuntu-20261107-010000.jsonl").write_text(json.dumps({**line, "median_us": 100.0}) + "\n")
    cases = load_cases(tmp_path)
    assert cases[("path_a", "fp16", 1, 4096, 1)]["median_us"] == 90.0
