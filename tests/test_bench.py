from scripts.bench import bytes_per_token, compare, main


def line(kernel, median, chunks=None):
    return {"kernel": kernel, "dtype": "fp16", "batch": 1, "history": 4096,
            "chunks": chunks, "median_us": median}


def test_bytes_per_token_match_the_cache_layout():
    # 8 KV heads x 128 values, K and V. INT8 adds one FP32 scale per row.
    assert bytes_per_token("fp16") == 4096
    assert bytes_per_token("int8") == 2048 + 64


def test_compare_matches_cases_by_key():
    a = [line("path_a", 100.0), line("path_b", 50.0, chunks=4), line("sdpa", 80.0)]
    b = [line("path_b", 51.0, chunks=4), line("path_a", 104.0), line("path_b", 9.0, chunks=8)]
    got = dict(compare(a, b))
    assert got.keys() == {("path_a", "fp16", 1, 4096, None), ("path_b", "fp16", 1, 4096, 4)}
    assert abs(got[("path_a", "fp16", 1, 4096, None)] - 0.04) < 1e-12
    assert abs(got[("path_b", "fp16", 1, 4096, 4)] - 0.02) < 1e-12


def test_compare_exit_code_flags_more_than_3_percent(tmp_path):
    import json
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    a.write_text(json.dumps(line("path_a", 100.0)) + "\n")
    b.write_text(json.dumps(line("path_a", 102.0)) + "\n")
    c.write_text(json.dumps(line("path_a", 104.0)) + "\n")
    assert main(["--compare", str(a), str(b)]) == 0
    assert main(["--compare", str(a), str(c)]) == 1
