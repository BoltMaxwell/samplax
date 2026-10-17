import pathlib, re, time

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)

def test_readme_has_examples():
    assert len(BLOCKS) >= 3

def test_each_block_runs_standalone():
    for i, src in enumerate(BLOCKS):
        t0 = time.time()
        exec(compile(src, f"README block {i}", "exec"), {"__name__": "__readme__"})
        assert time.time() - t0 < 30, f"block {i} too slow"
