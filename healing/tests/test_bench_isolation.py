"""Every `hex` call runs with its own data folder, never the real Hex's."""

import json
import os
import stat

from healing.eval import bench


def test_benchmark_runs_hex_in_an_isolated_data_folder(tmp_path, monkeypatch):
    home = tmp_path / "healing-home"
    seen = tmp_path / "env.json"
    fake = tmp_path / "hex"
    fake.write_text(
        "#!/bin/sh\n"
        f"python3 -c 'import json,os; json.dump(dict(os.environ), open(\"{seen}\", \"w\"))'\n"
        "echo '{\"type\":\"model\",\"load_ms\":1}'\n"
        "echo '{\"type\":\"summary\",\"clips\":0,\"wer\":0,\"word_errors\":0,"
        "\"reference_words\":0,\"median_rtf\":0,\"quality\":{},\"groups\":{}}'\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("HEALING_HOME", str(home))
    monkeypatch.setenv("HEX_BIN", str(fake))
    monkeypatch.delenv("HEX_APPLICATION_SUPPORT_DIR", raising=False)

    bench.run_benchmark(tmp_path / "manifest.json")

    env = json.loads(seen.read_text())
    support = env["HEX_APPLICATION_SUPPORT_DIR"]
    assert os.path.commonpath([support, str(home)]) == str(home)
    assert "Library/Application Support/hex/" not in support + "/"
    assert env["HEX_DISABLE_UPDATES"] == "1"
