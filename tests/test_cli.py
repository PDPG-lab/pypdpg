"""End-to-end CLI flow: keygen -> encrypt -> inspect -> decrypt."""

import csv

import numpy as np
import pytest

from pypdpg.cli import main

rng = np.random.default_rng(9)
DATA = rng.normal(50, 10, size=(20, 3)).round(3)
COLUMNS = ["income", "debt", "age"]


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    root = tmp_path_factory.mktemp("cli")
    main(["keygen", "-o", str(root / "keys")])
    csv_path = root / "applicants.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(DATA.tolist())
    main(
        [
            "encrypt",
            str(csv_path),
            "-c",
            str(root / "keys" / "controller.key"),
            "-o",
            str(root / "data.enc"),
        ]
    )
    return root


def test_keygen_wrote_both_files(workdir):
    assert (workdir / "keys" / "controller.key").exists()
    assert (workdir / "keys" / "processor.ctx").exists()


def test_inspect_enc(workdir, capsys):
    main(["inspect", str(workdir / "data.enc")])
    out = capsys.readouterr().out
    assert "(20, 3)" in out
    assert "income, debt, age" in out
    assert "nothing to show" in out


def test_inspect_context_files(workdir, capsys):
    main(["inspect", str(workdir / "keys" / "controller.key")])
    assert "do not ship" in capsys.readouterr().out
    main(["inspect", str(workdir / "keys" / "processor.ctx")])
    assert "safe to ship" in capsys.readouterr().out


def test_decrypt_roundtrip(workdir):
    out_csv = workdir / "back.csv"
    main(
        [
            "decrypt",
            str(workdir / "data.enc"),
            "-c",
            str(workdir / "keys" / "controller.key"),
            "-o",
            str(out_csv),
        ]
    )
    with open(out_csv, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == COLUMNS
    values = np.array([[float(cell) for cell in row] for row in rows[1:]])
    assert np.allclose(values, DATA, atol=1e-2)


def test_decrypt_with_public_context_refused(workdir, capsys):
    with pytest.raises(SystemExit):
        main(
            [
                "decrypt",
                str(workdir / "data.enc"),
                "-c",
                str(workdir / "keys" / "processor.ctx"),
            ]
        )
    assert "Why:" in capsys.readouterr().err


def test_inspect_rejects_random_file(workdir):
    other = workdir / "not_ours.bin"
    other.write_bytes(b"whatever")
    with pytest.raises(SystemExit, match="not a pypdpg file"):
        main(["inspect", str(other)])
