from pathlib import Path

import pytest

import freshqa.__main__ as cli


def test_benchmark_rejects_historical_report_output_before_credentials_or_inference(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid output must stop before credentials or inference")

    monkeypatch.setattr(cli, "load_dotenv", forbidden)
    monkeypatch.setattr(cli, "get_client", forbidden)
    monkeypatch.setattr(cli, "benchmark", forbidden)
    canonical = Path(cli.__file__).resolve().parents[1] / "reports"
    assert cli.main(["benchmark", "--out", str(canonical)]) == 2


def test_report_defaults_to_separate_output_without_credentials(monkeypatch):
    outputs = []
    monkeypatch.setattr(cli, "load_rows", lambda path: [])
    monkeypatch.setattr(cli, "validate_rows", lambda rows: [])
    monkeypatch.setattr(cli, "render_report", lambda rows, traces, out: outputs.append(out))

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline reporting must not read credentials")

    monkeypatch.setattr(cli, "load_dotenv", forbidden)
    assert cli.main(["report"]) == 0
    assert outputs == ["tmp/current-report"]


def test_benchmark_defaults_to_separate_current_report(monkeypatch):
    outputs, calls = [], []
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(cli, "load_rows", lambda path: [])
    monkeypatch.setattr(cli, "validate_rows", lambda rows: [])
    monkeypatch.setattr(cli, "get_client", lambda budget: object())
    monkeypatch.setattr(cli, "benchmark", lambda rows, **kwargs: calls.append(kwargs) or True)
    monkeypatch.setattr(cli, "render_report", lambda rows, traces, out: outputs.append(out))
    assert cli.main(["benchmark"]) == 0
    assert len(calls) == 1 and outputs == ["tmp/current-report"]


def test_smoke_still_requires_an_explicit_separate_report_output(monkeypatch):
    monkeypatch.setattr(cli, "load_dotenv", lambda *args: None)
    monkeypatch.setattr(cli, "load_rows", lambda path: [{}])
    monkeypatch.setattr(cli, "validate_rows", lambda rows: [])

    def forbidden(*args, **kwargs):
        raise AssertionError("Smoke output selection must be checked before inference")

    monkeypatch.setattr(cli, "get_client", forbidden)
    assert cli.main(["benchmark", "--limit", "1", "--traces", "tmp/separate-smoke"]) == 2


@pytest.mark.parametrize("command", ["report", "benchmark"])
@pytest.mark.parametrize("location", ["artifact", "temporary", "output", "ancestor"])
def test_cli_rejects_output_symlinks_before_credentials_or_inference(
        tmp_path, monkeypatch, capsys, command, location):
    target = tmp_path / "historical"
    target.mkdir()
    sentinel = target / "report.md"
    original = b"accepted historical report\n"
    sentinel.write_bytes(original)
    out = tmp_path / "current"
    if location in {"artifact", "temporary"}:
        out.mkdir()
        name = "report.md" if location == "artifact" else "grading.tmp"
        (out / name).symlink_to(sentinel)
    else:
        out.symlink_to(target, target_is_directory=True)
        if location == "ancestor":
            out = out / "not-created" / "output"

    def forbidden(*args, **kwargs):
        raise AssertionError("Output validation must precede credentials, input reads and inference")

    for name in ("load_dotenv", "get_client", "benchmark", "load_rows", "render_report"):
        monkeypatch.setattr(cli, name, forbidden)
    assert cli.main([command, "--out", str(out)]) == 2
    assert "символическ" in capsys.readouterr().err
    assert sentinel.read_bytes() == original
    assert not (target / "not-created").exists()
