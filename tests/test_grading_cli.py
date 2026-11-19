from pathlib import Path

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
