import os
import subprocess

import dotenv
import pytest

from freshqa import __main__ as freshqa_cli
from hw4 import __main__ as launcher


def test_one_command_defaults_to_fresh_152_without_reading_dotenv(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=unused-stub-secret\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def unexpectedly_read_dotenv(*args, **kwargs):
        pytest.fail("The stub launch must not read .env")

    monkeypatch.setattr(dotenv, "dotenv_values", unexpectedly_read_dotenv)
    monkeypatch.setattr(dotenv, "load_dotenv", unexpectedly_read_dotenv)

    def ask(argv):
        assert argv[0:2] == ["ask", "--mcp"]
        assert "--openrouter" not in argv
        assert "OPENROUTER_API_KEY" not in os.environ
        assert argv[-1] == (
            "What official designation did ESA report on 19 October 2026 "
            "for the first previously unknown comet discovered by Euclid?"
        )
        return 0

    monkeypatch.setattr(freshqa_cli, "main", ask)
    assert launcher._run_ask([]) == 0


def test_openrouter_in_positional_question_does_not_enable_paid_mode(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=unused-stub-secret\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def ask(argv):
        assert "OPENROUTER_API_KEY" not in os.environ
        assert argv == ["ask", "--mcp", "--", "--openrouter"]
        return 0

    monkeypatch.setattr(freshqa_cli, "main", ask)
    assert launcher._run_ask(["--", "--openrouter"]) == 0


@pytest.mark.parametrize("env_key", [None, "environment-key"])
def test_paid_launch_prefers_env_and_only_forwards_openrouter_key(monkeypatch, tmp_path, env_key):
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.delenv("UNRELATED_TEST_TOKEN", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", env_key or "unused-test-key")
    if env_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY")
    else:
        def unexpectedly_read_dotenv(*args, **kwargs):
            pytest.fail("An exported key must not require reading .env")

        monkeypatch.setattr(dotenv, "dotenv_values", unexpectedly_read_dotenv)
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=local-override\nUNRELATED_TEST_TOKEN=do-not-load\n",
        encoding="utf-8",
    )
    options = ["--openrouter", "--budget-usd", "0.06", "What did Euclid discover?"]

    def ask(argv):
        assert argv == ["ask", "--mcp", *options]
        assert os.environ["OPENROUTER_API_KEY"] == (env_key or "local-override")
        assert "UNRELATED_TEST_TOKEN" not in os.environ
        return 0

    monkeypatch.setattr(freshqa_cli, "main", ask)
    assert launcher._run_ask(options) == 0


@pytest.mark.parametrize("offline", [False, True])
def test_bootstrap_prepares_client_and_local_image_before_running(
    monkeypatch, tmp_path, offline,
):
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "bootstrap-must-not-use-this")
    monkeypatch.delenv("FRESHQA_WHEELHOUSE", raising=False)
    wheelhouse = tmp_path / "wheel house"
    if offline:
        wheelhouse.mkdir()
        monkeypatch.setenv("FRESHQA_WHEELHOUSE", str(wheelhouse))

    calls = []

    def run(argv, **kwargs):
        calls.append([os.fspath(arg) for arg in argv])
        if len(calls) <= 3:
            assert "OPENROUTER_API_KEY" not in kwargs["env"]
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    assert launcher.main(["--", "What did Euclid discover?"]) == 0

    venv, pip, build, ask = calls
    assert venv[1:3] == ["-m", "venv"]
    assert venv[-1].endswith("/tmp/hw4-mcp-venv")
    assert pip[1:3] == ["-m", "pip"]
    assert "install" in pip
    assert "requirements.txt" in " ".join(pip)
    assert build[0:3] == ["docker", "--host", "unix:///var/run/docker.sock"]
    assert "build" in build
    assert ask[0].endswith("/tmp/hw4-mcp-venv/bin/python")
    assert ask[-2:] == ["--", "What did Euclid discover?"]
    if offline:
        assert "--no-index" in pip and str(wheelhouse) in " ".join(pip)
        assert f"wheels={wheelhouse}" in build and "FRESHQA_MCP_OFFLINE=1" in build
    else:
        assert "--no-index" not in pip
        assert "FRESHQA_MCP_OFFLINE=1" not in build


def test_bootstrap_failure_does_not_launch_agent(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(launcher, "REPO", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FRESHQA_WHEELHOUSE", raising=False)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) != 1:
            pytest.fail("A failed virtual environment setup must stop the launcher")
        if kwargs.get("check"):
            raise subprocess.CalledProcessError(returncode=6, cmd=argv)
        return subprocess.CompletedProcess(argv, returncode=6)

    monkeypatch.setattr(subprocess, "run", run)
    assert launcher.main([]) != 0
    assert len(calls) == 1
    assert "Не удалось запустить" in capsys.readouterr().err
