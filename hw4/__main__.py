"""Prepare the local FreshQA environment and run one question through MCP."""

import fcntl
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
DEFAULT_QUESTION = (
    "What official designation did ESA report on 19 October 2026 for the first previously unknown comet discovered by Euclid?"
)


def _run_ask(argv=None):
    from freshqa.__main__ import main as ask

    options = list(sys.argv[1:] if argv is None else argv)
    if not options:
        options = ["--", DEFAULT_QUESTION]
    flags = options[:options.index("--")] if "--" in options else options
    if "--openrouter" in flags and not os.getenv("OPENROUTER_API_KEY", "").strip():
        from dotenv import dotenv_values

        path = REPO / ".env"
        if path.is_file():
            key = (dotenv_values(path, interpolate=False).get("OPENROUTER_API_KEY") or "").strip()
            if key:
                os.environ["OPENROUTER_API_KEY"] = key
    return ask(["ask", "--mcp", *options])


def main(argv=None):
    options = list(sys.argv[1:] if argv is None else argv)
    if options in (["--help"], ["-h"]):
        print("python3.12 -m hw4 [--openrouter --budget-usd USD] [QUESTION]")
        print("По умолчанию запускает контрольный вопрос на заглушке.")
        return 0

    venv = REPO / "tmp/hw4-mcp-venv"
    python = str(venv / "bin/python")
    pip = [python, "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--quiet"]
    docker = ["docker", "--host", "unix:///var/run/docker.sock", "build",
              "--builder", "default", "--tag", "freshqa-hw4-mcp:local", "--file", "hw4/Dockerfile"]
    offline = os.getenv("FRESHQA_WHEELHOUSE")
    if offline:
        wheels = Path(offline).expanduser().resolve()
        if not wheels.is_dir():
            print(f"Каталог пакетов FRESHQA_WHEELHOUSE не найден: {wheels}", file=sys.stderr)
            return 2
        pip.extend(["--no-index", "--find-links", str(wheels)])
        docker.extend(["--build-context", f"wheels={wheels}", "--build-arg", "FRESHQA_MCP_OFFLINE=1", "--network", "none"])

    bootstrap_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("OPENROUTER_API_KEY", "DOCKER_HOST", "DOCKER_CONTEXT", "BUILDX_BUILDER"):
        bootstrap_env.pop(name, None)
    try:
        venv.parent.mkdir(parents=True, exist_ok=True)
        with (venv.parent / "hw4-bootstrap.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run([sys.executable, "-m", "venv", str(venv)], cwd=REPO, env=bootstrap_env, check=True)
            subprocess.run([*pip, "-r", "hw4/requirements.txt"], cwd=REPO, env=bootstrap_env, check=True)
            subprocess.run([*docker, "."], cwd=REPO, env=bootstrap_env, check=True)
        child = "from hw4.__main__ import _run_ask; raise SystemExit(_run_ask())"
        return subprocess.run([python, "-c", child, *options], cwd=REPO, check=False).returncode
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"Не удалось запустить ДЗ 4. {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
