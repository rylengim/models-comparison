import json
import multiprocessing
import time
from types import SimpleNamespace

import pytest
import requests
from urllib3.exceptions import ConnectTimeoutError, MaxRetryError, NewConnectionError, ReadTimeoutError

from freshqa.client import CostUnknownError, OpenRouterError
from freshqa.agent import run_agent
from reports.run_paced import RequestGate


def _paced_worker(path, interval, ready, start, output):
    gate = RequestGate(path, interval)
    ready.put(True)
    if not start.wait(5):
        raise RuntimeError("Test start timed out")
    for _ in range(2):
        gate.call(lambda: output.put(time.monotonic()))


def _unknown_worker(path, interval, started, fail, finished, output):
    def completion():
        started.set()
        if not fail.wait(5):
            raise RuntimeError("Test failure signal timed out")
        raise CostUnknownError("Synthetic uncertain charge")

    try:
        RequestGate(path, interval).call(completion)
    except CostUnknownError:
        output.put("unknown")
        finished.set()


def _waiting_worker(path, interval, ready, start, waiting, output):
    import reports.run_paced as paced

    original_sleep = time.sleep

    def observed_sleep(seconds):
        waiting.set()
        original_sleep(seconds)

    paced.time.sleep = observed_sleep
    ready.set()
    if not start.wait(5):
        raise RuntimeError("Test start timed out")
    try:
        RequestGate(path, interval).call(lambda: output.put("POST sent"))
    except OpenRouterError as error:
        output.put(str(error))


def _finish(processes):
    for process in processes:
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(2)
            pytest.fail("Gate worker did not finish")
        assert process.exitcode == 0


def test_interval_is_shared_across_independent_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, output, start = context.Queue(), context.Queue(), context.Event()
    interval = .08
    processes = [context.Process(target=_paced_worker,
                 args=(str(tmp_path / "requests.timestamp"), interval, ready, start, output))
                 for _ in range(3)]
    for process in processes:
        process.start()
    for _ in processes:
        assert ready.get(timeout=5) is True
    start.set()
    _finish(processes)
    starts = sorted(output.get(timeout=2) for _ in range(6))
    assert all(later - earlier >= interval - .015 for earlier, later in zip(starts, starts[1:]))


def test_unknown_charge_blocks_a_waiting_process_and_future_starts(tmp_path):
    context = multiprocessing.get_context("spawn")
    path, interval = str(tmp_path / "requests.timestamp"), .5
    started, fail, finished, waiting = (context.Event() for _ in range(4))
    ready, start = context.Event(), context.Event()
    output = context.Queue()
    unknown = context.Process(target=_unknown_worker,
              args=(path, interval, started, fail, finished, output))
    blocked = context.Process(target=_waiting_worker,
              args=(path, interval, ready, start, waiting, output))
    blocked.start()
    assert ready.wait(5)
    unknown.start()
    assert started.wait(5)
    start.set()
    assert waiting.wait(5)
    fail.set()
    assert finished.wait(2)  # Marker creation must not wait for the sleeping gate lock.
    _finish([unknown, blocked])
    outcomes = [output.get(timeout=2) for _ in range(2)]
    assert "unknown" in outcomes
    assert any("BLOCKED" in value for value in outcomes)
    assert "POST sent" not in outcomes
    gate = RequestGate(path, interval)
    assert gate.blocked_path.exists()
    with pytest.raises(OpenRouterError, match="BLOCKED"):
        gate.call(lambda: pytest.fail("Blocked gate sent another request"))
    assert gate.blocked_path.exists()


@pytest.mark.parametrize("interval", ["0", "-1", "nan", "inf", "invalid"])
def test_invalid_environment_interval_fails_before_requests(monkeypatch, tmp_path, interval):
    monkeypatch.setenv("FRESHQA_GATE_PATH", str(tmp_path / "gate"))
    monkeypatch.setenv("FRESHQA_GATE_INTERVAL_S", interval)
    with pytest.raises(ValueError, match="FRESHQA_GATE_INTERVAL_S"):
        RequestGate.from_env()


@pytest.mark.parametrize("path", ["", "   "])
def test_empty_environment_path_is_rejected(monkeypatch, path):
    monkeypatch.setenv("FRESHQA_GATE_PATH", path)
    with pytest.raises(ValueError, match="FRESHQA_GATE_PATH"):
        RequestGate.from_env()


def test_environment_settings_and_invalid_existing_state_fail_closed(monkeypatch, tmp_path):
    path = tmp_path / "gate"
    monkeypatch.setenv("FRESHQA_GATE_PATH", str(path))
    monkeypatch.setenv("FRESHQA_GATE_INTERVAL_S", "0.02")
    gate = RequestGate.from_env()
    assert gate.path == path
    assert gate.interval_s == .02
    path.write_text("not a timestamp")
    with pytest.raises(OpenRouterError, match="state"):
        gate.call(lambda: pytest.fail("Invalid gate state sent a request"))


def test_blocked_worker_records_known_zero_cost_without_sending_a_request(tmp_path):
    gate = RequestGate(tmp_path / "gate", .01)
    gate.blocked_path.touch()
    client = SimpleNamespace(spent_usd=0.0, accounting_uncertain=False)
    client.complete = lambda **kwargs: gate.call(lambda: pytest.fail("Unexpected POST"))
    result = run_agent("Q", model="test", client=client, registry=None, tool_names=[])
    assert result["status"] == "error"
    assert result["cost_usd"] == result["known_cost_usd"] == 0
    assert result["events"][0]["cost_usd"] == 0
    assert "BLOCKED" in result["events"][0]["error"]


def test_wrapper_forwards_payload_and_restores_client_method(monkeypatch, tmp_path):
    import reports.run_paced as paced

    monkeypatch.setenv("FRESHQA_GATE_PATH", str(tmp_path / "gate"))
    monkeypatch.setenv("FRESHQA_GATE_INTERVAL_S", ".01")
    arguments = {"model": "test", "messages": [{"role": "user", "content": "Q"}], "tools": None}
    observed = []

    def completion(client, **kwargs):
        observed.append(kwargs)
        return "unchanged response"

    def cli(argv):
        assert argv == ["synthetic"]
        assert paced.OpenRouterClient.complete(object(), **arguments) == "unchanged response"
        return 0

    monkeypatch.setattr(paced.OpenRouterClient, "complete", completion)
    monkeypatch.setattr(paced, "cli_main", cli)
    assert paced.main(["synthetic"]) == 0
    assert observed == [arguments]
    assert paced.OpenRouterClient.complete is completion


def test_directory_environment_path_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("FRESHQA_GATE_PATH", str(tmp_path))
    with pytest.raises(ValueError, match="FRESHQA_GATE_PATH"):
        RequestGate.from_env()


@pytest.mark.parametrize("error_class,reason", [
    (requests.ConnectTimeout, ConnectTimeoutError("secret-connect-message")),
    (requests.ReadTimeout, ReadTimeoutError(None, "https://secret-url.invalid", "secret-read-message")),
    (requests.ConnectionError, NewConnectionError(None, "secret-connection-message")),
])
def test_transport_diagnostic_reports_only_types_and_restores_patches(
        monkeypatch, tmp_path, capsys, error_class, reason):
    import reports.run_paced as paced

    monkeypatch.setenv("FRESHQA_GATE_PATH", str(tmp_path / "gate"))
    monkeypatch.setenv("FRESHQA_GATE_INTERVAL_S", ".01")
    failure = error_class("secret-outer-message")
    wrapped = MaxRetryError(None, "https://secret-url.invalid", reason=reason)
    calls = []
    arguments = {"headers": {"Authorization": "Bearer secret-header"},
                 "json": {"message": "secret-payload"}, "timeout": (10, 90),
                 "allow_redirects": False}

    def post(*args, **kwargs):
        calls.append((args, kwargs))
        try:
            raise wrapped
        except MaxRetryError:
            raise failure

    def cli(argv):
        return requests.post("https://secret-url.invalid", **arguments)

    original_complete = paced.OpenRouterClient.complete
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(paced, "cli_main", cli)
    with pytest.raises(error_class) as caught:
        paced.main(["synthetic"])
    assert caught.value is failure
    assert calls == [(("https://secret-url.invalid",), arguments)]
    captured = capsys.readouterr()
    assert captured.out == ""
    diagnostic, stack_line = captured.err.splitlines()
    assert diagnostic.startswith(f"POST transport failure: type={error_class.__name__} "
                                 f"reason_type={type(reason).__name__} elapsed_s=")
    assert float(diagnostic.rsplit("=", 1)[1]) >= 0
    stack = json.loads(stack_line.removeprefix("POST exception chain: "))
    assert stack[0]["type"] == error_class.__name__
    assert all(set(frame) == {"module", "function"} for item in stack for frame in item["frames"])
    assert "secret" not in captured.err
    assert requests.post is post
    assert paced.OpenRouterClient.complete is original_complete


def test_transport_trace_identifies_handshake_context_and_post_elapsed(monkeypatch, tmp_path, capsys):
    import reports.run_paced as paced

    monkeypatch.setenv("FRESHQA_GATE_PATH", str(tmp_path / "gate"))
    ticks = iter([20.0, 20.125])
    monkeypatch.setattr(paced.time, "monotonic", lambda: next(ticks))
    failure = requests.ReadTimeout("secret-request-message")
    calls = []

    def synthetic_handshake():
        local_token = "secret-local-variable"
        raise TimeoutError(local_token)

    def validate_connection():
        try:
            synthetic_handshake()
        except TimeoutError:
            raise ReadTimeoutError(None, "https://secret-url.invalid", "secret-tls-message")

    def post(*args, **kwargs):
        calls.append(1)
        try:
            validate_connection()
        except ReadTimeoutError:
            raise failure

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(paced, "cli_main", lambda argv: requests.post("https://secret-url.invalid"))
    original_complete = paced.OpenRouterClient.complete
    with pytest.raises(requests.ReadTimeout) as caught:
        paced.main([])
    assert caught.value is failure
    assert calls == [1]
    diagnostic = capsys.readouterr().err
    line, stack_line = diagnostic.splitlines()
    assert line.endswith("elapsed_s=0.125000")
    stack = json.loads(stack_line.removeprefix("POST exception chain: "))
    assert [item["type"] for item in stack] == ["ReadTimeout", "ReadTimeoutError", "TimeoutError"]
    assert stack[-1]["frames"][-1] == {"module": __name__, "function": "synthetic_handshake"}
    assert all(len(item["frames"]) <= 3 for item in stack)
    assert "secret" not in diagnostic
    assert requests.post is post
    assert paced.OpenRouterClient.complete is original_complete
