import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from io import _io
from time import sleep

import pytest

from appworld.common import system
from appworld.common.background_server import BackgroundServer
from appworld.common.io import read_file, safely_remove_file
from appworld.common.time import TimeoutError  # needs to be imported from here.


class TestCommonBackgroundServer:
    python_buffered: str

    @classmethod
    def setup_class(cls) -> None:
        cls.python_buffered = os.environ.get("PYTHONUNBUFFERED", "")
        os.environ["PYTHONUNBUFFERED"] = "1"

    @classmethod
    def teardown_class(cls) -> None:
        if cls.python_buffered:
            os.environ["PYTHONUNBUFFERED"] = cls.python_buffered

    def test_server_starts_and_port_is_open(self) -> None:
        server = BackgroundServer(sys.executable + " -m http.server {port}")
        with server:
            assert server.proc is not None
            assert server.proc.poll() is None  # Still running
            assert not system.is_port_available(server.port)

    def test_server_auto_port_assignment(self) -> None:
        server1 = BackgroundServer(sys.executable + " -m http.server {port}")
        server2 = BackgroundServer(sys.executable + " -m http.server {port}")
        with server1, server2:
            assert server1.port != server2.port

    def test_server_timeout_error(self) -> None:
        with pytest.raises(TimeoutError):
            BackgroundServer(
                f"sleep 5 && {sys.executable} -m http.server {{port}}", timeout=1
            )._wait_until_ready(timeout=1)

    def test_port_provided_but_not_free_raises_error(self) -> None:
        with BackgroundServer(sys.executable + " -m http.server {port}") as server:
            with pytest.raises(ValueError) as exception_info:
                with BackgroundServer(
                    f"{sys.executable} -m http.server {server.port}", port=server.port
                ):
                    pass
            assert str(exception_info.value) == (
                f"Port {server.port} is not available for '{server.command}'. "
                "Please provide a different port, ensure the port is free, "
                "or set 'force_start' to True."
            )

    def test_error_when_no_port_placeholder_and_no_port_passed(self) -> None:
        command = sys.executable + " -m http.server"
        with pytest.raises(ValueError) as exception_info:
            BackgroundServer(command, port=None)
        assert str(exception_info.value) == (
            "If the server is enabled and the command does not have a '{port}' or '{port:<name>}' "
            f"placeholder for all ports, the 'port' must be provided. Found command: '{command}', port: 'None'"
        )

    def test_process_killed_after_context_exit_and_logging_to_file_once(self) -> None:
        server = BackgroundServer(sys.executable + " -m http.server {port}")
        with server:
            assert server.proc.poll() is None  # running
        assert server.proc.poll() is not None  # terminated
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            log_path = tmpfile.name
            server = BackgroundServer(
                sys.executable + " -m http.server {port}", log_file_path=str(log_path)
            )
            with server:
                sleep(1)
            log_contents = read_file(log_path)
            assert ("Serving HTTP on" in log_contents) and (f"port {server.port}" in log_contents)

    def test_logging_to_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            log_path = tmpfile.name
            server = BackgroundServer(
                sys.executable + " -m http.server {port}", log_file_path=str(log_path)
            )
            with server:
                sleep(1)
            log_contents = read_file(log_path)
            assert ("Serving HTTP on" in log_contents) and (f"port {server.port}" in log_contents)

    def test_live_logging_to_stdout(self) -> None:
        captured_output = _io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            server = BackgroundServer(sys.executable + " -m http.server {port}", show_logs=True)
            with server:
                sleep(1)
        output = captured_output.getvalue()
        assert ("Serving HTTP on" in output) and (f"port {server.port}" in output)

    def test_logging_to_stdout_and_file(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            log_path = tmpfile.name
            captured_output = _io.StringIO()
            with contextlib.redirect_stdout(captured_output):
                server = BackgroundServer(
                    f"{sys.executable} -m http.server {{port}}",
                    log_file_path=str(log_path),
                    show_logs=True,
                )
                with server:
                    sleep(1)
            output = captured_output.getvalue()
            log_contents = read_file(log_path)
            assert ("Serving HTTP on" in output) and (f"port {server.port}" in output)
            assert ("Serving HTTP on" in log_contents) and (f"port {server.port}" in log_contents)

    def test_logging_suppressed_when_no_log_targets_are_set(self) -> None:
        captured_output = _io.StringIO()
        with contextlib.redirect_stdout(captured_output):
            server = BackgroundServer(sys.executable + " -m http.server {port}")
            with server:
                sleep(1)
        output = captured_output.getvalue()
        assert ("Serving HTTP on" not in output) and (f"port {server.port}" not in output)

    def test_force_start_kills_existing_process(self) -> None:
        server1 = BackgroundServer(sys.executable + " -m http.server {port}")
        with server1:
            assert server1.proc.poll() is None  # running
            server2 = BackgroundServer(
                f"{sys.executable} -m http.server {{port}}",
                port=server1.port,
                force_start=True,
            )
            with server2:
                assert server2.port == server1.port
                assert server2.proc.poll() is None  # running
                # server1 should have been terminated forcibly
                assert server1.proc.poll() is not None

    def test_started_requires_port_and_skips_launch(self) -> None:
        with pytest.raises(ValueError) as exception_info:
            BackgroundServer(sys.executable + " -m http.server {port}", started=True, port=None)
        assert str(exception_info.value) == (
            "If 'enabled' is True and 'started' is True, i.e., the server is already expected to be running, "
            "the 'port' must be provided."
        )
        # test started=True skips launching a new process
        port = system.get_available_port()
        # Manually start a server beforehand
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            proc = subprocess.Popen(
                [sys.executable, "-m", "http.server", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        else:
            proc = subprocess.Popen(
                ["python", "-m", "http.server", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        try:
            server = BackgroundServer(
                sys.executable + " -m http.server {port}", port=port, started=True
            )
            with server:
                # server.proc should be None since it assumes server is already running
                assert server.proc is None
                # _wait_until_ready should still check if port is listening
                assert not system.is_port_available(port)
                assert server.url == f"http://localhost:{port}"
        finally:
            if sys.platform == "win32":
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    with contextlib.suppress(Exception):
                        proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait()

    def test_enabled_false_is_noop(self) -> None:
        # test enabled=False is a no-op
        server = BackgroundServer(sys.executable + " -m http.server {port}", enabled=False)
        assert server.enabled is False
        assert server.url is None
        server.start()  # should do nothing
        assert server.proc is None
        with server:  # context manager should be a no-op
            assert server.proc is None
        server.stop()  # should do nothing
        assert server.proc is None

    def test_server_with_dict_ports(self) -> None:
        # test that server starts correctly when port is a dict
        command = (
            f"{sys.executable} -u -c "
            "'import threading as th, http.server as hs, sys; "
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", int(sys.argv[1])), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", int(sys.argv[2])), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            "th.Event().wait()' "
            "{port:api} {port:ws}"
        )
        ports = {"api": system.get_available_port(), "ws": system.get_available_port()}
        server = BackgroundServer(command, port=ports, show_logs=True)
        with server:
            assert isinstance(server.port, dict)
            # both named ports should be bound
            for p in server.port.values():
                assert not system.is_port_available(p)
            # url should map names to localhost URLs
            assert isinstance(server.url, dict)
            assert set(server.url.keys()) == {"api", "ws"}
            assert server.url["api"].startswith("http://localhost:")
            assert server.url["ws"].startswith("http://localhost:")

    def test_dict_ports_with_health_check(self) -> None:
        command = (
            f"{sys.executable} -u -c "
            "'import threading as th, http.server as hs, sys; "
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", int(sys.argv[1])), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", int(sys.argv[2])), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            "th.Event().wait()' "
            "{port:api} {port:ws}"
        )
        ports = {"api": system.get_available_port(), "ws": system.get_available_port()}
        health_paths = {"api": "/", "ws": "/status"}
        server = BackgroundServer(command, port=ports, health_check_at=health_paths, show_logs=True)
        with server:
            assert isinstance(server.port, dict)
            for p in server.port.values():
                assert not system.is_port_available(p)
            assert isinstance(server.url, dict)

    def test_named_placeholders_filled_in_command(self) -> None:
        # test that server ports are correctly filled in the command with named placeholders with dict ports
        command = (
            f"{sys.executable} -u -c "
            "'import threading as th, http.server as hs, sys; "
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", int(sys.argv[1])), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", int(sys.argv[2])), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            "th.Event().wait()' "
            "{port:api} {port:ws}"
        )
        server = BackgroundServer(command, port=None)
        with server:
            assert isinstance(server.port, dict)
            for name, port in server.port.items():
                assert f"{port}" in server.command
                assert f"{{port:{name}}}" not in server.command
                for p in server.port.values():
                    assert not system.is_port_available(p)

    def test_mismatched_dict_keys_raise(self) -> None:
        # test that if both port and health_check_at are dicts, their keys must match
        command = f"{sys.executable} -m http.server {{port:api}}"
        ports = {"api": 12345, "ws": 23456}
        health_paths = {"api": "/"}
        with pytest.raises(ValueError) as ei:
            BackgroundServer(command, port=ports, health_check_at=health_paths)
        assert str(ei.value) == (
            "If both 'port' and 'health_check_at' are dictionaries, "
            "they must have the same keys. "
            f"Found port keys: {list(ports.keys())}, "
            f"health_check_at keys: {list(health_paths.keys())}."
        )

    def test_timeout_error_includes_recent_logs(self) -> None:
        command = (
            f"{sys.executable} -u -c "
            "'import sys, time; "
            '[sys.stdout.write(f"BOOT {i}\\n") or sys.stdout.flush() or time.sleep(0.05) for i in range(5)]; '
            "time.sleep(2)' "
            "{port}"
        )
        server = BackgroundServer(command, timeout=1)
        with pytest.raises(TimeoutError) as exception:
            server.start()
        message = str(exception.value)
        assert "did not start on port" in message
        assert "BOOT 0" in message or "BOOT 1" in message
        timeout_idx = message.find("did not start on port")
        log_idx = message.find("BOOT 0") if "BOOT 0" in message else message.find("BOOT 1")
        assert timeout_idx != -1 and log_idx != -1 and timeout_idx < log_idx

    def test_logs_saved_to_file_on_timeout(self) -> None:
        # test that logs are saved to file on timeout
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            log_path = tmpfile.name
        try:
            command = (
                f"{sys.executable} -u -c "
                "'import sys, time; "
                '[sys.stdout.write(f"BOOT {i}\\n") or sys.stdout.flush() or time.sleep(0.01) for i in range(30)]; '
                "time.sleep(2)' "
                "{port}"
            )
            server = BackgroundServer(command, timeout=1, log_file_path=str(log_path))
            with pytest.raises(TimeoutError):
                server.start()
            contents = read_file(log_path)
            assert "BOOT 0" in contents or "BOOT 1" in contents
        finally:
            safely_remove_file(log_path)

    def test_heavy_output_drained_properly(self) -> None:
        # test heavy output is drained properly (no blocking)
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            log_path = tmpfile.name
        try:
            command = (
                f"{sys.executable} -u -c "
                "'import sys, threading as th, http.server as hs; import time; "
                "p=int(sys.argv[1]); "
                'th.Thread(target=lambda: [print(f"SPAM {i}", flush=True) for i in range(2000)], daemon=True).start(); '
                'hs.ThreadingHTTPServer(("localhost", p), hs.SimpleHTTPRequestHandler).serve_forever()\' '
                "{port}"
            )
            server = BackgroundServer(command, log_file_path=str(log_path))
            with server:
                # let the spammer run a bit
                sleep(2)
            contents = read_file(log_path)
            spam_lines = [ln for ln in contents.splitlines() if ln.startswith("SPAM ")]
            assert len(spam_lines) >= 200
        finally:
            try:
                os.remove(log_path)
            except OSError:
                pass

    def test_timeout_when_health_check_never_passes(self) -> None:
        command = (
            f"{sys.executable} -u -c "
            "'import threading as th, http.server as hs, sys, time; "
            "p1=int(sys.argv[1]); p2=int(sys.argv[2]); "
            'th.Thread(target=lambda: hs.ThreadingHTTPServer(("localhost", p1), hs.SimpleHTTPRequestHandler).serve_forever(), daemon=True).start(); '
            "time.sleep(3)' "
            "{port:api} {port:ws}"
        )
        ports = {"api": system.get_available_port(), "ws": system.get_available_port()}
        server = BackgroundServer(command, port=ports, timeout=1)
        with pytest.raises(TimeoutError):
            server.start()
