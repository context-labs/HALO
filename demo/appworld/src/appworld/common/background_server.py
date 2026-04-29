import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import IO, Any
from typing import get_args as get_type_args  # noqa: F401 (better name)

import requests
from rich import print as rprint

from appworld.common.io import maybe_create_parent_directory, safely_remove_file
from appworld.common.system import get_available_port
from appworld.common.time import TimeoutError


class BackgroundServer:
    def __init__(
        self,
        # The command can have a "{port}" or "{port:<name>}" placeholder for dynamic port assignment.
        command: str | None = None,
        port: int | dict[str, int | None] | None = None,
        health_check_at: str | dict[str, str] = "/",
        timeout: int = 10,
        force_start: bool = False,
        env: dict[str, str] | None = None,
        log_file_path: str | None = None,
        show_logs: bool = False,
        started: bool = False,
        enabled: bool = True,
        name: str = "",
    ):
        self.name = name
        self.enabled = enabled
        command = command or ""
        if enabled and not started and not command:
            raise ValueError(
                "If 'enabled' is True, 'started' is False, i.e., the server needs to be started now, "
                "the 'command' must be provided."
            )
        if enabled and started and port is None:
            raise ValueError(
                "If 'enabled' is True and 'started' is True, i.e., the server is already expected to be running, "
                "the 'port' must be provided."
            )
        if isinstance(port, dict) and isinstance(health_check_at, dict):
            keys_match = set(port.keys()) == set(health_check_at.keys())
            if not keys_match:
                raise ValueError(
                    "If both 'port' and 'health_check_at' are dictionaries, "
                    "they must have the same keys. "
                    f"Found port keys: {list(port.keys())}, "
                    f"health_check_at keys: {list(health_check_at.keys())}."
                )
        self.command = command
        matches_multiple = re.findall(r"\{port:[a-zA-Z0-9_]*\}", self.command)
        matches_single = re.findall(r"\{port\}", self.command)
        if matches_single and isinstance(port, dict):
            raise ValueError(
                "If the command has a '{port}' placeholder, the 'port' cannot be a dictionary. "
                f"Found command: '{self.command}', port: '{port}'"
            )
        if matches_multiple and isinstance(port, int):
            raise ValueError(
                "If the command has '{port:<name>}' placeholders, the 'port' cannot be an integer. "
                f"Found command: '{self.command}', port: '{port}'"
            )
        if enabled:
            # Replace port placeholders in the command with provided ports, if any.
            if isinstance(port, dict):
                for name, value in port.items():
                    if value is not None:
                        self.command = self.command.replace(f"{{port:{name}}}", str(value))
            elif port is not None:
                self.command = self.command.replace("{port}", str(port))
            # For the remaining placeholders, assign available ports, and update the command and port accordingly.
            matches_multiple = re.findall(r"\{port:[a-zA-Z0-9_]*\}", self.command)
            matches_single = re.findall(r"\{port\}", self.command)
            if matches_single:
                port = get_available_port()
                self.command = self.command.replace(matches_single[0], str(port))
            if matches_multiple:
                port_dict: dict[str, int | None]
                if isinstance(port, dict):
                    port_dict = port
                else:
                    port_dict = {}
                for match in matches_multiple:
                    name = match.replace("{port", "").strip("}").strip(":")
                    port_dict[name] = get_available_port()
                    self.command = self.command.replace(match, str(port_dict[name]))
                port = port_dict
            # Ensure that all placeholders have been filled, and port does not have None values.
            if (port is None) or (isinstance(port, dict) and None in port.values()):
                raise ValueError(
                    "If the server is enabled and the command does not have a '{port}' "
                    "or '{port:<name>}' placeholder for all ports, the 'port' must be provided. "
                    f"Found command: '{command}', port: '{port}'"
                )
        self.port = port
        self.proc: subprocess.Popen[str] | None = None
        self.timeout = timeout
        self.health_check_at = health_check_at
        self.force_start = force_start
        self.env = env or {}
        self.log_file_path = log_file_path
        self._log_file: IO[str] | None = None
        self.show_logs = show_logs
        self.started = started
        self.captured_logs: list[str] = []

    @property
    def url(self) -> str | dict[str, str] | None:
        if not self.enabled:
            return None
        if isinstance(self.port, dict):
            return {name: f"http://localhost:{port}" for name, port in self.port.items()}
        return f"http://localhost:{self.port}"

    def start(self, quiet: bool = False) -> None:
        import shlex

        from appworld.common.printer import print_rule
        from appworld.common.system import is_port_available, maybe_kill_processes

        if not self.enabled:
            return
        if not quiet:
            if self.started:
                rprint(f"[bold cyan]Connecting: {self.url}[/bold cyan]")
            else:
                rprint(f"[bold cyan]Running: {self.command}[/bold cyan]")
        if self.started:
            self._wait_until_ready(self.timeout)
            return
        self._verify_port_defined()
        ports: list[int]
        if isinstance(self.port, int):
            ports = [self.port]
        elif isinstance(self.port, dict):
            ports = [p for p in self.port.values() if p is not None]
        for port in ports:
            if not is_port_available(port):
                if not self.force_start:
                    raise ValueError(
                        f"Port {port} is not available for '{self.command}'. "
                        "Please provide a different port, ensure the port is free, "
                        "or set 'force_start' to True."
                    )
                maybe_kill_processes("port", [port])
        if self.log_file_path:
            safely_remove_file(self.log_file_path)
            maybe_create_parent_directory(self.log_file_path)
            self._log_file = open(self.log_file_path, "a", encoding="utf-8")
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        argv = shlex.split(self.command, posix=(sys.platform != "win32"))
        if sys.platform == "win32":
            # Windows argument parsing doesn't treat single quotes specially, which can cause Python
            # to receive a quoted string and do nothing. Strip surrounding quotes. This is a workaround
            # for some of the tests in the tests/package/common/test_background_server.py.
            for index in range(len(argv) - 1):
                if argv[index] == "-c":
                    arg = argv[index + 1]
                    if len(arg) >= 2 and ((arg[0] == arg[-1] == "'") or (arg[0] == arg[-1] == '"')):
                        argv[index + 1] = arg[1:-1]
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", **self.env}
        popen_kwargs: dict[str, Any] = {
            "args": argv,
            "stdout": stdout,
            "stderr": stderr,
            "env": env,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if sys.platform != "win32":
            popen_kwargs["preexec_fn"] = os.setsid
        else:
            if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(**popen_kwargs)
        threading.Thread(target=self._stream_output, daemon=True).start()
        self._wait_until_ready(self.timeout)
        self.started = True
        print()
        substring = f"{self.name} server" if self.name else "Server"
        print_rule(f"🚀 {substring} started at {self.url}")
        print()

    def fill_port_in(self, url: str | None = None) -> str | None:
        if not self.enabled:
            return url
        if url:
            if isinstance(self.port, dict):
                for name, port in self.port.items():
                    url = url.replace(f"{{port:{name}}}", str(port))
            else:
                url = url.replace("{port}", str(self.port))
        return url

    def __enter__(self) -> "BackgroundServer":
        self.start()
        return self

    def _stream_output(self) -> None:
        assert self.proc is not None
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            if self.show_logs:
                print(line, end="")
            if self._log_file:
                self._log_file.write(line)
            self.captured_logs.append(line)
            self.captured_logs = self.captured_logs[-300:]

    def stop(self, timeout: float = 5.0) -> None:
        if not self.enabled:
            return
        if self.proc and self.proc.poll() is None:
            if sys.platform != "win32":
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    self.proc.wait()
            else:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
        if self._log_file:
            self._log_file.close()
        self.started = False

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: object | None
    ) -> None:
        self.stop()

    def _wait_until_ready(self, timeout: int = 10, health_check_at: str | None = None) -> None:
        start_time = time.time()
        health_check_at_: str | dict[str, str] = (
            self.health_check_at if health_check_at is None else health_check_at
        )
        self._verify_port_defined()
        ports: list[int]
        if isinstance(self.port, int):
            ports = [self.port]
        elif isinstance(self.port, dict):
            ports = [p for p in self.port.values() if p is not None]
        health_check_ats: list[str]
        if isinstance(health_check_at_, str):
            health_check_ats = [health_check_at_] * len(ports)
        elif isinstance(self.port, dict):
            health_check_ats = [health_check_at_.get(name, "/") for name in self.port.keys()]
        else:
            health_check_ats = ["/"] * len(ports)
        for port, health_check_at_ in zip(ports, health_check_ats, strict=True):
            url = f"http://localhost:{port}{health_check_at_}"
            go_to_next_port = False
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url, timeout=(1, 3))
                    if response.status_code < 500:  # Don't limit to only 200 OK
                        go_to_next_port = True
                        break
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(0.3)
            if go_to_next_port:
                continue
            message = (
                f"A server for '{self.command}' did not start on port {port} within {timeout} seconds.\n\n"
                "Last captured logs:\n\n"
            )
            message += "".join(self.captured_logs).strip()
            try:
                self.stop()
            except Exception:
                pass
            raise TimeoutError(message.strip())

    def _verify_port_defined(self) -> None:
        if self.port is None or (isinstance(self.port, dict) and None in self.port.values()):
            raise ValueError(
                "The port(s) is/are left None. Cannot start the server without defined port(s). "
                "This should not happen if this class was initialized correctly."
            )
