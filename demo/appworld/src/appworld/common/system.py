import gc
import os
import subprocess
import threading
from collections.abc import Iterable, Iterator
from typing import IO, Literal, Protocol, cast

from appworld.common.collections import list_of
from appworld.common.printer import rprint


class OpenFile(Protocol):
    path: str
    fd: int


def maybe_kill_processes(
    identifier_type: Literal["pid", "port"], identifiers: Iterable[int], timeout: float = 3.0
) -> None:
    import psutil

    if identifier_type == "pid":
        for pid in identifiers:
            try:
                p = psutil.Process(pid)
                p.terminate()
                try:
                    p.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    elif identifier_type == "port":
        try:
            pids = {
                c.pid
                for c in psutil.net_connections(kind="inet")
                if c.laddr and c.laddr.port in identifiers and c.pid
            }
        except (psutil.AccessDenied, PermissionError):
            pids = set()
        # Fallback: per-process scan (works even when system-wide PIDs are hidden)
        if not pids:
            for proc in psutil.process_iter(["pid"]):
                try:
                    for c in proc.connections(kind="inet"):
                        if c.laddr and c.laddr.port in identifiers:
                            pids.add(proc.pid)
                            break
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
        for pid in pids:
            try:
                p = psutil.Process(pid)
                p.terminate()
                try:
                    p.wait(timeout=timeout)
                except psutil.TimeoutExpired:
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        raise ValueError("identifier_type must be either 'pid' or 'port'.")


def set_thread_events(events: list[threading.Event]) -> None:
    for event in events:
        event.set()


def memory_usage_mb() -> float:
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used_mb: float = float(round(memory_info.rss / (1024 * 1024), 2))
    return memory_used_mb


def show_open_file_descriptors() -> None:
    import psutil

    p = psutil.Process(os.getpid())
    rows = []
    # Regular files
    for f in p.open_files():
        rows.append(("file", f.fd, f.path))
    # Sockets (TCP/UDP/UNIX)
    for c in p.connections(kind="all"):
        addr = f"{c.laddr}->{c.raddr}" if c.raddr else f"{c.laddr}"
        rows.append(("socket", c.fd, f"{c.type.name} {addr}"))
    # Fallback count: macOS has no /proc; use psutil's num_fds for the total
    total = p.num_fds() if hasattr(p, "num_fds") else len(rows)
    print(f"Total open descriptors (approx): {total}")
    for type, file_descriptor, description in rows:
        print(f"FD={file_descriptor:<4} {type:<7} {description}")


def get_process_ids_by_keyword(keyword: str) -> list[int]:
    import psutil

    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info.get("cmdline")
        if isinstance(cmdline, list) and keyword in " ".join(cmdline):
            pid = proc.info.get("pid")
            if isinstance(pid, int):
                pids.append(pid)
    return pids


def get_open_files(show: bool = True) -> list[OpenFile]:
    import psutil

    open_files = cast(list[OpenFile], psutil.Process().open_files())
    if show:
        rprint(list_of(open_files, "path"))
    return open_files


class GCThreshold:
    threshold: int | None = None

    @classmethod
    def set(cls, threshold: int | None = 500000) -> None:  # keep gen 1, gen 2 same as default.
        if threshold is None:
            return
        current_threshold = gc.get_threshold()[0]
        if current_threshold == threshold:
            return
        cls.threshold = current_threshold
        gc.set_threshold(threshold)

    @classmethod
    def reset(cls) -> None:
        if cls.threshold is not None:
            gc.set_threshold(cls.threshold)


def subprocess_run_with_capture(subprocess_arguments: list[str]) -> str:
    # Runs a command and captures its output, printing it as its running.
    # Taken verbatim from:
    # https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running/
    def execute(subprocess_arguments_: list[str]) -> Iterator[str]:
        popen = subprocess.Popen(
            subprocess_arguments_, stdout=subprocess.PIPE, universal_newlines=True
        )
        stdout: IO[str] = cast(IO[str], popen.stdout)
        try:
            yield from iter(stdout.readline, "")
        finally:
            stdout.close()
        return_code = popen.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, subprocess_arguments_)

    output = ""
    for path in execute(subprocess_arguments):
        output += path
        rprint(path, end="")
    return output


def get_available_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return cast(int, sock.getsockname()[1])


def is_port_available(port: int) -> bool:
    import socket

    def _is_in_use_ipv4() -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                return sock.connect_ex(("127.0.0.1", port)) == 0
        except OSError:
            return False

    def _is_in_use_ipv6() -> bool:
        if not getattr(socket, "has_ipv6", False):
            return False
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                return sock.connect_ex(("::1", port, 0, 0)) == 0
        except OSError:
            return False

    in_use = _is_in_use_ipv4() or _is_in_use_ipv6()
    return not in_use
