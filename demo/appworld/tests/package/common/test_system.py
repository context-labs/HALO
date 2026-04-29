import subprocess
import sys
import time as _time

import psutil

from appworld.common.system import get_available_port, is_port_available, maybe_kill_processes


class TestCommonSystem:
    def test_maybe_kill_processes(self):
        # test by pid
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        maybe_kill_processes("pid", [p.pid])
        assert not psutil.pid_exists(p.pid)
        # test by port
        port = get_available_port()
        p = subprocess.Popen([sys.executable, "-m", "http.server", str(port)])
        for _ in range(50):
            if not is_port_available(port):
                break
            _time.sleep(0.1)
        maybe_kill_processes("port", [port])
        for _ in range(50):
            if is_port_available(port):
                break
            _time.sleep(0.1)
        assert is_port_available(port)
        if sys.platform != "win32":
            assert not psutil.pid_exists(p.pid)
