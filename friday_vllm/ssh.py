"""SSH 连接（paramiko）：执行命令、上传 agent.py、本机端口隧道。

不依赖系统自带的 ssh：Windows、Mac 上都一样；密码、密钥文件、ssh-agent 都能用。
"""
import json
import os
import select
import shlex
import socket
import socketserver
import threading
from pathlib import Path

import paramiko

AGENT_NAME = "agent.py"
# 找服务器上能跑 agent.py 的 Python（3.8 起，只用标准库）：非交互的 SSH 常常没有 conda 的 PATH
FIND_PYTHON = r"""
for p in python3 python /root/miniconda3/bin/python3 /opt/conda/bin/python3 /usr/local/bin/python3 /usr/bin/python3 "$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3"; do
  if command -v "$p" >/dev/null 2>&1 && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    command -v "$p"; exit 0
  fi
done
exit 1
"""


class HostKeyQuestion(Exception):
    """没见过的服务器主机密钥：界面问用户要不要信任"""

    def __init__(self, host, key):
        super().__init__(host)
        self.host = host
        self.key = key

    @property
    def fingerprint(self):
        import base64
        import hashlib
        return "SHA256:" + base64.b64encode(hashlib.sha256(self.key.asbytes()).digest()).decode().rstrip("=")


class _AskPolicy(paramiko.MissingHostKeyPolicy):
    def __init__(self, trusted):
        self.trusted = trusted

    def missing_host_key(self, client, hostname, key):
        if self.trusted and self.trusted == key.get_base64():
            client.get_host_keys().add(hostname, key.get_name(), key)
            return
        raise HostKeyQuestion(hostname, key)


class Connection:
    def __init__(self, host, port, user, password=None, key_file=None, known_hosts=None):
        self.host = host
        self.port = int(port or 22)
        self.user = user
        self.password = password or None
        self.key_file = key_file or None
        self.known_hosts = known_hosts
        self.client = None
        self.python = None
        self.base = None
        self._tunnels = {}

    # ── 连接 ──

    def connect(self, trusted_key=None):
        client = paramiko.SSHClient()
        try:
            client.load_system_host_keys()
        except OSError:
            pass
        if self.known_hosts and os.path.exists(self.known_hosts):
            client.load_host_keys(self.known_hosts)
        client.set_missing_host_key_policy(_AskPolicy(trusted_key))
        client.connect(self.host, port=self.port, username=self.user, password=self.password, key_filename=self.key_file,
                       timeout=20, banner_timeout=30, auth_timeout=30, allow_agent=True, look_for_keys=self.key_file is None)
        if trusted_key and self.known_hosts:
            os.makedirs(os.path.dirname(self.known_hosts), exist_ok=True)
            client.save_host_keys(self.known_hosts)
        transport = client.get_transport()
        transport.set_keepalive(30)
        self.client = client
        code, out, err = self.run(FIND_PYTHON, timeout=30)
        if code != 0 or not out.strip():
            raise RuntimeError("服务器上找不到 Python 3.8 以上（运行助手脚本要用）")
        self.python = out.strip().splitlines()[-1]

    @property
    def connected(self):
        return self.client is not None and self.client.get_transport() is not None and self.client.get_transport().is_active()

    def close(self):
        for local_port in list(self._tunnels):
            self.close_tunnel(local_port)
        if self.client:
            self.client.close()
        self.client = None

    # ── 命令 ──

    def run(self, command, timeout=120):
        """跑一条 shell 命令，返回 (退出码, stdout, stderr)"""
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        stdin.close()
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err

    def prepare(self, base):
        """建工作目录、上传 agent.py（每次连上都传：换了新版本的工具也能用上新的 agent）"""
        self.base = base
        code, home, _ = self.run("printf %s \"$HOME\"")
        if base.startswith("~"):
            base = home + base[1:]
            self.base = base
        self.run("mkdir -p %s" % shlex.quote(base))
        # 打包后（PyInstaller）agent.py 作为数据文件放在同样的相对位置
        source = (Path(__file__).resolve().parent / "agent" / AGENT_NAME).read_bytes()
        sftp = self.client.open_sftp()
        try:
            with sftp.open(base.rstrip("/") + "/" + AGENT_NAME, "wb") as f:
                f.write(source)
        finally:
            sftp.close()

    def agent(self, *args, timeout=120):
        """agent.py 的子命令，返回它最后一行 JSON"""
        command = "%s %s %s --base %s" % (shlex.quote(self.python), shlex.quote(self.base.rstrip("/") + "/" + AGENT_NAME),
                                           " ".join(shlex.quote(str(a)) for a in args), shlex.quote(self.base))
        code, out, err = self.run(command, timeout=timeout)
        lines = [line for line in out.splitlines() if line.strip().startswith("{")]
        if not lines:
            raise RuntimeError((err or out or "助手脚本没有输出（退出码 %d）" % code).strip()[-800:])
        return json.loads(lines[-1])

    # ── 隧道：本机 127.0.0.1:local_port → 服务器上的 127.0.0.1:remote_port ──

    def open_tunnel(self, local_port, remote_port):
        if local_port in self._tunnels:
            return
        transport = self.client.get_transport()

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    channel = transport.open_channel("direct-tcpip", ("127.0.0.1", int(remote_port)), self.request.getpeername())
                except Exception:  # noqa: BLE001
                    return
                if channel is None:
                    return
                try:
                    while True:
                        readable, _, _ = select.select([self.request, channel], [], [], 60)
                        if self.request in readable:
                            data = self.request.recv(65536)
                            if not data:
                                break
                            channel.sendall(data)
                        if channel in readable:
                            data = channel.recv(65536)
                            if not data:
                                break
                            self.request.sendall(data)
                except OSError:
                    pass
                finally:
                    channel.close()

        class Server(socketserver.ThreadingTCPServer):
            daemon_threads = True
            allow_reuse_address = True

        server = Server(("127.0.0.1", int(local_port)), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._tunnels[local_port] = server

    def close_tunnel(self, local_port):
        server = self._tunnels.pop(local_port, None)
        if server:
            server.shutdown()
            server.server_close()

    def tunnel_open(self, local_port):
        return local_port in self._tunnels


def free_local_port(preferred):
    """本机上能用的端口：先试 preferred，被占了就让系统给一个"""
    for port in (preferred, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            continue
    return 0
