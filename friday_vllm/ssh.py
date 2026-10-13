"""SSH 连接（paramiko）：执行命令、上传 agent.py、本机端口隧道、给服务器中转下载、把本机的模型传上去。

不依赖系统自带的 ssh：Windows、Mac 上都一样；密码、密钥文件、ssh-agent 都能用。
"""
import json
import os
import select
import shlex
import socket
import socketserver
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import paramiko

AGENT_NAME = "agent.py"
PARALLEL = 4                 # 中转、上传开几条连接
UPLOAD_CHUNK = 64 << 20      # 上传每段 64 MB
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
        self._relay = []             # 服务器上给本机中转开的端口：[(连接, 端口)]
        self._extra = []             # 中转、上传多开的几条连接（paramiko 一条连接只跑得到十几 MB/s，几条一起才跑得满）
        self._extra_lock = threading.Lock()

    # ── 连接 ──

    def _open_client(self, trusted_key=None):
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
        transport = client.get_transport()
        transport.set_keepalive(30)
        # 窗口开大些：跨境、经网关的链路延迟高，2 MB 的窗口跑不满
        transport.default_window_size = 16 << 20
        return client

    def connect(self, trusted_key=None):
        client = self._open_client(trusted_key)
        if trusted_key and self.known_hosts:
            os.makedirs(os.path.dirname(self.known_hosts), exist_ok=True)
            client.save_host_keys(self.known_hosts)
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
        self._relay = []
        with self._extra_lock:
            for client in self._extra:
                client.close()
            self._extra = []
        if self.client:
            self.client.close()
        self.client = None

    def clients(self, n):
        """n 条连接：这条加上多开的（同一台服务器、同样的登录方式；多开的连不上就少用几条）"""
        with self._extra_lock:
            self._extra = [c for c in self._extra if c.get_transport() is not None and c.get_transport().is_active()]
            key = self.client.get_transport().get_remote_server_key().get_base64()
            while len(self._extra) < n - 1:
                try:
                    self._extra.append(self._open_client(key))
                except Exception:  # noqa: BLE001
                    break
            return [self.client] + self._extra[:n - 1]

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

    # ── 本机中转：服务器上的 127.0.0.1:<端口> → 本机 → 外网 ──

    def open_relay(self, ports=None, n=PARALLEL):
        """服务器经本机上网下模型：几条连接各用 SSH 远程转发在服务器上开一个端口，进来的连接在本机当 HTTPS 代理处理（CONNECT），
        本机设了系统代理就接着走它。返回服务器上的端口（下载时轮流用）；ports 给了就开这几个（重连后接上还在下的任务）"""
        if self.relay_ports:
            return self.relay_ports

        def handler(channel, _origin, _server):
            threading.Thread(target=_relay_connection, args=(channel,), daemon=True).start()

        relay = []
        for index, client in enumerate(self.clients(len(ports) if ports else n)):
            wanted = int(ports[index]) if ports else 0
            try:
                port = client.get_transport().request_port_forward("127.0.0.1", wanted, handler=handler) or wanted
            except Exception:  # noqa: BLE001
                if not relay and index == 0:
                    raise
                continue
            relay.append((client, port))
        if not relay:
            raise RuntimeError("服务器不让开转发端口（sshd 的 AllowTcpForwarding 关了？）")
        self._relay = relay
        return [port for _, port in relay]

    def close_relay(self):
        relay, self._relay = self._relay, []
        for client, port in relay:
            try:
                client.get_transport().cancel_port_forward("127.0.0.1", port)
            except Exception:  # noqa: BLE001
                pass

    @property
    def relay_ports(self):
        return [port for client, port in self._relay if client.get_transport() is not None and client.get_transport().is_active()]

    # ── 把本机的文件传上去 ──

    def remote_free(self, directory):
        """服务器上 directory 所在盘的剩余空间（字节）；查不到为 None"""
        code, out, _ = self.run("mkdir -p %s && df -Pk %s | tail -1" % (shlex.quote(directory), shlex.quote(directory)), timeout=30)
        try:
            return int(out.split()[3]) * 1024 if code == 0 else None
        except (IndexError, ValueError):
            return None

    def upload(self, files, remote, progress, stop, n=PARALLEL):
        """files：[(本机路径, 相对路径, 大小)]，传到服务器的 remote 目录下。大文件切成 64 MB 一段、几条连接一起传；
        先写 <名字>.part（传好的段记在 <名字>.partdone）再改名，已经有同样大小的跳过，断了再传接着传。
        progress(字节数, 相对路径) 每传一块调一次；stop 设了就停"""
        remote = remote.rstrip("/")
        dirs = sorted({os.path.dirname(remote + "/" + rel) for _, rel, _ in files} | {remote})
        code, _, err = self.run("mkdir -p " + " ".join(shlex.quote(d) for d in dirs), timeout=60)
        if code != 0:
            raise RuntimeError("建不了目录 %s：%s" % (remote, err.strip()[-300:]))
        sftp = self.client.open_sftp()
        lock = threading.Lock()
        pieces = []          # (本机路径, 相对路径, 开始, 结束)
        state = {}           # 相对路径 → {"size", "left": 没传的段数, "done": 传好的段}
        try:
            for local, rel, size in sorted(files, key=lambda f: -f[2]):
                dest = remote + "/" + rel
                try:
                    if sftp.stat(dest).st_size == size:
                        progress(size, rel)
                        continue
                except IOError:
                    pass
                part = dest + ".part"
                finished = set()
                try:
                    if sftp.stat(part).st_size == size:
                        with sftp.open(part + "done", "r") as f:
                            finished = set(json.loads(f.read().decode() or "[]"))
                    else:
                        raise IOError
                except (IOError, ValueError):
                    with sftp.open(part, "w") as f:
                        f.truncate(size)
                ranges = [(i, min(i + UPLOAD_CHUNK, size)) for i in range(0, size, UPLOAD_CHUNK)] or [(0, 0)]
                progress(sum(e - b for b, e in ranges if b in finished), rel)
                todo = [(local, rel, b, e) for b, e in ranges if b not in finished]
                state[rel] = {"size": size, "left": len(todo), "done": finished}
                pieces += todo
                if not todo:
                    self._finish_upload(sftp, part, dest)
        finally:
            sftp.close()
        errors = []

        def worker(client):
            channel = client.open_sftp()
            try:
                while not stop.is_set():
                    with lock:
                        if not pieces or errors:
                            return
                        local, rel, begin, end = pieces.pop(0)
                    part = remote + "/" + rel + ".part"
                    with open(local, "rb") as src, channel.open(part, "r+") as dst:
                        dst.set_pipelined(True)
                        src.seek(begin)
                        dst.seek(begin)
                        left = end - begin
                        while left > 0:
                            if stop.is_set():
                                return
                            block = src.read(min(1 << 20, left))
                            if not block:
                                raise RuntimeError("%s 读不完（本机的文件变了？）" % rel)
                            dst.write(block)
                            left -= len(block)
                            progress(len(block), rel)
                    with lock:
                        info = state[rel]
                        info["done"].add(begin)
                        info["left"] -= 1
                        last = info["left"] == 0
                        marker = json.dumps(sorted(info["done"]))
                    if last:
                        self._finish_upload(channel, part, remote + "/" + rel, info["size"])
                    else:
                        with channel.open(part + "done", "w") as f:
                            f.write(marker)
            except Exception as error:  # noqa: BLE001
                with lock:
                    errors.append(error)
                stop.set()
            finally:
                channel.close()

        clients = self.clients(n) if pieces else []
        threads = [threading.Thread(target=worker, args=(client,), daemon=True) for client in clients[:max(1, len(pieces))]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        if stop.is_set():
            raise RuntimeError("已取消")

    @staticmethod
    def _finish_upload(sftp, part, dest, size=None):
        if size is not None and sftp.stat(part).st_size != size:
            raise RuntimeError("%s 传完大小不对" % dest)
        try:
            sftp.posix_rename(part, dest)
        except IOError:
            try:
                sftp.remove(dest)
            except IOError:
                pass
            sftp.rename(part, dest)
        try:
            sftp.remove(part + "done")
        except IOError:
            pass

    def write_file(self, path, data):
        sftp = self.client.open_sftp()
        try:
            with sftp.open(path, "w") as f:
                f.write(data)
        finally:
            sftp.close()


def _read_head(conn):
    """读 HTTP 请求头；返回 (头, 头后面已经读到的字节)"""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(4096)
        if not chunk or len(data) > 65536:
            return None, b""
        data += chunk
    head, rest = data.split(b"\r\n\r\n", 1)
    return head.decode("latin-1"), rest


def _connect_out(host, port):
    """本机连外网：设了系统代理（Windows 的「代理服务器」、Mac 的网络代理、环境变量）就经它 CONNECT 过去"""
    proxy = urllib.request.getproxies().get("https") or urllib.request.getproxies().get("http")
    if proxy and not urllib.request.proxy_bypass(host):
        parsed = urllib.parse.urlsplit(proxy if "://" in proxy else "http://" + proxy)
        if parsed.scheme in ("http", "https") and parsed.hostname:
            sock = socket.create_connection((parsed.hostname, parsed.port or 8080), timeout=20)
            sock.sendall(("CONNECT %s:%d HTTP/1.1\r\nHost: %s:%d\r\n\r\n" % (host, port, host, port)).encode())
            head, rest = _read_head(sock)
            if not head or head.split(" ")[1:2] != ["200"]:
                sock.close()
                raise OSError("本机代理 %s 不让连 %s:%d：%s" % (proxy, host, port, (head or "").splitlines()[0:1]))
            sock.settimeout(None)
            return sock, rest
    sock = socket.create_connection((host, port), timeout=20)
    sock.settimeout(None)
    return sock, b""


def _pipe(a, b):
    """双向转发，任一边断了两边都关"""
    def copy(src, dst):
        try:
            while True:
                data = src.recv(262144)
                if not data:
                    break
                dst.sendall(data)
        except Exception:  # noqa: BLE001
            pass
        finally:
            for conn in (src, dst):
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    thread = threading.Thread(target=copy, args=(b, a), daemon=True)
    thread.start()
    copy(a, b)
    thread.join()


def _relay_connection(channel):
    """服务器经 SSH 远程转发进来的一个连接：当 HTTPS 代理处理（只认 CONNECT，HTTPS 两端直接加密，本机看不到内容）"""
    upstream = None
    try:
        head, rest = _read_head(channel)
        if not head:
            return
        method, target = (head.split(" ") + ["", ""])[:2]
        if method.upper() != "CONNECT":
            channel.sendall(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        host, _, port = target.rpartition(":")
        try:
            upstream, early = _connect_out(host.strip("[]"), int(port or 443))
        except (OSError, ValueError) as error:
            channel.sendall(("HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nX-Friday-Error: %s\r\n\r\n" % str(error)[:200]
                             .replace("\r", " ").replace("\n", " ")).encode("latin-1", "replace"))
            return
        channel.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        if early:
            channel.sendall(early)
        if rest:
            upstream.sendall(rest)
        _pipe(channel, upstream)
    except Exception:  # noqa: BLE001
        pass
    finally:
        for conn in (channel, upstream):
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass


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
