#!/usr/bin/env python3
"""Friday vLLM 的服务器端助手：图形界面经 SSH 上传到 <工作目录>/agent.py，用服务器上的 Python（3.8 起，只用标准库）运行。

    agent.py probe   --base D                          机器信息（显卡、驱动、磁盘、Python、装没装 vLLM、服务在不在跑），JSON
    agent.py install --base D [--engine vllm|sglang] [--version V] [--index URL|auto]
                                                       后台任务：建虚拟环境、装 uv、装 vLLM / SGLang
    agent.py download --base D --repo ORG/NAME --source modelscope|huggingface|hf-mirror --dir 目标目录
                                                       后台任务：多连接分段下载、断点续传、按 sha256 校验
    agent.py models  --dirs 目录1,目录2                   列出已下好的模型（有 config.json 和 *.safetensors 的目录），JSON
    agent.py serve   --base D [--engine vllm|sglang] --model 目录 ...   后台起一个服务（可以同时有好几个），记下 pid / 端口 / 显卡 / 密钥
    agent.py stop    --base D --job install|download|serve [--id 服务]   停掉
    agent.py forget  --base D --id 服务                   删掉停了的服务的记录和日志
    agent.py status  --base D                          各任务、各服务的状态（含健康检查），JSON
    agent.py tail    --base D --job install|download|serve:<服务> --offset N   从第 N 个字节起读日志，JSON

工作目录 D 的布局：env/（vLLM 的虚拟环境）、env-sglang/（SGLang 的）、uv-cache/、models/（缺省下载位置）、
logs/<任务>.log、state/<任务>.json；服务各一份：state/services/<id>.json、logs/serve-<id>.log。
后台任务用 setsid 脱离 SSH 会话：界面关了、断线了照样跑完，重连后接着看。
"""
import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

VERSION = "0.1.3"
DEFAULT_VLLM = "0.31.0"
# 推理引擎：各装在自己的虚拟环境里（依赖的 torch 版本不一样，装一起会互相顶掉）；vLLM 照旧在 env/
ENGINES = {
    "vllm": {
        "title": "vLLM", "env": "env", "package": "vllm", "spec": "vllm", "default": DEFAULT_VLLM,
        "archs": "import json; from vllm import ModelRegistry; print(json.dumps(sorted(ModelRegistry.get_supported_archs())))",
    },
    "sglang": {
        "title": "SGLang", "env": "env-sglang", "package": "sglang", "spec": "sglang[all]", "default": "latest",
        "archs": "import json; from sglang.srt.models.registry import ModelRegistry; print(json.dumps(sorted(ModelRegistry.get_supported_archs())))",
    },
}
# pip 镜像（自动挑最快的）；国内机器上 PyPI 官方往往最慢
MIRRORS = [
    ("清华", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("中科大", "https://mirrors.ustc.edu.cn/pypi/simple"),
    ("北外", "https://mirrors.bfsu.edu.cn/pypi/web/simple"),
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
    ("腾讯云", "https://mirrors.cloud.tencent.com/pypi/simple"),
    ("PyPI", "https://pypi.org/simple"),
]
SOURCES = {
    "modelscope": "https://modelscope.cn",
    "huggingface": "https://huggingface.co",
    "hf-mirror": "https://hf-mirror.com",
}
# 下载时缺省不要的：GGUF（vLLM 用不了）、原始权重、和 safetensors 重复的 PyTorch 权重、评测结果这类杂项目录
DEFAULT_EXCLUDE = ["*.gguf", "original/*", "*.pth", "*.pt", "*.msgpack", "*.h5", "*.onnx", "onnx/*", "openvino/*"]
CHUNK = 64 << 20          # 分段下载每段 64 MB
CONNECTIONS = 16          # 同时开的连接
DNS_TTL = 300             # 域名解析结果缓存多久（秒）


def cache_dns():
    """下载时缓存域名解析：16 个连接、每段都查 modelscope.cn 和它跳转到的 CDN，查得太多；
    AutoDL 的 DNS 只等 1 秒、不重试（resolv.conf 的 timeout:1 attempts:1），常常报 Temporary failure in name resolution。
    几分钟内同一个域名不再查；查不到时用上一次查到的"""
    real = socket.getaddrinfo
    cache = {}

    def cached(host, *args, **kwargs):
        key = (host, args, tuple(sorted(kwargs.items())))
        hit = cache.get(key)
        if hit and time.time() - hit[0] < DNS_TTL:
            return hit[1]
        try:
            result = real(host, *args, **kwargs)
        except socket.gaierror:
            if hit:
                return hit[1]
            raise
        cache[key] = (time.time(), result)
        return result

    socket.getaddrinfo = cached


def final_url(url):
    """跟着跳转找到真正的下载地址（魔搭的 resolve 会跳到 CDN 上带签名的地址）：之后各段直接用它，少一半请求"""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "friday-vllm/" + VERSION})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.geturl() or url
    except Exception:  # noqa: BLE001
        return url


# ── 小工具 ──

def emit(obj):
    """一行 JSON（界面按行解析）"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(text):
    sys.stdout.write(time.strftime("%H:%M:%S ") + text + "\n")
    sys.stdout.flush()


def paths(base):
    return {
        "env": os.path.join(base, "env"),
        "cache": os.path.join(base, "uv-cache"),
        "models": os.path.join(base, "models"),
        "logs": os.path.join(base, "logs"),
        "state": os.path.join(base, "state"),
    }


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return default


def write_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def run(cmd, timeout=30, env=None):
    try:
        out = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)
        return out.returncode, out.stdout.strip(), out.stderr.strip()
    except Exception as error:  # noqa: BLE001
        return -1, "", str(error)


def alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    # 僵尸进程不算
    try:
        with open("/proc/%d/stat" % int(pid)) as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def human(size):
    size = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return ("%.0f %s" if unit in ("B", "KB") else "%.1f %s") % (size, unit)
        size /= 1024


def disk_free(path):
    probe = path
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    try:
        usage = shutil.disk_usage(probe or "/")
        return {"path": probe, "free": usage.free, "total": usage.total}
    except OSError:
        return None


# ── probe ──

def gpus():
    # index 是 nvidia-smi 的编号（按 PCI 总线顺序）；启动时设 CUDA_DEVICE_ORDER=PCI_BUS_ID，CUDA_VISIBLE_DEVICES 才和它对得上
    fake = os.environ.get("FRIDAY_VLLM_FAKE_GPUS")   # 测试用：没有显卡的机器上假装有（JSON，字段同下）
    if fake:
        return json.loads(fake)
    code, out, _ = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap",
                        "--format=csv,noheader,nounits"])
    result = []
    if code != 0:
        return result
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            result.append({"index": int(parts[0]), "name": parts[1], "memory_total_mib": int(float(parts[2])),
                           "memory_used_mib": int(float(parts[3])), "driver": parts[4], "compute_cap": parts[5]})
    return result


def bundled_cuda(env):
    """引擎环境里 pip 装进来的 CUDA 工具链（nvidia/cu13/bin/nvcc 这类），有好几个时取版本最新的；没有为空。
    SGLang 的 DeepGEMM、FlashInfer 启动时要现场编译内核，没设 CUDA_HOME 就用系统的 /usr/local/cuda——
    AutoDL 镜像里是 12.8，Blackwell（算力 12.x）要 12.9 起，报 NVCC version must be at least 12.9"""
    import glob
    found = []
    for nvcc in glob.glob(os.path.join(env, "lib", "python3*", "site-packages", "nvidia", "*", "bin", "nvcc")):
        home = os.path.dirname(os.path.dirname(nvcc))
        code, out, _ = run([nvcc, "--version"], timeout=20)
        match = re.search(r"release (\d+)\.(\d+)", out)
        if code == 0 and match:
            found.append(((int(match.group(1)), int(match.group(2))), home))
    return max(found)[1] if found else ""


def link_bundled_cuda(home):
    """pip 装的 CUDA 工具链少两样现场编译要的东西：lib64（FlashInfer 用 -L$CUDA_HOME/lib64）、不带版本号的 libcudart.so（-lcudart）"""
    try:
        if not os.path.exists(os.path.join(home, "lib64")) and os.path.isdir(os.path.join(home, "lib")):
            os.symlink("lib", os.path.join(home, "lib64"))
        lib = os.path.join(home, "lib")
        if os.path.isdir(lib) and not os.path.exists(os.path.join(lib, "libcudart.so")):
            versioned = sorted(n for n in os.listdir(lib) if re.match(r"libcudart\.so\.\d+$", n))
            if versioned:
                os.symlink(versioned[-1], os.path.join(lib, "libcudart.so"))
    except OSError as error:
        log("补 CUDA 工具链的链接失败（不一定影响）：%s" % error)


# 现场编译内核用的工具链包：版本要和 torch 用的 CUDA 运行库同一个小版本（CCCL 要求编译器和 CUDA 头文件版本一致，
# nvvm 新了生成的 PTX 老 ptxas 不认）。有的包只要求「不低于」，uv 会装成最新的，装完要对齐
TOOLCHAIN = {
    "nvidia-cuda-runtime": ["nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm"],
    "nvidia-cuda-runtime-cu12": ["nvidia-cuda-nvcc-cu12"],
}


def align_toolchain(env, index, environment):
    python = os.path.join(env, "bin", "python")
    code, out, _ = run([python, "-c", "import importlib.metadata as m, json; print(json.dumps({d.metadata['Name'].lower(): d.version "
                                      "for d in m.distributions() if d.metadata['Name'].lower().startswith('nvidia-')}))"], timeout=60)
    if code != 0:
        return
    installed = json.loads(out.splitlines()[-1])
    wanted = []
    for runtime, tools in TOOLCHAIN.items():
        if runtime not in installed:
            continue
        target = ".".join(installed[runtime].split(".")[:2])
        for tool in tools:
            if tool in installed and ".".join(installed[tool].split(".")[:2]) != target:
                wanted.append("%s==%s.*" % (tool, target))
    if not wanted:
        return
    log("CUDA 编译工具链对齐到运行库的版本：%s" % "、".join(wanted))
    code = subprocess.call([os.path.join(env, "bin", "uv"), "pip", "install", "--python", python, "--index-url", index] + wanted, env=environment)
    if code != 0:
        log("对齐没成（现场编译内核时可能报 CUDA 版本不一致）")


def attention_heads(model_dir):
    """模型的注意力头数（张量并行的份数要能整除它）；读不到为 0"""
    config = read_json(os.path.join(model_dir, "config.json"), {})
    for source in (config, config.get("text_config") or {}, config.get("llm_config") or {}):
        heads = source.get("num_attention_heads")
        if heads:
            return int(heads)
    return 0


def engine_env(base, engine):
    return os.path.join(base, ENGINES[engine]["env"])


def engine_version(base, engine):
    """装好的版本（没装为空）；只读包的元数据（import 要好几秒）"""
    python = os.path.join(engine_env(base, engine), "bin", "python")
    if not os.path.exists(python):
        return ""
    code, out, _ = run([python, "-c", "import importlib.metadata as m; print(m.version(%r))" % ENGINES[engine]["package"]], timeout=20)
    return out if code == 0 else ""


def engine_archs(base, engine):
    state = paths(base)["state"]
    found = read_json(os.path.join(state, "archs-%s.json" % engine))
    if found is None and engine == "vllm":
        found = read_json(os.path.join(state, "archs.json"))   # 0.1.0 写的
    return found or []


def cmd_probe(args):
    p = paths(args.base)
    code, ips, _ = run("hostname -I 2>/dev/null || hostname -i", timeout=5)
    info = {
        "agent": VERSION,
        "os": run("uname -sr")[1],
        "distro": (read_os_release().get("PRETTY_NAME") or ""),
        "arch": os.uname().machine,
        "hostname": socket.gethostname(),
        "ips": [ip for ip in ips.split() if ip and ":" not in ip][:4],
        "python": sys.executable,
        "python_version": "%d.%d.%d" % sys.version_info[:3],
        "gpus": gpus(),
        "disk": disk_free(args.base),
        "base": args.base,
        "engines": {engine: engine_version(args.base, engine) for engine in ENGINES},
        "archs": {engine: engine_archs(args.base, engine) for engine in ENGINES},
        "jobs": jobs_status(args.base),
        "services": services_status(args.base),
    }
    emit(info)


def read_os_release():
    result = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    result[key] = value.strip('"')
    except OSError:
        pass
    return result


# ── 后台任务 ──

def start_job(base, job, argv):
    """把自己以 argv 重新拉起来，脱离 SSH 会话（setsid），日志写 logs/<job>.log，状态写 state/<job>.json"""
    p = paths(base)
    os.makedirs(p["logs"], exist_ok=True)
    os.makedirs(p["state"], exist_ok=True)
    state = read_json(os.path.join(p["state"], job + ".json"), {})
    if state.get("status") == "running" and alive(state.get("pid")):
        emit({"ok": False, "error": "已经在跑了（pid %s）" % state.get("pid")})
        return
    log_path = os.path.join(p["logs"], job + ".log")
    with open(log_path, "w") as out:
        child = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + argv + ["--foreground"], stdout=out, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True, cwd=base)
    write_json(os.path.join(p["state"], job + ".json"),
               {"status": "running", "pid": child.pid, "started": time.time(), "argv": argv, "log": log_path})
    emit({"ok": True, "pid": child.pid, "log": log_path})


def finish_job(base, job, ok, message="", **extra):
    path = os.path.join(paths(base)["state"], job + ".json")
    state = read_json(path, {})
    state.update({"status": "done" if ok else "failed", "finished": time.time(), "message": message})
    state.update(extra)
    write_json(path, state)


def jobs_status(base):
    p = paths(base)
    result = {}
    for job in ("install", "download"):
        state = read_json(os.path.join(p["state"], job + ".json"))
        if not state:
            continue
        if state.get("status") == "running" and not alive(state.get("pid")):
            state["status"] = "failed"
            state["message"] = state.get("message") or "进程不在了（被杀掉或机器重启）"
        result[job] = state
    return result


# ── 服务（可以同时有好几个：不同的模型、不同的显卡、不同的端口）──

def services_dir(base):
    return os.path.join(paths(base)["state"], "services")


def service_path(base, sid):
    return os.path.join(services_dir(base), sid + ".json")


def service_id(name, port):
    return "%s-%s" % (re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "service", port)


def migrate_single_service(base):
    """0.1 只有一个服务（state/serve.json）：挪成 services/ 下的一份"""
    old = os.path.join(paths(base)["state"], "serve.json")
    state = read_json(old)
    if not state:
        return
    sid = service_id(state.get("name") or "service", state.get("port") or 0)
    state["id"] = sid
    write_json(service_path(base, sid), state)
    os.remove(old)


def services_status(base):
    migrate_single_service(base)
    result = {}
    directory = services_dir(base)
    if not os.path.isdir(directory):
        return result
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        state = read_json(os.path.join(directory, name))
        if not state:
            continue
        sid = state.get("id") or name[:-5]
        state["id"] = sid
        if state.get("status") == "running":
            if not alive(state.get("pid")):
                state["status"] = "failed"
                state["message"] = state.get("message") or "进程退出了（看日志）"
                write_json(os.path.join(directory, name), state)
            else:
                state["healthy"] = healthy(state.get("port"))
        result[sid] = state
    return result


def healthy(port):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%s/health" % port, timeout=3) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def cmd_status(args):
    emit({"jobs": jobs_status(args.base), "services": services_status(args.base), "gpus": gpus()})


def cmd_tail(args):
    if args.job.startswith("serve:"):
        path = (read_json(service_path(args.base, args.job[6:])) or {}).get("log") or ""
    else:
        path = os.path.join(paths(args.base)["logs"], args.job + ".log")
    try:
        size = os.path.getsize(path)
        offset = args.offset if 0 <= args.offset <= size else max(0, size - 65536)
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read(1 << 20)
        emit({"offset": offset + len(data), "size": size, "text": data.decode("utf-8", "replace")})
    except OSError:
        emit({"offset": 0, "size": 0, "text": ""})


def cmd_stop(args):
    job = args.job or "serve"
    if job == "serve":
        if not args.id:
            emit({"ok": False, "error": "停哪个服务？要 --id"})
            return
        migrate_single_service(args.base)
        path = service_path(args.base, args.id)
    else:
        path = os.path.join(paths(args.base)["state"], job + ".json")
    state = read_json(path, {})
    pid = state.get("pid")
    if not pid or not alive(pid):
        if state.get("status") == "running":
            state.update({"status": "stopped", "finished": time.time()})
            write_json(path, state)
        emit({"ok": True, "message": "没在跑"})
        return
    # 整个进程组（vLLM 会起好几个子进程）
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except OSError:
        os.kill(int(pid), signal.SIGTERM)
    for _ in range(30):
        if not alive(pid):
            break
        time.sleep(0.5)
    if alive(pid):
        try:
            os.killpg(int(pid), signal.SIGKILL)
        except OSError:
            pass
    state.update({"status": "stopped", "finished": time.time(), "message": "已停止"})
    write_json(path, state)
    emit({"ok": True, "message": "已停止"})


def cmd_forget(args):
    path = service_path(args.base, args.id)
    state = read_json(path)
    if not state:
        emit({"ok": True})
        return
    if state.get("status") == "running" and alive(state.get("pid")):
        emit({"ok": False, "error": "服务还在跑，先停掉"})
        return
    os.remove(path)
    if state.get("log") and os.path.exists(state["log"]):
        os.remove(state["log"])
    emit({"ok": True})


# ── install ──

def fastest_mirror():
    """每个镜像真下一个文件（pip 的轮子，约 2 MB）比速度：只看索引页打开得快不准（阿里云索引页最快、大文件却只有 0.5 MB/s）"""
    best = None
    for name, url in MIRRORS:
        try:
            page_url = url.rstrip("/") + "/pip/"
            with urllib.request.urlopen(page_url, timeout=8) as response:
                page = response.read().decode("utf-8", "replace")
            wheels = re.findall(r'href="([^"]+?\.whl)[^"]*"', page)
            if not wheels:
                raise RuntimeError("索引页里没有轮子")
            wheel = urllib.parse.urljoin(page_url, wheels[-1])
            started = time.time()
            size = 0
            with urllib.request.urlopen(wheel, timeout=10) as response:
                while time.time() - started < 8:
                    block = response.read(256 << 10)
                    if not block:
                        break
                    size += len(block)
            speed = size / max(time.time() - started, 0.001)
            log("镜像 %s：%s/s" % (name, human(speed)))
            if best is None or speed > best[2]:
                best = (name, url, speed)
        except Exception as error:  # noqa: BLE001
            log("镜像 %s 连不上：%s" % (name, str(error)[:80]))
    return best


def find_python():
    """建虚拟环境用的 Python：3.10–3.13（vLLM 的要求）"""
    candidates = [sys.executable, "python3.12", "python3.11", "python3.10", "python3.13", "python3", "python",
                  "/root/miniconda3/bin/python3", "/opt/conda/bin/python3", "/usr/bin/python3"]
    for candidate in candidates:
        path = shutil.which(candidate) or (candidate if os.path.isabs(candidate) and os.path.exists(candidate) else None)
        if not path:
            continue
        code, out, _ = run([path, "-c", "import sys, venv; print('%d.%d' % sys.version_info[:2])"])
        if code == 0:
            major, minor = (int(x) for x in out.split("."))
            if major == 3 and 10 <= minor <= 13:
                return path, out
    return None, None


def cmd_install(args):
    if not args.foreground:
        argv = ["install", "--base", args.base, "--engine", args.engine, "--version", args.version, "--index", args.index]
        return start_job(args.base, "install", argv)
    p = paths(args.base)
    engine = ENGINES[args.engine]
    env = engine_env(args.base, args.engine)
    title = engine["title"]
    try:
        log("装 %s" % title)
        cards = gpus()
        if cards:
            driver = cards[0]["driver"]
            log("显卡：%s，驱动 %s" % (cards[0]["name"], driver))
            if int(driver.split(".")[0]) < 580:
                log("注意：驱动 %s 低于 580，默认的 CUDA 13 版可能起不来；起不来就换旧一点的版本" % driver)
        else:
            # 装不需要显卡（AutoDL 的无卡模式开机装更省钱），运行时才要
            log("注意：现在没检测到 NVIDIA 显卡（AutoDL 无卡模式？）。可以先装，运行时要带显卡开机")
        python, version = find_python()
        if not python:
            raise RuntimeError("服务器上没有 Python 3.10–3.13（带 venv 模块）。Ubuntu 上可以 apt install python3-venv，或装 Miniconda")
        log("Python：%s（%s）" % (python, version))
        index = args.index
        if index == "auto":
            best = fastest_mirror()
            if not best:
                raise RuntimeError("所有 pip 镜像都连不上，检查服务器的网络")
            log("用镜像：%s（%s）" % (best[0], best[1]))
            index = best[1]
        if not os.path.exists(os.path.join(env, "bin", "python")):
            log("建虚拟环境：%s" % env)
            code, out, err = run([python, "-m", "venv", env], timeout=300)
            if code != 0:
                raise RuntimeError("建虚拟环境失败：%s" % (err or out))
        env_python = os.path.join(env, "bin", "python")
        log("装 uv（并行下载、自动重试）")
        code = subprocess.call([env_python, "-m", "pip", "install", "-q", "-i", index, "--upgrade", "uv"])
        if code != 0:
            raise RuntimeError("装 uv 失败")
        environment = dict(os.environ)
        # uv 的缓存放在工作目录里（同一个盘上用硬链接，不多占一份；系统盘常常很小）
        environment.update({"UV_CACHE_DIR": p["cache"], "UV_LINK_MODE": "hardlink", "UV_HTTP_TIMEOUT": "180",
                            "UV_CONCURRENT_DOWNLOADS": "16", "UV_HTTP_RETRIES": "10"})
        version = args.version or engine["default"]
        spec = engine["spec"] if version == "latest" else "%s==%s" % (engine["spec"], version)
        log("装 %s（torch 和 CUDA 库一共好几 GB，要一会儿）" % spec)
        tried = [index] + [url for _, url in MIRRORS if url != index]
        ok = False
        for attempt in tried[:3]:
            log("从 %s 装" % attempt)
            code = subprocess.call([os.path.join(env, "bin", "uv"), "pip", "install", "--python", env_python, "--index-url", attempt, spec],
                                   env=environment)
            if code == 0:
                ok = True
                break
            log("这个镜像没装成，换下一个")
        if not ok:
            raise RuntimeError("%s 没装成，看上面的日志" % title)
        align_toolchain(env, attempt, environment)
        installed = engine_version(args.base, args.engine)
        log("%s %s 装好了；读它支持的模型架构" % (title, installed))
        code, out, err = run([env_python, "-c", engine["archs"]], timeout=600)
        try:
            write_json(os.path.join(p["state"], "archs-%s.json" % args.engine), json.loads(out.splitlines()[-1]))
        except (ValueError, IndexError):
            log("读支持的架构失败（不影响使用）：%s" % (err or out)[-300:])
        finish_job(args.base, "install", True, "%s %s 装好了" % (title, installed), engine=args.engine, version=installed)
        log("完成")
    except Exception as error:  # noqa: BLE001
        log("失败：%s" % error)
        finish_job(args.base, "install", False, str(error))


# ── download ──

def http_json(url, method="GET", body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json",
                                                                             "User-Agent": "friday-vllm/" + VERSION})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_files(source, repo, revision):
    """[{path, size, sha256}]"""
    base = SOURCES[source]
    if source == "modelscope":
        data = http_json("%s/api/v1/models/%s/repo/files?Recursive=true&Revision=%s" % (base, repo, revision or "master"))
        if data.get("Code") not in (200, None) or not data.get("Data"):
            raise RuntimeError("魔搭上找不到 %s：%s" % (repo, data.get("Message")))
        return [{"path": f["Path"], "size": int(f.get("Size") or 0), "sha256": (f.get("Sha256") or "").lower()}
                for f in data["Data"]["Files"] if f.get("Type") == "blob"]
    data = http_json("%s/api/models/%s/tree/%s?recursive=1" % (base, repo, revision or "main"))
    files = []
    for f in data:
        if f.get("type") != "file":
            continue
        lfs = f.get("lfs") or {}
        files.append({"path": f["path"], "size": int(lfs.get("size") or f.get("size") or 0), "sha256": (lfs.get("oid") or "").lower()})
    return files


def file_url(source, repo, revision, path):
    quoted = urllib.parse.quote(path)
    if source == "modelscope":
        return "%s/models/%s/resolve/%s/%s" % (SOURCES[source], repo, revision or "master", quoted)
    return "%s/%s/resolve/%s/%s" % (SOURCES[source], repo, revision or "main", quoted)


def wanted(files, include, exclude):
    names = [f["path"] for f in files]
    has_safetensors = any(n.endswith(".safetensors") for n in names)
    result = []
    for f in files:
        name = f["path"]
        if include and not any(fnmatch.fnmatch(name, pattern) for pattern in include):
            continue
        if any(fnmatch.fnmatch(name, pattern) for pattern in exclude):
            continue
        # 有 safetensors 时不要重复的 PyTorch 权重
        if has_safetensors and re.search(r"(pytorch_model.*\.bin|\.bin\.index\.json)$", name):
            continue
        result.append(f)
    return result


class Progress:
    def __init__(self, total):
        self.total = total
        self.done = 0
        self.lock = threading.Lock()
        self.started = time.time()
        self.last = 0.0
        self.window = []   # (时间, 已下) 算最近几秒的速度

    def add(self, n, name):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last >= 1.0:
                self.last = now
                self.window.append((now, self.done))
                self.window = [w for w in self.window if now - w[0] <= 10]
                speed = (self.window[-1][1] - self.window[0][1]) / max(self.window[-1][0] - self.window[0][0], 0.001) if len(self.window) > 1 else 0
                emit({"event": "progress", "done": self.done, "total": self.total, "speed": speed, "file": name})


def fetch_range(source, start, end, out_path, progress, name, stop):
    """把 [start, end] 写进 out_path 的同一位置；断了从断开的地方接着下（CDN 常常一段只给一部分就断开），连着失败才放弃。
    source = {"url": 跳转后的地址, "origin": 原地址}：跳转后的签名地址过期（403 / 410）时换回原地址、重新跳"""
    position = start
    failures = 0
    while position <= end:
        if stop.is_set():
            return False
        before = position
        try:
            url = source["url"]
            request = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (position, end), "User-Agent": "friday-vllm/" + VERSION})
            with urllib.request.urlopen(request, timeout=60) as response, open(out_path, "r+b") as f:
                if response.status != 206 and (position != 0 or response.headers.get("Content-Length") not in (None, str(end - position + 1))):
                    raise RuntimeError("服务器不支持分段下载（HTTP %d）" % response.status)
                f.seek(position)
                while position <= end:
                    block = response.read(min(1 << 20, end - position + 1))
                    if not block:
                        break
                    f.write(block)
                    position += len(block)
                    progress.add(len(block), name)
            if position <= end:
                raise RuntimeError("连接断开（这一段下到 %d / %d 字节）" % (position - start, end - start + 1))
        except Exception as error:  # noqa: BLE001
            if isinstance(error, urllib.error.HTTPError) and error.code in (403, 410) and source["url"] != source["origin"]:
                source["url"] = final_url(source["origin"])
            # 有进展就不算失败：接着下
            failures = 0 if position > before else failures + 1
            if failures >= 12:
                log("%s 的 %d–%d 段连续失败 12 次，放弃：%s" % (name, start, end, str(error)[:120]))
                return False
            if failures:
                # 偶尔断一下（DNS、CDN 卡一下）自动重试就好，连着 3 次以上才记
                if failures >= 3:
                    log("%s 的 %d–%d 段连续第 %d 次失败（自动重试）：%s" % (name, start, end, failures, str(error)[:120]))
                time.sleep(min(30, 2 ** failures))
    return True


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(source, repo, revision, item, target, progress, pool_size, stop):
    name = item["path"]
    path = os.path.join(target, name)
    part = path + ".part"
    done_marker = path + ".partdone"   # 已下好的段（断点续传）
    size = item["size"]
    if os.path.exists(path) and (not size or os.path.getsize(path) == size):
        progress.add(size, name)
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    origin = file_url(source, repo, revision, name)
    url = {"url": final_url(origin) if size > CHUNK else origin, "origin": origin}
    if size <= CHUNK:
        # 小文件：一次下完
        with open(part, "wb"):
            pass
        ok = fetch_range(url, 0, max(size - 1, 0), part, progress, name, stop) if size else fetch_whole(origin, part)
    else:
        if not os.path.exists(part) or os.path.getsize(part) != size:
            with open(part, "wb") as f:
                f.truncate(size)
            finished = set()
        else:
            finished = set(read_json(done_marker, []))
        ranges = [(i, min(i + CHUNK, size) - 1) for i in range(0, size, CHUNK)]
        progress.add(sum(e - s + 1 for s, e in ranges if s in finished), name)
        todo = [r for r in ranges if r[0] not in finished]
        lock = threading.Lock()
        failed = []

        def worker():
            while not stop.is_set():
                with lock:
                    if not todo:
                        return
                    start, end = todo.pop(0)
                if fetch_range(url, start, end, part, progress, name, stop):
                    with lock:
                        finished.add(start)
                        write_json(done_marker, sorted(finished))
                else:
                    with lock:
                        failed.append(start)
                    stop.set()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(min(pool_size, len(todo)))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok = not failed and len(finished) == len(ranges)
    if not ok:
        return False
    if item.get("sha256"):
        log("校验 %s" % name)
        actual = sha256_of(part)
        if actual != item["sha256"]:
            os.remove(part)
            if os.path.exists(done_marker):
                os.remove(done_marker)
            raise RuntimeError("%s 校验不对（sha256 %s，应为 %s），已删掉，重新下载即可" % (name, actual[:12], item["sha256"][:12]))
    os.replace(part, path)
    if os.path.exists(done_marker):
        os.remove(done_marker)
    return True


def fetch_whole(url, path):
    request = urllib.request.Request(url, headers={"User-Agent": "friday-vllm/" + VERSION})
    with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)
    return True


def cmd_download(args):
    if not args.foreground:
        argv = ["download", "--base", args.base, "--repo", args.repo, "--source", args.source, "--dir", args.dir,
                "--revision", args.revision or ""]
        for pattern in args.include or []:
            argv += ["--include", pattern]
        return start_job(args.base, "download", argv)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    cache_dns()
    try:
        target = os.path.abspath(os.path.expanduser(args.dir))
        log("列 %s 上 %s 的文件" % (args.source, args.repo))
        files = wanted(list_files(args.source, args.repo, args.revision), args.include or [], DEFAULT_EXCLUDE)
        if not any(f["path"].endswith(".safetensors") for f in files):
            raise RuntimeError("这个仓库里没有 safetensors 权重，vLLM 用不了（GGUF 请用 Friday 的本地模型）")
        total = sum(f["size"] for f in files)
        free = disk_free(target)
        if free and free["free"] < total:
            raise RuntimeError("%s 只剩 %s，模型要 %s" % (free["path"], human(free["free"]), human(total)))
        log("%d 个文件，共 %s，下到 %s" % (len(files), human(total), target))
        write_json(os.path.join(paths(args.base)["state"], "download.json"),
                   dict(read_json(os.path.join(paths(args.base)["state"], "download.json"), {}), repo=args.repo, dir=target, total=total))
        progress = Progress(total)
        # 大文件一个个来、每个分段并行；小文件一起下
        for item in sorted(files, key=lambda f: f["size"]):
            if stop.is_set():
                raise RuntimeError("已取消")
            if not download_file(args.source, args.repo, args.revision, item, target, progress, CONNECTIONS, stop):
                raise RuntimeError("已取消" if stop.is_set() else "%s 下载失败" % item["path"])
        with open(os.path.join(target, ".friday-vllm.json"), "w", encoding="utf-8") as f:
            json.dump({"repo": args.repo, "source": args.source, "downloaded": time.time()}, f, ensure_ascii=False)
        emit({"event": "progress", "done": total, "total": total, "speed": 0, "file": ""})
        finish_job(args.base, "download", True, "%s 下好了" % args.repo, dir=target)
        log("完成：%s" % target)
    except Exception as error:  # noqa: BLE001
        log("失败：%s" % error)
        finish_job(args.base, "download", False, str(error))


# ── models ──

def describe_model(directory):
    config = read_json(os.path.join(directory, "config.json"), {})
    size = 0
    partial = False
    for root, _, names in os.walk(directory):
        for name in names:
            if name.endswith((".part", ".partdone")):
                partial = True
            try:
                size += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    text = config.get("text_config") or {}
    quant = config.get("quantization_config") or text.get("quantization_config") or {}
    method = quant.get("quant_method") or ""
    if method == "compressed-tensors":
        groups = quant.get("config_groups") or {}
        formats = {str((g.get("weights") or {}).get("num_bits")) + "bit-" + str((g.get("weights") or {}).get("type")) for g in groups.values()}
        method = "compressed-tensors（%s）" % "、".join(sorted(formats)) if formats else method
        if quant.get("format"):
            method = quant["format"]
    meta = read_json(os.path.join(directory, ".friday-vllm.json"), {})
    return {
        "path": directory,
        "name": meta.get("repo") or os.path.basename(directory.rstrip("/")),
        "size": size,
        "architectures": config.get("architectures") or [],
        "model_type": config.get("model_type") or text.get("model_type") or "",
        "quantization": method,
        "max_len": config.get("max_position_embeddings") or text.get("max_position_embeddings") or 0,
        "heads": attention_heads(directory),
        "complete": not partial,
    }


def cmd_models(args):
    found = []
    seen = set()
    for directory in [d for d in (args.dirs or "").split(",") if d]:
        directory = os.path.abspath(os.path.expanduser(directory))
        if not os.path.isdir(directory):
            continue
        for root, dirs, names in os.walk(directory):
            depth = root[len(directory):].count(os.sep)
            if depth > 4:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            if "config.json" in names and any(n.endswith((".safetensors", ".safetensors.part")) for n in names):
                if root not in seen:
                    seen.add(root)
                    found.append(describe_model(root))
                dirs[:] = []
    emit({"models": found})


def cmd_delete_model(args):
    """删掉服务器上的一个模型目录：只删列模型的那些目录里面的、看得出是模型的；在跑的服务用着、正在下的不删"""
    target = os.path.realpath(args.path)
    roots = [os.path.realpath(os.path.expanduser(d)) for d in (args.dirs or "").split(",") if d]
    root = next((r for r in roots if target.startswith(r.rstrip(os.sep) + os.sep)), None)
    if not root:
        emit({"ok": False, "error": "%s 不在列模型的目录里，不删" % target})
        return
    if not os.path.isdir(target):
        emit({"ok": False, "error": "%s 已经不在了" % target})
        return
    names = os.listdir(target)
    if not ("config.json" in names or ".friday-vllm.json" in names or any(n.endswith((".safetensors", ".safetensors.part")) for n in names)):
        emit({"ok": False, "error": "%s 看着不像模型目录（没有 config.json、safetensors），不删" % target})
        return
    for service in services_status(args.base).values():
        model = os.path.realpath(service.get("model") or "")
        if service.get("status") == "running" and (model == target or model.startswith(target + os.sep)):
            emit({"ok": False, "error": "服务 %s 正用着这个模型：先停掉它" % service["id"]})
            return
    download = jobs_status(args.base).get("download") or {}
    if download.get("status") == "running" and os.path.realpath(download.get("dir") or "") == target:
        emit({"ok": False, "error": "这个模型正在下：先取消下载"})
        return
    size = describe_model(target)["size"]
    shutil.rmtree(target)
    # 空下来的上层目录（models/<组织>/）一起删，删到列模型的那个目录为止
    parent = os.path.dirname(target)
    while parent.startswith(root.rstrip(os.sep) + os.sep):
        try:
            os.rmdir(parent)
        except OSError:
            break
        parent = os.path.dirname(parent)
    emit({"ok": True, "freed": size, "message": "删掉了 %s，空出 %s" % (target, human(size))})


# ── serve ──

def parsers_for(model_dir, engine="vllm"):
    """按模型家族给工具调用、思考解析器（vLLM / SGLang 各自的叫法，照它们的文档）；认不出来的不开工具调用"""
    tool, reasoning = vllm_parsers(model_dir)
    if engine == "sglang":
        tool = {"hermes": "qwen25", "deepseek_v3": "deepseekv3", "openai": "gpt-oss", "llama3_json": "llama3"}.get(tool, tool)
        reasoning = {"deepseek_r1": "deepseek-r1", "openai_gptoss": "gpt-oss"}.get(reasoning, reasoning)
        if tool == "kimi_k2" and not reasoning:
            reasoning = "kimi"
    return tool, reasoning


def vllm_parsers(model_dir):
    config = read_json(os.path.join(model_dir, "config.json"), {})
    kind = (config.get("model_type") or (config.get("text_config") or {}).get("model_type") or "").lower()
    name = os.path.basename(model_dir.rstrip("/")).lower()
    if kind.startswith(("qwen3_5", "qwen3_6", "qwen3_next")) or "coder" in name and kind.startswith("qwen3"):
        return "qwen3_coder", "qwen3"
    if kind.startswith("qwen3"):
        return "hermes", "qwen3"
    if kind.startswith("qwen2"):
        return "hermes", ""
    if kind.startswith("deepseek_v3"):
        return "deepseek_v3", "deepseek_r1"
    if kind.startswith("glm4_moe") or kind.startswith("glm"):
        return "glm45", "glm45"
    if kind.startswith("gpt_oss"):
        return "openai", "openai_gptoss"
    if kind.startswith("kimi"):
        return "kimi_k2", ""
    if kind.startswith("llama"):
        return "llama3_json", ""
    if kind.startswith("mistral") or kind.startswith("mixtral"):
        return "mistral", ""
    return "", ""


def cmd_serve(args):
    p = paths(args.base)
    engine = ENGINES[args.engine]
    python = os.path.join(engine_env(args.base, args.engine), "bin", "python")
    if not engine_version(args.base, args.engine):
        emit({"ok": False, "error": "还没装 %s" % engine["title"]})
        return
    sid = service_id(args.name, args.port)
    running = {k: v for k, v in services_status(args.base).items() if v.get("status") == "running"}
    if sid in running:
        emit({"ok": False, "error": "服务 %s 已经在跑，先停掉" % sid})
        return
    for other in running.values():
        if int(other.get("port") or 0) == args.port:
            emit({"ok": False, "error": "端口 %d 已经给服务 %s 用了，换一个端口" % (args.port, other["id"])})
            return
    cards = gpus()
    if not cards:
        emit({"ok": False, "error": "没检测到 NVIDIA 显卡（nvidia-smi 不能用）：%s 要显卡才能跑。AutoDL 上要用「有卡模式」开机" % engine["title"]})
        return
    model = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.exists(os.path.join(model, "config.json")):
        emit({"ok": False, "error": "%s 不是模型目录（没有 config.json）" % model})
        return
    # 多显卡：用哪几张（不给 = 全部）、张量并行 TP（一份模型切到几张卡上）× 数据并行 DP（开几份副本）
    known = [c["index"] for c in cards]
    chosen = [int(x) for x in args.gpus.split(",") if x.strip()] if args.gpus else known
    missing = [i for i in chosen if i not in known]
    if missing:
        emit({"ok": False, "error": "没有这几张显卡：%s（有 %s）" % (missing, known)})
        return
    tp, dp = max(1, args.tp), max(1, args.dp)
    # 别的服务正用着的卡不能再用（显存会被两边抢光，两个都起不来）
    for other in running.values():
        overlap = sorted(set(other.get("gpus") or []) & set(chosen[:tp * dp]))
        if overlap:
            emit({"ok": False, "error": "显卡 %s 正给服务 %s 用着：换别的卡，或先停掉它" % ("、".join(str(i) for i in overlap), other["id"])})
            return
    if tp * dp > len(chosen):
        emit({"ok": False, "error": "TP %d × DP %d 要 %d 张显卡，只选了 %d 张" % (tp, dp, tp * dp, len(chosen))})
        return
    heads = attention_heads(model)
    if tp > 1 and heads and heads % tp:
        emit({"ok": False, "error": "张量并行 TP %d 要能整除模型的注意力头数 %d：换成 %s" % (
            tp, heads, "、".join(str(n) for n in (1, 2, 4, 8, 16) if heads % n == 0 and n <= len(chosen)))})
        return
    tool, reasoning = parsers_for(model, args.engine)
    if args.tool_parser != "auto":
        tool = args.tool_parser if args.tool_parser != "none" else ""
    if args.reasoning_parser != "auto":
        reasoning = args.reasoning_parser if args.reasoning_parser != "none" else ""
    if args.engine == "sglang":
        # SGLang：RadixAttention 前缀缓存默认开着；显存占比是给权重 + KV 的静态部分
        cmd = [python, "-m", "sglang.launch_server", "--model-path", model, "--served-model-name", args.name, "--host", args.host,
               "--port", str(args.port), "--mem-fraction-static", str(args.gpu_util), "--max-running-requests", str(args.max_seqs),
               "--tp-size", str(tp), "--dp-size", str(dp)]
        if args.max_len:
            cmd += ["--context-length", str(args.max_len)]
        if tool:
            cmd += ["--tool-call-parser", tool]
    else:
        cmd = [os.path.join(engine_env(args.base, args.engine), "bin", "vllm"), "serve", model, "--served-model-name", args.name,
               "--host", args.host, "--port", str(args.port), "--gpu-memory-utilization", str(args.gpu_util),
               "--max-num-seqs", str(args.max_seqs), "--enable-prefix-caching", "--tensor-parallel-size", str(tp)]
        if dp > 1:
            cmd += ["--data-parallel-size", str(dp)]
        if args.max_len:
            cmd += ["--max-model-len", str(args.max_len)]
        if tool:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", tool]
    if args.key:
        cmd += ["--api-key", args.key]
    if reasoning:
        cmd += ["--reasoning-parser", reasoning]
    if args.extra:
        cmd += args.extra.split()
    os.makedirs(p["logs"], exist_ok=True)
    log_path = os.path.join(p["logs"], "serve-%s.log" % sid)
    environment = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1")
    # torch.compile 的缓存放工作目录里：第二次启动快很多
    environment.setdefault("VLLM_CACHE_ROOT", os.path.join(args.base, "vllm-cache"))
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # 引擎环境的 bin（ninja 这类现场编译内核要用的工具在这）和它自带的 CUDA 工具链放在 PATH 最前面
    environment["PATH"] = os.path.join(engine_env(args.base, args.engine), "bin") + os.pathsep + environment.get("PATH", "")
    environment["VIRTUAL_ENV"] = engine_env(args.base, args.engine)
    cuda = bundled_cuda(engine_env(args.base, args.engine))
    if cuda:
        link_bundled_cuda(cuda)
        environment["CUDA_HOME"] = environment["CUDA_PATH"] = cuda
        environment["PATH"] = os.path.join(cuda, "bin") + os.pathsep + environment["PATH"]
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in chosen[:tp * dp])
    with open(log_path, "w") as out:
        out.write("$ " + " ".join(c if args.key is None or c != args.key else "***" for c in cmd) + "\n")
        if cuda:
            out.write("CUDA_HOME=%s（引擎环境自带的 CUDA 工具链）\n" % cuda)
        out.flush()
        child = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                 cwd=args.base, env=environment)
    write_json(service_path(args.base, sid),
               {"id": sid, "status": "running", "pid": child.pid, "started": time.time(), "log": log_path, "port": args.port, "host": args.host,
                "engine": args.engine, "gpus": chosen[:tp * dp], "tp": tp, "dp": dp,
                "key": args.key, "name": args.name, "model": model, "tool_parser": tool, "reasoning_parser": reasoning,
                "options": serve_options(args)})
    emit({"ok": True, "id": sid, "pid": child.pid, "tool_parser": tool, "reasoning_parser": reasoning, "gpus": chosen[:tp * dp], "tp": tp, "dp": dp})


# 启动时的选项原样记进服务记录：停了以后「重新启动」照它再起一次
SERVE_OPTIONS = ("engine", "model", "name", "host", "port", "key", "gpu_util", "max_len", "max_seqs", "tool_parser", "reasoning_parser",
                 "extra", "gpus", "tp", "dp")


def serve_options(args):
    return {name: getattr(args, name) for name in SERVE_OPTIONS}


def cmd_restart(args):
    """照服务记录里的选项再起一次（停了、没起来、机器重启过的服务）"""
    migrate_single_service(args.base)
    state = read_json(service_path(args.base, args.id))
    if not state:
        emit({"ok": False, "error": "没有服务 %s" % args.id})
        return
    if state.get("status") == "running" and alive(state.get("pid")):
        emit({"ok": False, "error": "服务 %s 还在跑" % args.id})
        return
    # 0.1.2 之前的记录没存选项：用记下的那几项，其余按缺省
    options = {"engine": "vllm", "host": "0.0.0.0", "port": 8000, "key": None, "gpu_util": 0.9, "max_len": 0, "max_seqs": 256,
               "tool_parser": state.get("tool_parser") or "none", "reasoning_parser": state.get("reasoning_parser") or "none",
               "extra": "", "tp": 1, "dp": 1}
    options.update({k: state[k] for k in ("engine", "model", "name", "host", "port", "key", "tp", "dp") if state.get(k) is not None})
    options["gpus"] = ",".join(str(i) for i in state.get("gpus") or [])
    options.update(state.get("options") or {})
    if not options.get("model") or not options.get("name"):
        emit({"ok": False, "error": "服务 %s 的记录里没有模型或模型名，到「新建服务」里重新开" % args.id})
        return
    for name, value in options.items():
        setattr(args, name, value)
    args.port = int(args.port)
    cmd_serve(args)


# ── 入口 ──

def main():
    parser = argparse.ArgumentParser(prog="agent.py")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, function):
        command = sub.add_parser(name)
        command.add_argument("--base", default=os.path.expanduser("~/friday-vllm"))
        command.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
        command.set_defaults(function=function)
        return command

    add("probe", cmd_probe)
    add("status", cmd_status)
    command = add("install", cmd_install)
    command.add_argument("--engine", default="vllm", choices=sorted(ENGINES))
    command.add_argument("--version", default="", help="版本号或 latest；不给用各引擎的缺省")
    command.add_argument("--index", default="auto")
    command = add("download", cmd_download)
    command.add_argument("--repo", required=True)
    command.add_argument("--source", default="modelscope", choices=sorted(SOURCES))
    command.add_argument("--dir", required=True)
    command.add_argument("--revision", default="")
    command.add_argument("--include", action="append")
    command = add("models", cmd_models)
    command.add_argument("--dirs", default="")
    command = add("delete-model", cmd_delete_model)
    command.add_argument("--path", required=True)
    command.add_argument("--dirs", required=True, help="列模型的目录（逗号分开）：只删这些目录里面的")
    command = add("serve", cmd_serve)
    command.add_argument("--engine", default="vllm", choices=sorted(ENGINES))
    command.add_argument("--model", required=True)
    command.add_argument("--name", required=True)
    command.add_argument("--host", default="0.0.0.0")
    command.add_argument("--port", type=int, default=8000)
    command.add_argument("--key", default=None)
    command.add_argument("--gpu-util", type=float, default=0.9)
    command.add_argument("--max-len", type=int, default=0)
    command.add_argument("--max-seqs", type=int, default=256)
    command.add_argument("--tool-parser", default="auto")
    command.add_argument("--reasoning-parser", default="auto")
    command.add_argument("--extra", default="")
    command.add_argument("--gpus", default="", help="用哪几张显卡（nvidia-smi 的编号，逗号分开；不给 = 全部）")
    command.add_argument("--tp", type=int, default=1, help="张量并行：一份模型切到几张卡上")
    command.add_argument("--dp", type=int, default=1, help="数据并行：开几份副本")
    command = add("restart", cmd_restart)
    command.add_argument("--id", required=True)
    command = add("stop", cmd_stop)
    command.add_argument("--job", default="serve")
    command.add_argument("--id", default="")
    command = add("forget", cmd_forget)
    command.add_argument("--id", required=True)
    command = add("tail", cmd_tail)
    command.add_argument("--job", required=True)
    command.add_argument("--offset", type=int, default=-1)
    args = parser.parse_args()
    # 后台任务的工作目录是 base：相对路径都先转成绝对的
    args.base = os.path.abspath(os.path.expanduser(args.base))
    for name in ("dir", "model", "path"):
        if getattr(args, name, None):
            setattr(args, name, os.path.abspath(os.path.expanduser(getattr(args, name))))
    os.makedirs(args.base, exist_ok=True)
    args.function(args)


if __name__ == "__main__":
    main()
