"""在本机查可下载的模型：魔搭（ModelScope）、Hugging Face（或 hf-mirror）。

只列 vLLM 能用的：有 safetensors 权重的仓库（名字带 GGUF / MLX / ONNX 的排除）。下载在服务器上做（agent.py download），
这里只负责搜索和看看仓库里有什么。
"""
import json
import re
import urllib.parse
import urllib.request

SOURCES = {
    "modelscope": ("魔搭 ModelScope", "https://modelscope.cn"),
    "huggingface": ("Hugging Face", "https://huggingface.co"),
    "hf-mirror": ("hf-mirror（Hugging Face 国内镜像）", "https://hf-mirror.com"),
}
NOT_FOR_VLLM = re.compile(r"(gguf|mlx|onnx|openvino|coreml|exl2|ggml)", re.IGNORECASE)
# 搜索框空着时列的：常用、vLLM 支持得好的官方仓库
SUGGESTED = [
    "Qwen/Qwen3.6-35B-A3B-FP8",
    "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-0.6B",
]


class Model:
    def __init__(self, repo, size=0, downloads=0, architectures=None, note=""):
        self.repo = repo
        self.size = size
        self.downloads = downloads
        self.architectures = architectures or []
        self.note = note


def _get(url, method="GET", body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "User-Agent": "friday-vllm"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def search(source, text, limit=40):
    """[Model]；text 空着时给常用的那几个"""
    base = SOURCES[source][1]
    text = text.strip()
    if source == "modelscope":
        body = {"PageSize": limit, "PageNumber": 1, "SortBy": "Default", "Target": "", "SingleCriterion": [], "Name": text}
        if not text:
            return [info(source, repo) or Model(repo) for repo in SUGGESTED]
        data = _get(base + "/api/v1/dolphin/models", method="PUT", body=body)
        models = []
        for item in (data.get("Data") or {}).get("Model", {}).get("Models") or []:
            repo = "%s/%s" % (item.get("Path"), item.get("Name"))
            if NOT_FOR_VLLM.search(repo):
                continue
            models.append(Model(repo, int(item.get("StorageSize") or 0), int(item.get("Downloads") or 0), item.get("Architectures") or []))
        return models
    if not text:
        return [Model(repo) for repo in SUGGESTED]
    query = urllib.parse.urlencode({"search": text, "limit": limit, "sort": "downloads", "direction": -1, "filter": "safetensors"})
    data = _get(base + "/api/models?" + query)
    return [Model(item["id"], 0, int(item.get("downloads") or 0)) for item in data if not NOT_FOR_VLLM.search(item["id"])]


def info(source, repo):
    """一个仓库的大小、架构（只算会下载的文件）；查不到为 None"""
    base = SOURCES[source][1]
    try:
        if source == "modelscope":
            data = _get("%s/api/v1/models/%s" % (base, repo))
            item = data.get("Data") or {}
            if not item:
                return None
            return Model(repo, int(item.get("StorageSize") or 0), int(item.get("Downloads") or 0), item.get("Architectures") or [])
        data = _get("%s/api/models/%s?blobs=true" % (base, repo))
        size = sum(int(s.get("size") or 0) for s in data.get("siblings") or [] if str(s.get("rfilename", "")).endswith(".safetensors"))
        config = data.get("config") or {}
        return Model(repo, size, int(data.get("downloads") or 0), config.get("architectures") or [])
    except Exception:  # noqa: BLE001
        return None


def human(size):
    size = float(size or 0)
    if size <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return ("%.0f %s" if unit in ("B", "KB") else "%.1f %s") % (size, unit)
        size /= 1024
    return ""
