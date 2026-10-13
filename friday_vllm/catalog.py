"""在本机查可下载的模型：魔搭（ModelScope）、Hugging Face（或 hf-mirror）。

只列 vLLM / SGLang 能用的：有 safetensors 权重的仓库（名字带 GGUF / MLX / ONNX 的排除）。下载在服务器上做（agent.py download），
这里只负责搜索、排序和看看仓库里有什么；筛选（量化、显存放不放得下、引擎支不支持）在界面上对搜到的结果做。
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
# 排序：交给搜索接口排的（魔搭的 SortBy、HF 的 sort）；按大小排在本机做
SORTS = {
    "default": ("相关度", "Default", None),
    "downloads": ("下载量", "DownloadsCount", "downloads"),
    "updated": ("最近更新", "GmtModified", "lastModified"),
    "stars": ("收藏数", "StarsCount", "likes"),
}
PAGE_SIZE = 100
# 不是给 N 卡上的 vLLM / SGLang 用的：GGUF、MLX 这类格式，昇腾 NPU、海光 / 沐曦 / 天数这类国产卡、手机芯片的转换版
NOT_FOR_VLLM = re.compile(r"(gguf|mlx|onnx|openvino|coreml|exl2|ggml|npu|ascend|310p|910b|mindie|rknn|qnn|hygon|metax|iluvatar|corex|"
                          r"cambricon|enflame|kunlunxin)", re.IGNORECASE)
# 量化方式：按仓库名认（顺序要紧：NVFP4 先于 FP8 之类）
QUANTS = [
    ("NVFP4", re.compile(r"nvfp4|(?<![a-z])fp4|mxfp4", re.IGNORECASE)),
    ("FP8", re.compile(r"fp8|w8a8-fp8|fp8-dynamic", re.IGNORECASE)),
    ("AWQ", re.compile(r"awq", re.IGNORECASE)),
    ("GPTQ", re.compile(r"gptq", re.IGNORECASE)),
    ("INT8", re.compile(r"w8a8|int8|8bit|8-bit", re.IGNORECASE)),
    ("INT4", re.compile(r"w4a16|w4a8|int4|4bit|4-bit|bnb", re.IGNORECASE)),
]
QUANT_NAMES = ["原版"] + [name for name, _ in QUANTS]
# 量化方式要的最低算力（compute capability）：NVFP4 是 Blackwell，FP8 是 Ada / Hopper 起
QUANT_MIN_CAP = {"NVFP4": 10.0, "FP8": 8.9}
# 搜索框空着时列的：常用、vLLM / SGLang 支持得好的官方仓库
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


# 量化版的名字在原版后面加的、本身不是量化方式的词（FP8-dynamic、AWQ-4bit、int4-AutoRound、SF-FP8-WC……）
QUANT_EXTRA_WORDS = {"dynamic", "static", "fast", "autoround", "mixed", "sf", "wc", "channel", "block", "quantized", "quant", "nf4",
                     "g128", "g64", "g32", "sym", "asym", "marlin", "gemm", "gemv", "compressed", "tensors"}
# 同一个量化方式有好几家时排前面的：官方、常做量化版的
TRUSTED = ["Qwen", "deepseek-ai", "nvidia", "RedHatAI", "QuantTrio", "unsloth", "Intel", "meta-llama", "mistralai", "zai-org", "moonshotai",
           "openai", "google"]


def quant_of(repo):
    for name, pattern in QUANTS:
        if pattern.search(repo):
            return name
    return "原版"


class Model:
    def __init__(self, repo, size=0, downloads=0, architectures=None, updated=0, stars=0):
        self.repo = repo
        self.size = size
        self.downloads = downloads
        self.architectures = architectures or []
        self.updated = updated      # 秒（Unix 时间）；不知道为 0
        self.stars = stars
        self.quant = quant_of(repo)


def base_name(repo):
    """仓库名去掉末尾的量化后缀：Qwen/Qwen3.6-35B-A3B-FP8 → Qwen3.6-35B-A3B；不是量化版的原样"""
    parts = re.split(r"([-_])", repo.split("/")[-1])
    end = len(parts)
    while end >= 3:
        word = parts[end - 1]
        if not (word.lower() in QUANT_EXTRA_WORDS or any(pattern.fullmatch(word) or pattern.search(word) for _, pattern in QUANTS)):
            break
        end -= 2
    return "".join(parts[:end])


def variants(source, repo):
    """同一个模型的各种量化版（含原版）：按原版的名字搜，名字是「原版名 + 量化后缀」的才算（微调、蒸馏版不算）。
    按量化方式排（原版、NVFP4、FP8……），同一种里官方和常做量化版的在前、再按下载量"""
    base = base_name(repo)
    found = {m.repo: m for m in search(source, base, "downloads") if base_name(m.repo).lower() == base.lower()}
    if repo not in found:
        found[repo] = info(source, repo) or Model(repo)
    models = list(found.values())
    # Hugging Face 的搜索结果没有大小：一个个查（并行）
    missing = [m for m in models if not m.size][:24]
    if missing:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as pool:
            for model, detail in zip(missing, pool.map(lambda m: info(source, m.repo), missing)):
                if detail:
                    model.size, model.architectures = detail.size, detail.architectures or model.architectures
    order = {name: index for index, name in enumerate(QUANT_NAMES)}
    owner = repo.split("/")[0]

    def rank(model):
        org = model.repo.split("/")[0]
        trust = 0 if org == owner else 1 + TRUSTED.index(org) if org in TRUSTED else 99
        return order.get(model.quant, 99), trust, -(model.downloads or 0)
    return sorted(models, key=rank)


def _get(url, method="GET", body=None, timeout=20):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "User-Agent": "friday-vllm"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _hf_time(text):
    """HF 的 lastModified（2026-10-12T08:00:00.000Z）→ 秒"""
    import datetime
    try:
        return int(datetime.datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp())
    except (TypeError, ValueError):
        return 0


def search(source, text, sort="default", limit=PAGE_SIZE):
    """[Model]；text 空着时给常用的那几个"""
    base = SOURCES[source][1]
    text = text.strip()
    if not text:
        return [info(source, repo) or Model(repo) for repo in SUGGESTED]
    if source == "modelscope":
        body = {"PageSize": limit, "PageNumber": 1, "SortBy": SORTS[sort][1], "Target": "", "SingleCriterion": [], "Name": text}
        data = _get(base + "/api/v1/dolphin/models", method="PUT", body=body)
        models = []
        for item in (data.get("Data") or {}).get("Model", {}).get("Models") or []:
            repo = "%s/%s" % (item.get("Path"), item.get("Name"))
            if NOT_FOR_VLLM.search(repo):
                continue
            models.append(Model(repo, int(item.get("StorageSize") or 0), int(item.get("Downloads") or 0), item.get("Architectures") or [],
                                int(item.get("LastUpdatedTime") or 0), int(item.get("Stars") or 0)))
        return models
    params = {"search": text, "limit": limit, "filter": "safetensors", "full": "false"}
    if SORTS[sort][2]:
        params.update({"sort": SORTS[sort][2], "direction": -1})
    data = _get(base + "/api/models?" + urllib.parse.urlencode(params))
    return [Model(item["id"], 0, int(item.get("downloads") or 0), [], _hf_time(item.get("lastModified")), int(item.get("likes") or 0))
            for item in data if not NOT_FOR_VLLM.search(item["id"])]


def info(source, repo):
    """一个仓库的大小、架构；查不到为 None"""
    base = SOURCES[source][1]
    try:
        if source == "modelscope":
            data = _get("%s/api/v1/models/%s" % (base, repo))
            item = data.get("Data") or {}
            if not item:
                return None
            return Model(repo, int(item.get("StorageSize") or 0), int(item.get("Downloads") or 0), item.get("Architectures") or [],
                         int(item.get("LastUpdatedTime") or 0), int(item.get("Stars") or 0))
        data = _get("%s/api/models/%s?blobs=true" % (base, repo))
        size = sum(int(s.get("size") or 0) for s in data.get("siblings") or [] if str(s.get("rfilename", "")).endswith(".safetensors"))
        config = data.get("config") or {}
        return Model(repo, size, int(data.get("downloads") or 0), config.get("architectures") or [], _hf_time(data.get("lastModified")),
                     int(data.get("likes") or 0))
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
