"""在本机查可下载的模型：魔搭（ModelScope）、Hugging Face（或 hf-mirror）。

什么格式的模型都列（GGUF、MLX、ONNX……按名字标出格式，界面上可以只看 vLLM / SGLang 能跑的）。下载在服务器上做（agent.py download），
这里只负责搜索、排序和看看仓库里有什么；筛选（格式、量化、显存放不放得下、引擎支不支持）在界面上对搜到的结果做。
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
# 不是给 N 卡上的 vLLM / SGLang 用的格式（按仓库名认）：GGUF、MLX 这类，昇腾 NPU、海光 / 沐曦 / 天数这类国产卡、手机芯片的转换版；
# 照样能下，界面上标出来
FORMATS = [
    ("GGUF", re.compile(r"gguf|ggml", re.IGNORECASE)),
    ("MLX", re.compile(r"mlx", re.IGNORECASE)),
    ("ONNX", re.compile(r"onnx", re.IGNORECASE)),
    ("OpenVINO", re.compile(r"openvino", re.IGNORECASE)),
    ("CoreML", re.compile(r"coreml", re.IGNORECASE)),
    ("EXL2", re.compile(r"exl2|exl3", re.IGNORECASE)),
    ("昇腾", re.compile(r"npu|ascend|310p|910b|mindie", re.IGNORECASE)),
    ("国产卡", re.compile(r"hygon|metax|iluvatar|corex|cambricon|enflame|kunlunxin", re.IGNORECASE)),
    ("端侧", re.compile(r"rknn|qnn", re.IGNORECASE)),
]
FORMAT_NAMES = [name for name, _ in FORMATS]


def format_of(repo):
    """按仓库名认出的格式；看不出（一般是 safetensors，vLLM / SGLang 能用）为空"""
    for name, pattern in FORMATS:
        if pattern.search(repo):
            return name
    return ""
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
                     "g128", "g64", "g32", "sym", "asym", "marlin", "gemm", "gemv", "compressed", "tensors",
                     # 别的格式的转换版（Qwen3-8B-GGUF、Qwen3-8B-MLX-4bit）也算同一个模型的版本
                     "gguf", "mlx", "onnx", "openvino", "exl2", "exl3"}
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
        self.format = format_of(repo)
        # GGUF 仓库里一般有好几种量化，下的时候选
        self.quant = "多种" if self.format == "GGUF" else quant_of(repo)


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
        # 别的格式（GGUF、MLX……）排在 vLLM / SGLang 能用的后面
        kind = 100 + FORMAT_NAMES.index(model.format) if model.format else 0
        return kind, order.get(model.quant, 99), trust, -(model.downloads or 0)
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
            models.append(Model(repo, int(item.get("StorageSize") or 0), int(item.get("Downloads") or 0), item.get("Architectures") or [],
                                int(item.get("LastUpdatedTime") or 0), int(item.get("Stars") or 0)))
        return models
    params = {"search": text, "limit": limit, "full": "false"}
    if SORTS[sort][2]:
        params.update({"sort": SORTS[sort][2], "direction": -1})
    data = _get(base + "/api/models?" + urllib.parse.urlencode(params))
    return [Model(item["id"], 0, int(item.get("downloads") or 0), [], _hf_time(item.get("lastModified")), int(item.get("likes") or 0))
            for item in data]


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
        siblings = data.get("siblings") or []
        weights = [s for s in siblings if str(s.get("rfilename", "")).endswith(".safetensors")]
        # 有 safetensors 算权重的大小；别的格式算整个仓库（GGUF 是所有量化档加起来）
        size = sum(int(s.get("size") or 0) for s in (weights or siblings))
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
