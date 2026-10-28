"""Friday vLLM 的主窗口：连上远程服务器 → 装 vLLM / SGLang → 列出 / 下载模型 → 启动，给出 OpenAI 兼容地址（远程地址和本机隧道地址）。

Friday 本身不用改：把这里给的地址、密钥、模型名填进 Friday「设置 › 服务与密钥」的自定义服务（OpenAI 兼容）即可。
"""
import json
import os
import fnmatch
import secrets
import shlex
import sys
import threading
import time
import urllib.request

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from . import __version__, catalog
from .agent import agent as agentlib
from .agent.agent import DEFAULT_EXCLUDE
from .ssh import Connection, HostKeyQuestion, free_local_port
from .worker import background

try:
    import keyring
except Exception:  # noqa: BLE001
    keyring = None

APP = "Friday vLLM"
KEYRING_SERVICE = "Friday vLLM"
# 推理引擎：(名字, 缺省版本, 工具调用解析器, 思考解析器)；解析器的叫法两家不一样，照各自的文档
ENGINES = {
    "vllm": ("vLLM", "0.31.0",
             ["auto", "none", "hermes", "qwen3_coder", "qwen3_xml", "llama3_json", "mistral", "deepseek_v3", "deepseek_v31", "glm45",
              "kimi_k2", "openai", "pythonic"],
             ["auto", "none", "qwen3", "deepseek_r1", "glm45", "openai_gptoss", "mistral"]),
    "sglang": ("SGLang", "latest",
               ["auto", "none", "qwen25", "qwen3_coder", "llama3", "mistral", "deepseekv3", "deepseekv31", "glm45", "kimi_k2", "gpt-oss",
                "pythonic"],
               ["auto", "none", "qwen3", "deepseek-r1", "glm45", "gpt-oss", "kimi"]),
}


def human(size):
    return catalog.human(size) or "—"


def duration(seconds):
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return "%d 小时 %d 分" % (seconds // 3600, seconds % 3600 // 60)
    if seconds >= 60:
        return "%d 分 %d 秒" % (seconds // 60, seconds % 60)
    return "%d 秒" % seconds


def local_model_files(folder):
    """本机模型目录里要传的文件 [(本机路径, 相对路径, 大小)]：有 safetensors 时跟服务器上下载时一样不要 GGUF、原始权重、
    重复的 PyTorch 权重；别的格式（GGUF、MLX、ONNX……）整个文件夹都传。隐藏目录（.cache、.git）和没下完的不要"""
    files = []
    for root, dirs, names in os.walk(folder, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder).replace(os.sep, "/")
            if name.startswith(".") or name.endswith((".part", ".partdone", ".incomplete", ".lock")):
                continue
            try:
                files.append((path, rel, os.path.getsize(path)))
            except OSError:
                continue
    if any(rel.endswith(".safetensors") for _, rel, _ in files):
        files = [f for f in files if not any(fnmatch.fnmatch(f[1], pattern) for pattern in DEFAULT_EXCLUDE)
                 and not (f[1].startswith("pytorch_model") and f[1].endswith(".bin")) and not f[1].endswith(".bin.index.json")]
    return files


class FilesDialog(QDialog):
    """选要下仓库里的哪些文件：缺省勾的是 agent.py 缺省会下的（GGUF 只勾一种量化）"""

    def __init__(self, parent, repo, kind, files, chosen):
        super().__init__(parent)
        self.setWindowTitle("选文件：%s" % repo)
        self.resize(760, 560)
        self.files = files
        layout = QVBoxLayout(self)
        note = QLabel("%s 格式，%d 个文件。勾上要下的%s" % (kind, len(files), "（GGUF 一般只要一种量化，比如 Q4_K_M；带视觉的再加一个 mmproj）"
                                                       if kind == "GGUF" else ""))
        note.setWordWrap(True)
        layout.addWidget(note)
        self.table = QTableWidget(len(files), 2)
        self.table.setHorizontalHeaderLabels(["文件", "大小"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for row, f in enumerate(files):
            item = QTableWidgetItem(f["path"])
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if f["path"] in chosen else Qt.Unchecked)
            self.table.setItem(row, 0, item)
            size = QTableWidgetItem(human(f["size"]))
            size.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 1, size)
        self.table.itemChanged.connect(self._summary)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        for title, pick in (("按缺省", lambda f: f.get("default")), ("全选", lambda f: True), ("全不选", lambda f: False)):
            button = QPushButton(title)
            button.clicked.connect(lambda _=False, pick=pick: self._check(pick))
            row.addWidget(button)
        self.total = QLabel()
        row.addWidget(self.total, 1)
        layout.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._summary()

    def _check(self, pick):
        self.table.blockSignals(True)
        for row, f in enumerate(self.files):
            self.table.item(row, 0).setCheckState(Qt.Checked if pick(f) else Qt.Unchecked)
        self.table.blockSignals(False)
        self._summary()

    def chosen(self):
        return {f["path"] for row, f in enumerate(self.files) if self.table.item(row, 0).checkState() == Qt.Checked}

    def _summary(self, *_):
        chosen = self.chosen()
        self.total.setText("勾了 %d 个，共 %s" % (len(chosen), human(sum(f["size"] for f in self.files if f["path"] in chosen))))


def guess_repo(folder):
    """本机模型目录对应的仓库名（组织/名字）：Hugging Face 缓存（models--组织--名字/snapshots/…）、魔搭缓存（…/组织/名字）、
    之前下载时记的；认不出来用目录名"""
    parts = os.path.normpath(os.path.abspath(folder)).split(os.sep)
    for part in parts:
        if part.startswith("models--") and "--" in part[8:]:
            return part[8:].replace("--", "/", 1)
    try:
        with open(os.path.join(folder, ".friday-vllm.json"), encoding="utf-8") as f:
            repo = json.load(f).get("repo") or ""
        if "/" in repo:
            return repo
    except (OSError, ValueError):
        pass
    if "modelscope" in (p.lower() for p in parts) and len(parts) >= 2:
        return "%s/%s" % (parts[-2], parts[-1])
    return parts[-1]


def copy_button(source):
    """「复制」按钮：source 是取文字的函数"""
    button = QPushButton("复制")
    button.setFixedWidth(64)
    button.clicked.connect(lambda: QGuiApplication.clipboard().setText(source()))
    return button


def readonly(text=""):
    field = QLineEdit(text)
    field.setReadOnly(True)
    return field


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("%s %s" % (APP, __version__))
        self.resize(1080, 820)
        self.settings = QSettings("Friday", "Friday vLLM")
        self.conn = None
        self.probe = {}
        self.jobs = {}
        self.models = []
        self.offsets = {}
        self.polling = False
        self.watch = None            # 正在看日志的任务：install / download / serve:<服务 id>
        self.services = {}           # 服务器上的服务（可以同时有好几个），id → 状态
        self.selectedService = None
        self.tunnels = {}            # 服务 id → 本机隧道端口
        self.gpuUse = None           # 别的服务占着的显卡（变了才重建显卡勾选）
        self.download_started = None

        tabs = QTabWidget()
        self.installPage = self._install_tab()
        self.modelsPage = self._models_tab()
        self.downloadPage = self._download_tab()
        self.servePage = self._serve_tab()
        self.servicesPage = self._services_tab()
        tabs.addTab(self._server_tab(), "服务器")
        tabs.addTab(self.installPage, "安装引擎")
        tabs.addTab(self.modelsPage, "模型管理")
        tabs.addTab(self.downloadPage, "下载模型")
        tabs.addTab(self.servePage, "新建服务")
        tabs.addTab(self.servicesPage, "服务")
        self.tabs = tabs

        self.logView = QPlainTextEdit()
        self.logView.setReadOnly(True)
        self.logView.setMaximumBlockCount(5000)
        mono = QFont("Menlo" if sys.platform == "darwin" else "Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.logView.setFont(mono)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(tabs)
        splitter.addWidget(self.logView)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("没连上服务器")

        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self._poll)
        self._engine_changed()
        self._load_profiles()
        self._set_connected(False)

    # ── 日志 ──

    def log(self, text):
        for line in str(text).rstrip("\n").splitlines():
            self.logView.appendPlainText(line)

    def fail(self, title):
        def handler(error):
            message = str(error) or error.__class__.__name__
            self.log("%s：%s" % (title, message))
            if not self.conn or not self.conn.connected:
                self._set_connected(False)
            QMessageBox.warning(self, title, message)
        return handler

    # ── 服务器 ──

    def _server_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.profile = QComboBox()
        self.profile.setEditable(False)
        self.profile.currentIndexChanged.connect(self._pick_profile)
        removeProfile = QPushButton("删除")
        removeProfile.clicked.connect(self._remove_profile)
        row = QHBoxLayout()
        row.addWidget(self.profile, 1)
        row.addWidget(removeProfile)
        form.addRow("已保存的", row)
        self.host = QLineEdit()
        self.host.setPlaceholderText("例如 connect.westb.seetacloud.com 或 192.168.1.10")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(22)
        hostRow = QHBoxLayout()
        hostRow.addWidget(self.host, 1)
        hostRow.addWidget(QLabel("端口"))
        hostRow.addWidget(self.port)
        form.addRow("主机", hostRow)
        self.user = QLineEdit("root")
        form.addRow("用户", self.user)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setPlaceholderText("用密钥登录可以不填")
        self.remember = QCheckBox("记住密码（存在系统的钥匙串 / 凭据管理器里）")
        self.remember.setEnabled(keyring is not None)
        passwordRow = QHBoxLayout()
        passwordRow.addWidget(self.password, 1)
        passwordRow.addWidget(self.remember)
        form.addRow("密码", passwordRow)
        self.keyFile = QLineEdit()
        self.keyFile.setPlaceholderText("不填就用 ~/.ssh 里的密钥和 ssh-agent")
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse_key)
        keyRow = QHBoxLayout()
        keyRow.addWidget(self.keyFile, 1)
        keyRow.addWidget(browse)
        form.addRow("密钥文件", keyRow)
        self.base = QLineEdit()
        self.base.setPlaceholderText("不填：AutoDL 上用 /root/autodl-tmp/friday-vllm（数据盘），其他机器 ~/friday-vllm")
        form.addRow("工作目录", self.base)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.connectButton = QPushButton("连接")
        self.connectButton.setDefault(True)
        self.connectButton.clicked.connect(self._connect)
        self.disconnectButton = QPushButton("断开")
        self.disconnectButton.clicked.connect(self._disconnect)
        buttons.addWidget(self.connectButton)
        buttons.addWidget(self.disconnectButton)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.serverInfo = QLabel()
        self.serverInfo.setTextFormat(Qt.RichText)
        self.serverInfo.setWordWrap(True)
        self.serverInfo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.serverInfo.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        box = QGroupBox("这台服务器")
        boxLayout = QVBoxLayout(box)
        boxLayout.addWidget(self.serverInfo)
        layout.addWidget(box, 1)
        return page

    def _browse_key(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 SSH 私钥", os.path.expanduser("~/.ssh"))
        if path:
            self.keyFile.setText(path)

    def _profiles(self):
        try:
            return json.loads(self.settings.value("profiles", "[]"))
        except (TypeError, ValueError):
            return []

    def _load_profiles(self):
        self.profile.blockSignals(True)
        self.profile.clear()
        self.profile.addItem("（新服务器）", None)
        for item in self._profiles():
            self.profile.addItem("%s@%s:%s" % (item["user"], item["host"], item["port"]), item)
        self.profile.blockSignals(False)
        last = self.settings.value("last", "")
        for i in range(self.profile.count()):
            if self.profile.itemText(i) == last:
                self.profile.setCurrentIndex(i)
                self._pick_profile(i)

    def _pick_profile(self, index):
        item = self.profile.itemData(index)
        if not item:
            return
        self.host.setText(item["host"])
        self.port.setValue(int(item["port"]))
        self.user.setText(item["user"])
        self.keyFile.setText(item.get("key_file", ""))
        self.base.setText(item.get("base", ""))
        self.password.clear()
        self.remember.setChecked(False)
        if keyring is not None:
            try:
                saved = keyring.get_password(KEYRING_SERVICE, self.profile.itemText(index))
            except Exception:  # noqa: BLE001
                saved = None
            if saved:
                self.password.setText(saved)
                self.remember.setChecked(True)

    def _save_profile(self):
        item = {"host": self.host.text().strip(), "port": self.port.value(), "user": self.user.text().strip(),
                "key_file": self.keyFile.text().strip(), "base": self.base.text().strip()}
        name = "%s@%s:%s" % (item["user"], item["host"], item["port"])
        profiles = [p for p in self._profiles() if "%s@%s:%s" % (p["user"], p["host"], p["port"]) != name]
        profiles.insert(0, item)
        self.settings.setValue("profiles", json.dumps(profiles[:20], ensure_ascii=False))
        self.settings.setValue("last", name)
        if keyring is not None:
            try:
                if self.remember.isChecked() and self.password.text():
                    keyring.set_password(KEYRING_SERVICE, name, self.password.text())
                else:
                    keyring.delete_password(KEYRING_SERVICE, name)
            except Exception:  # noqa: BLE001
                pass
        self._load_profiles()

    def _remove_profile(self):
        item = self.profile.currentData()
        if not item:
            return
        name = self.profile.currentText()
        self.settings.setValue("profiles", json.dumps([p for p in self._profiles() if "%s@%s:%s" % (p["user"], p["host"], p["port"]) != name],
                                                      ensure_ascii=False))
        if keyring is not None:
            try:
                keyring.delete_password(KEYRING_SERVICE, name)
            except Exception:  # noqa: BLE001
                pass
        self._load_profiles()

    def _connect(self, trusted_key=None):
        host = self.host.text().strip()
        user = self.user.text().strip()
        if not host or not user:
            QMessageBox.information(self, APP, "先填主机和用户")
            return
        known = os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), "known_hosts")
        conn = Connection(host, self.port.value(), user, self.password.text() or None, self.keyFile.text().strip() or None, known)
        base = self.base.text().strip()
        self.connectButton.setEnabled(False)
        self.statusBar().showMessage("正在连接 %s…" % host)
        self.log("连接 %s@%s:%d" % (user, host, self.port.value()))

        def work():
            conn.connect(trusted_key)
            chosen = base
            if not chosen:
                code, out, _ = conn.run("test -d /root/autodl-tmp && test -w /root/autodl-tmp && echo autodl")
                chosen = "/root/autodl-tmp/friday-vllm" if "autodl" in out else "~/friday-vllm"
            conn.prepare(chosen)
            return conn, conn.agent("probe", timeout=60)

        def done(result):
            self.conn, self.probe = result
            self.base.setText(self.conn.base)
            self._save_profile()
            self._set_connected(True)
            self._show_probe()
            self.log("已连上，Python：%s，工作目录：%s" % (self.conn.python, self.conn.base))
            self._refresh_models()
            self.timer.start()

        def failed(error):
            self.connectButton.setEnabled(True)
            if isinstance(error, HostKeyQuestion):
                answer = QMessageBox.question(self, "第一次连这台服务器",
                                              "%s 的主机密钥指纹是\n\n%s\n\n确认是这台服务器、信任它吗？" % (error.host, error.fingerprint))
                if answer == QMessageBox.Yes:
                    self._connect(trusted_key=error.key.get_base64())
                else:
                    self.statusBar().showMessage("没连上服务器")
                return
            self.statusBar().showMessage("没连上服务器")
            self.fail("连接失败")(error)

        background(work, done=done, failed=failed)

    def _disconnect(self):
        self.timer.stop()
        if self.conn:
            self.conn.close()
        self.conn = None
        self._set_connected(False)
        self.log("已断开（服务器上在跑的安装、下载、服务不受影响）")

    def _set_connected(self, connected):
        self.connectButton.setEnabled(not connected)
        self.disconnectButton.setEnabled(connected)
        for widget in (self.host, self.port, self.user, self.password, self.keyFile, self.base, self.profile):
            widget.setEnabled(not connected)
        # 「下载模型」没连也能用：下到这台电脑上
        for page in (self.installPage, self.modelsPage, self.servePage, self.servicesPage):
            page.setEnabled(connected)
        self._download_place()
        if connected:
            self.statusBar().showMessage("已连上 %s@%s（工作目录 %s）" % (self.conn.user, self.conn.host, self.conn.base))
        else:
            self.statusBar().showMessage("没连上服务器")
            self.serverInfo.setText("")

    def _show_probe(self):
        p = self.probe
        rows = []
        rows.append(("系统", "%s（%s，%s）" % (p.get("distro") or p.get("os"), p.get("arch"), p.get("hostname"))))
        rows.append(("IP", "、".join(p.get("ips") or []) or "—"))
        cards = p.get("gpus") or []
        if cards:
            for card in cards:
                rows.append(("显卡 %s" % card.get("index", ""), "%s，显存 %.1f GB（已用 %.1f GB），驱动 %s，算力 %s" % (
                    card["name"], card["memory_total_mib"] / 1024, card["memory_used_mib"] / 1024, card["driver"], card["compute_cap"])))
        else:
            rows.append(("显卡", "<span style='color:#c33'>没找到 NVIDIA 显卡（vLLM、SGLang 都要 NVIDIA 显卡；AutoDL 无卡模式下可以先装）</span>"))
        disk = p.get("disk") or {}
        if disk:
            rows.append(("磁盘", "%s 剩 %s / 共 %s" % (disk.get("path"), human(disk.get("free")), human(disk.get("total")))))
        rows.append(("Python", "%s（%s）" % (p.get("python"), p.get("python_version"))))
        engines = p.get("engines") or {}
        for engine, (title, *_rest) in ENGINES.items():
            rows.append((title, engines.get(engine) or "没装（到「安装引擎」里装）"))
        for service in (p.get("services") or {}).values():
            if service.get("status") == "running":
                rows.append(("服务", "%s（%s），端口 %s%s%s" % (service.get("name"), ENGINES.get(service.get("engine") or "vllm", ("vLLM",))[0],
                                                          service.get("port"), "，显卡 %s" % ",".join(str(i) for i in service["gpus"]) if service.get("gpus") else "",
                                                          "" if service.get("healthy") else "，还在启动")))
        self.serverInfo.setText("<table cellspacing=6>" + "".join("<tr><td><b>%s</b></td><td>%s</td></tr>" % row for row in rows) + "</table>")
        installed = ["%s %s" % (ENGINES[e][0], v) for e, v in engines.items() if v and e in ENGINES]
        self.installedLabel.setText("已装：" + "、".join(installed) if installed else "还都没装")
        self._engine_changed()
        warning = ""
        if cards:
            major = int(str(cards[0]["driver"]).split(".")[0] or 0)
            if major < 580:
                warning = "显卡驱动 %s 低于 580：新版的 vLLM / SGLang 默认用 CUDA 13，可能起不来；起不来就装旧一点的版本" % cards[0]["driver"]
        else:
            warning = "现在没检测到 NVIDIA 显卡（AutoDL 无卡模式？）：可以先装，运行时要带显卡开机"
        self.driverWarning.setText(warning)
        self.driverWarning.setVisible(bool(warning))
        self._apply_filters()
        self._rebuild_gpus(cards)
        self._apply_jobs(p.get("jobs") or {})
        if p.get("services") is not None and p.get("services") != self.services:
            self._apply_services(p.get("services") or {})

    # ── 安装 ──

    def _install_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.installedLabel = QLabel()
        form.addRow("现在", self.installedLabel)
        self.installEngine = QComboBox()
        for engine, (title, *_rest) in ENGINES.items():
            self.installEngine.addItem(title, engine)
        self.installEngine.currentIndexChanged.connect(
            lambda: self.engineVersion.setText(ENGINES[self.installEngine.currentData()][1]))
        form.addRow("引擎", self.installEngine)
        self.engineVersion = QLineEdit(ENGINES["vllm"][1])
        self.engineVersion.setPlaceholderText("版本号，或 latest")
        form.addRow("版本", self.engineVersion)
        self.mirror = QComboBox()
        self.mirror.addItem("自动（测一下挑最快的）", "auto")
        for name, url in (("清华", "https://pypi.tuna.tsinghua.edu.cn/simple"), ("中科大", "https://mirrors.ustc.edu.cn/pypi/simple"),
                          ("北外", "https://mirrors.bfsu.edu.cn/pypi/web/simple"), ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
                          ("腾讯云", "https://mirrors.cloud.tencent.com/pypi/simple"), ("PyPI 官方", "https://pypi.org/simple")):
            self.mirror.addItem("%s（%s）" % (name, url), url)
        form.addRow("pip 镜像", self.mirror)
        layout.addLayout(form)
        self.driverWarning = QLabel()
        self.driverWarning.setStyleSheet("color: #c33")
        self.driverWarning.setWordWrap(True)
        self.driverWarning.setVisible(False)
        layout.addWidget(self.driverWarning)
        note = QLabel("vLLM 装在工作目录的 env/ 里，SGLang 装在 env-sglang/（两家要的 torch 版本不一样，分开装）；"
                      "各自连同 torch 和 CUDA 库一共 8–10 GB，下载缓存放 uv-cache/。"
                      "安装在服务器上后台跑，关掉这个窗口、断线了也会装完。")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QHBoxLayout()
        self.installButton = QPushButton("安装 / 更新")
        self.installButton.clicked.connect(self._install)
        self.installStop = QPushButton("停止安装")
        self.installStop.clicked.connect(lambda: self._stop_job("install"))
        buttons.addWidget(self.installButton)
        buttons.addWidget(self.installStop)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        self.installStatus = QLabel()
        self.installStatus.setWordWrap(True)
        layout.addWidget(self.installStatus)
        layout.addStretch(1)
        return page

    def _install(self):
        engine = self.installEngine.currentData()
        version = self.engineVersion.text().strip() or "latest"
        self.offsets["install"] = 0
        self.watch = "install"
        self.log("开始安装 %s %s" % (ENGINES[engine][0], version))
        background(self.conn.agent, "install", "--engine", engine, "--version", version, "--index", self.mirror.currentData(),
                   done=self._job_started("安装"), failed=self.fail("安装失败"))

    def _job_started(self, title):
        def done(result):
            if not result.get("ok"):
                QMessageBox.warning(self, title, result.get("error") or "没能开始")
                return
            self._poll()
        return done

    def _stop_job(self, job):
        background(self.conn.agent, "stop", "--job", job, done=lambda r: (self.log(r.get("message", "")), self._poll()),
                   failed=self.fail("停止失败"))

    # ── 模型 ──

    def _models_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        local = QGroupBox("服务器上的模型")
        localLayout = QVBoxLayout(local)
        dirRow = QHBoxLayout()
        dirRow.addWidget(QLabel("也看这些目录"))
        self.extraDirs = QLineEdit(self.settings.value("extraDirs", ""))
        self.extraDirs.setPlaceholderText("逗号分开，例如 /dev/shm,/data/models（工作目录的 models/ 总会看）")
        dirRow.addWidget(self.extraDirs, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._refresh_models)
        dirRow.addWidget(refresh)
        localLayout.addLayout(dirRow)
        self.modelTable = QTableWidget(0, 7)
        self.modelTable.setHorizontalHeaderLabels(["模型", "格式", "大小", "架构", "量化", "状态", "目录"])
        self._table_style(self.modelTable)
        self.modelTable.cellDoubleClicked.connect(self._use_model_row)
        self.modelTable.itemSelectionChanged.connect(self._model_selection_changed)
        localLayout.addWidget(self.modelTable)
        modelButtons = QHBoxLayout()
        self.useModelButton = QPushButton("用它新建服务")
        self.useModelButton.clicked.connect(lambda: self._use_model_row(self.modelTable.currentRow(), 0))
        self.deleteModelButton = QPushButton("删除…")
        self.deleteModelButton.setToolTip("从服务器上删掉选中的模型目录（在跑的服务用着、正在下的不能删）")
        self.deleteModelButton.clicked.connect(self._delete_model)
        modelButtons.addWidget(self.useModelButton)
        modelButtons.addWidget(self.deleteModelButton)
        hint = QLabel("双击一行：在「新建服务」里用它")
        hint.setStyleSheet("color: gray")
        modelButtons.addWidget(hint, 1)
        localLayout.addLayout(modelButtons)
        self._model_selection_changed()
        layout.addWidget(local)
        return page

    def _download_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        remote = QGroupBox("下载模型（下到服务器上）")
        self.downloadBox = remote
        remoteLayout = QVBoxLayout(remote)
        searchRow = QHBoxLayout()
        self.source = QComboBox()
        for key, (title, _) in catalog.SOURCES.items():
            self.source.addItem(title, key)
        searchRow.addWidget(self.source)
        self.query = QLineEdit()
        self.query.setPlaceholderText("搜模型，例如 Qwen3.6 FP8；空着搜给常用的")
        self.query.returnPressed.connect(self._search)
        searchRow.addWidget(self.query, 1)
        searchButton = QPushButton("搜索")
        searchButton.clicked.connect(self._search)
        searchRow.addWidget(searchButton)
        remoteLayout.addLayout(searchRow)
        filterRow = QHBoxLayout()
        filterRow.addWidget(QLabel("排序"))
        self.sortBy = QComboBox()
        for key, (title, _modelscope, _huggingface) in catalog.SORTS.items():
            self.sortBy.addItem(title, key)
        self.sortBy.addItem("大小（从小到大）", "size-asc")
        self.sortBy.addItem("大小（从大到小）", "size-desc")
        self.sortBy.currentIndexChanged.connect(self._sort_changed)
        filterRow.addWidget(self.sortBy)
        filterRow.addSpacing(12)
        filterRow.addWidget(QLabel("量化 / 格式"))
        self.quantFilter = QComboBox()
        self.quantFilter.addItem("全部", "")
        for name in catalog.QUANT_NAMES + catalog.FORMAT_NAMES:
            self.quantFilter.addItem(name, name)
        self.quantFilter.currentIndexChanged.connect(self._apply_filters)
        filterRow.addWidget(self.quantFilter)
        filterRow.addSpacing(12)
        self.fitsGpu = QCheckBox("只看这台显卡跑得动的")
        self.fitsGpu.setToolTip("模型大小放得进显存（多张卡按切开放算，留一成给 KV 缓存），量化方式这些卡也支持（NVFP4 要 Blackwell，FP8 要算力 8.9 起）")
        self.fitsGpu.toggled.connect(self._apply_filters)
        filterRow.addWidget(self.fitsGpu)
        self.engineSupported = QCheckBox("只看装好的引擎支持的")
        self.engineSupported.setToolTip("按装好的 vLLM / SGLang 支持的模型架构；架构不知道的（Hugging Face 的搜索结果）照样列")
        self.engineSupported.toggled.connect(self._apply_filters)
        filterRow.addWidget(self.engineSupported)
        self.vllmOnly = QCheckBox("只看 vLLM / SGLang 能跑的")
        self.vllmOnly.setToolTip("去掉 GGUF、MLX、ONNX 这类格式和昇腾、国产卡、手机芯片的转换版（按仓库名认）；不勾什么模型都列、都能下")
        self.vllmOnly.setChecked(self.settings.value("vllmOnly", "false") == "true")
        self.vllmOnly.toggled.connect(lambda on: (self.settings.setValue("vllmOnly", "true" if on else "false"), self._apply_filters()))
        filterRow.addWidget(self.vllmOnly)
        filterRow.addStretch(1)
        self.resultCount = QLabel()
        self.resultCount.setStyleSheet("color: gray")
        filterRow.addWidget(self.resultCount)
        remoteLayout.addLayout(filterRow)
        self.resultsAll = []
        self.results = []
        self.resultTable = QTableWidget(0, 8)
        self.resultTable.setHorizontalHeaderLabels(["仓库", "格式", "量化", "大小", "下载量", "更新", "架构", "装好的引擎支持吗"])
        self._table_style(self.resultTable)
        self.resultTable.itemSelectionChanged.connect(self._pick_result)
        remoteLayout.addWidget(self.resultTable)
        grid = QGridLayout()
        grid.addWidget(QLabel("仓库"), 0, 0)
        self.repo = QLineEdit()
        self.repo.setPlaceholderText("组织/名字，例如 Qwen/Qwen3.6-35B-A3B-FP8")
        self.repo.textChanged.connect(self._suggest_target)
        self.repo.textChanged.connect(self._repo_changed)
        grid.addWidget(self.repo, 0, 1)
        grid.addWidget(QLabel("量化版本"), 1, 0)
        self.variant = QComboBox()
        self.variant.activated.connect(self._pick_variant)
        grid.addWidget(self.variant, 1, 1)
        self.variantKey = None       # 列着量化版本的那个模型（来源、原版名），变了才重新找
        self.variantTimer = QTimer(self)
        self.variantTimer.setSingleShot(True)
        self.variantTimer.setInterval(600)
        self.variantTimer.timeout.connect(self._find_variants)
        self.source.currentIndexChanged.connect(self._repo_changed)
        grid.addWidget(QLabel("文件"), 2, 0)
        self.filesText = QLabel()
        self.filesText.setStyleSheet("color: gray")
        self.filesButton = QPushButton("选文件…")
        self.filesButton.setToolTip("选只下仓库里的哪些文件（GGUF 仓库常有十几种量化，缺省只下一种）")
        self.filesButton.clicked.connect(self._pick_files)
        filesRow = QHBoxLayout()
        filesRow.addWidget(self.filesText, 1)
        filesRow.addWidget(self.filesButton)
        grid.addLayout(filesRow, 2, 1)
        self.repoFiles = None        # 仓库里的文件：{"key": (来源, 仓库), "format", "files": [{path, size, default}]}
        self.fileChoice = None       # 自己选的文件（路径集合）；None = 按缺省
        grid.addWidget(QLabel("下到"), 3, 0)
        self.target = QLineEdit()
        self.target.setPlaceholderText("服务器上的目录")
        self.browseTarget = QPushButton("选目录…")
        self.browseTarget.setToolTip("没连服务器时模型下到这台电脑上：选放模型的目录（下到 <目录>/<组织>/<名字>）")
        self.browseTarget.clicked.connect(self._pick_local_dir)
        targetRow = QHBoxLayout()
        targetRow.addWidget(self.target, 1)
        targetRow.addWidget(self.browseTarget)
        grid.addLayout(targetRow, 3, 1)
        self.downloadButton = QPushButton("下载")
        self.downloadButton.clicked.connect(self._download)
        self.downloadStop = QPushButton("取消下载")
        self.downloadStop.clicked.connect(self._cancel_download)
        grid.addWidget(self.downloadButton, 0, 2)
        grid.addWidget(self.downloadStop, 3, 2)
        self.viaLabel = QLabel("下载方式")
        grid.addWidget(self.viaLabel, 4, 0)
        self.downloadVia = QComboBox()
        self.downloadVia.addItem("服务器直接下", "direct")
        self.downloadVia.addItem("经本机中转（服务器走这台电脑的网络下，本机有代理也走代理；窗口要开着）", "relay")
        self.downloadVia.setToolTip("服务器上下不动（Hugging Face 被墙、镜像跳到国外 CDN）时用：SSH 远程转发在服务器上开一个代理端口，"
                                    "下载请求经它从这台电脑出去。服务器那边照样多连接、断点续传；速度看这台电脑的网和上行带宽")
        index = self.downloadVia.findData(self.settings.value("downloadVia", "direct"))
        self.downloadVia.setCurrentIndex(max(0, index))
        grid.addWidget(self.downloadVia, 4, 1)
        self.uploadButton = QPushButton("从本机上传…")
        self.uploadButton.setToolTip("选这台电脑上已经下好的模型文件夹，传到服务器上；断了重新传会接着传")
        self.uploadButton.clicked.connect(self._upload_model)
        grid.addWidget(self.uploadButton, 4, 2)
        # 在本机跑的传输（上传到服务器、没连服务器时下到这台电脑上）：{"kind", "stop", "done", "total", "file", "lines", 速度窗口}
        self.transfer = None
        self.relayReopening = False
        self.transferTimer = QTimer(self)
        self.transferTimer.setInterval(500)
        self.transferTimer.timeout.connect(self._show_transfer)
        remoteLayout.addLayout(grid)
        self.downloadBar = QProgressBar()
        self.downloadBar.setRange(0, 1000)
        self.downloadText = QLabel()
        remoteLayout.addWidget(self.downloadBar)
        remoteLayout.addWidget(self.downloadText)
        layout.addWidget(remote)
        return page

    @staticmethod
    def _table_style(table):
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)

    def _model_dirs(self):
        dirs = [self.conn.base.rstrip("/") + "/models"]
        dirs += [d.strip() for d in self.extraDirs.text().split(",") if d.strip()]
        return dirs

    def _refresh_models(self):
        if not self.conn:
            return
        self.settings.setValue("extraDirs", self.extraDirs.text())
        background(self.conn.agent, "models", "--dirs", ",".join(self._model_dirs()), timeout=120,
                   done=self._show_models, failed=self.fail("列模型失败"))

    def _show_models(self, result):
        self.models = result.get("models") or []
        self.modelTable.setRowCount(len(self.models))
        for row, model in enumerate(self.models):
            values = [model["name"], model.get("format") or "safetensors", human(model["size"]), "、".join(model["architectures"]),
                      model["quantization"] or "—",
                      "完整" if model["complete"] else "没下完", model["path"]]
            for column, value in enumerate(values):
                self.modelTable.setItem(row, column, QTableWidgetItem(str(value)))
        current = self.serveModel.currentText()
        self.serveModel.blockSignals(True)
        self.serveModel.clear()
        for model in self.models:
            # 旧 agent 没有 servable：都算能跑
            if model["complete"] and model.get("servable", True):
                self.serveModel.addItem("%s（%s）" % (model["name"], human(model["size"])), model["path"])
        self.serveModel.blockSignals(False)
        index = self.serveModel.findText(current)
        if index >= 0:
            self.serveModel.setCurrentIndex(index)
        self._model_changed()
        self._model_selection_changed()

    def _selected_local_model(self):
        rows = self.modelTable.selectionModel().selectedRows()
        return self.models[rows[0].row()] if rows and rows[0].row() < len(self.models) else None

    def _model_selection_changed(self):
        model = self._selected_local_model()
        self.useModelButton.setEnabled(bool(model and model["complete"] and model.get("servable", True)))
        self.deleteModelButton.setEnabled(bool(model))

    def _delete_model(self):
        model = self._selected_local_model()
        if not model or not self.conn:
            return
        users = [sv.get("name") or sid for sid, sv in self.services.items()
                 if sv.get("status") == "running" and (sv.get("model") or "").rstrip("/") == model["path"].rstrip("/")]
        if users:
            QMessageBox.information(self, APP, "服务 %s 正用着这个模型：先在「服务」里停掉" % "、".join(users))
            return
        answer = QMessageBox.question(self, "删除模型", "从服务器上删掉这个模型？\n\n%s（%s）\n%s\n\n删了要用得重新下。" % (
            model["name"], human(model["size"]), model["path"]), QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return
        self.deleteModelButton.setEnabled(False)
        self.log("删除模型 %s" % model["path"])

        def done(result):
            if not result.get("ok"):
                QMessageBox.warning(self, "删除失败", result.get("error") or "没删掉")
            else:
                self.log(result.get("message") or "删掉了")
            self._refresh_models()

        background(self.conn.agent, "delete-model", "--path", model["path"], "--dirs", ",".join(self._model_dirs()), timeout=300,
                   done=done, failed=self.fail("删除失败"))

    def _use_model_row(self, row, _column):
        if not 0 <= row < len(self.models):
            return
        path = self.models[row]["path"]
        index = self.serveModel.findData(path)
        if index >= 0:
            self.serveModel.setCurrentIndex(index)
            self.tabs.setCurrentWidget(self.servePage)

    def _search(self):
        source = self.source.currentData()
        text = self.query.text()
        # 相关度、下载量、最近更新、收藏数交给搜索接口排（一页 100 个，排完再取）；按大小在本机排
        order = self.sortBy.currentData()
        sort = order if order in catalog.SORTS else "default"
        self.log("在%s搜「%s」（按%s）" % (catalog.SOURCES[source][0], text or "常用", self.sortBy.currentText()))
        background(catalog.search, source, text, sort, done=self._show_results, failed=self.fail("搜索失败"))

    def _sort_changed(self, *_):
        # 交给接口排的要重新搜（排在前 100 个以外的这样才看得到）；按大小的直接在搜到的里面排
        if self.sortBy.currentData() in catalog.SORTS and self.query.text().strip() and self.resultsAll:
            self._search()
        else:
            self._apply_filters()

    def _show_results(self, models):
        self.resultsAll = models
        self._apply_filters()
        if not models:
            self.log("没搜到")

    def _engine_archs(self):
        """装好的引擎各自支持的架构（没装、没读到的不算）"""
        engines = self.probe.get("engines") or {}
        return {e: set(a) for e, a in (self.probe.get("archs") or {}).items() if a and engines.get(e) and e in ENGINES}

    def _apply_filters(self, *_):
        models = list(self.resultsAll)
        quant = self.quantFilter.currentData()
        if quant:
            models = [m for m in models if m.quant == quant or m.format == quant]
        if self.vllmOnly.isChecked():
            models = [m for m in models if not m.format]
        cards = self.probe.get("gpus") or []
        self.fitsGpu.setEnabled(bool(cards))
        if not cards:
            self.fitsGpu.setToolTip("没检测到这台服务器的显卡（没连上，或 AutoDL 无卡模式）")
        if self.fitsGpu.isChecked() and cards:
            models = [m for m in models if not self._gpu_problem(m)]
        archs = self._engine_archs()
        self.engineSupported.setEnabled(bool(archs))
        if self.engineSupported.isChecked() and archs:
            models = [m for m in models if not m.architectures or any(a in known for known in archs.values() for a in m.architectures)]
        order = self.sortBy.currentData()
        if order == "size-asc":
            models.sort(key=lambda m: (m.size == 0, m.size))
        elif order == "size-desc":
            models.sort(key=lambda m: -m.size)
        self.results = models
        self.resultTable.setRowCount(len(models))
        for row, model in enumerate(models):
            supported = "—"
            if archs and model.architectures:
                supported = "、".join("%s %s" % (ENGINES[e][0], "支持" if any(x in known for x in model.architectures) else "不支持")
                                     for e, known in archs.items())
            values = [model.repo, model.format or "—", model.quant, human(model.size), str(model.downloads or ""),
                      time.strftime("%Y-%m-%d", time.localtime(model.updated)) if model.updated else "—",
                      "、".join(model.architectures), supported]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (3, 4):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.resultTable.setItem(row, column, item)
        self.resultCount.setText("显示 %d / 共 %d 个" % (len(models), len(self.resultsAll)) if self.resultsAll else "")

    def _gpu_problem(self, model):
        """这台的显卡跑不动这个模型的原因（量化方式不支持、放不下）；跑得动或不知道显卡时为空"""
        if model.format:
            return "vLLM / SGLang 跑不了（%s）" % model.format
        cards = self.probe.get("gpus") or []
        if not cards:
            return ""
        memory = sum(c["memory_total_mib"] for c in cards) * 1024 * 1024 * 0.9
        try:
            capability = min(float(c["compute_cap"]) for c in cards)
        except ValueError:
            capability = 0.0
        if capability < catalog.QUANT_MIN_CAP.get(model.quant, 0):
            return "这台的显卡不支持 %s" % model.quant
        if model.size and model.size > memory:
            return "显存放不下"
        return ""

    def _repo_changed(self, *_):
        self.fileChoice = None
        self.variantTimer.start()

    def _find_files(self):
        """仓库填好后列它的文件：缺省会下哪些、多大，「选文件…」里改"""
        repo = self.repo.text().strip()
        key = (self.source.currentData(), repo)
        if "/" not in repo:
            self.repoFiles = None
            self.filesText.setText("")
            self.filesButton.setEnabled(False)
            return
        if self.repoFiles and self.repoFiles["key"] == key:
            self._show_files()
            return
        self.repoFiles = {"key": key, "format": "", "files": None}
        self.filesText.setText("在列 %s 的文件…" % repo)
        self.filesButton.setEnabled(False)

        def work():
            agentlib.setup_http()
            files = agentlib.list_files(key[0], repo, "")
            chosen = {f["path"] for f in agentlib.default_files(files)}
            return agentlib.file_format([f["path"] for f in files]), [dict(f, default=f["path"] in chosen) for f in files]

        def done(result):
            if self.repoFiles and self.repoFiles["key"] == key:
                self.repoFiles["format"], self.repoFiles["files"] = result
                self._show_files()

        def failed(error):
            if self.repoFiles and self.repoFiles["key"] == key:
                self.repoFiles = None
                self.filesText.setText("列不出文件（%s），下载时按缺省挑" % (str(error) or error.__class__.__name__)[:120])

        background(work, done=done, failed=failed)

    def _show_files(self):
        info = self.repoFiles or {}
        files = info.get("files")
        if files is None:
            return
        self.filesButton.setEnabled(bool(files))
        if self.fileChoice is None:
            chosen = [f for f in files if f["default"]]
            how = {"safetensors": "缺省：safetensors 权重和配置", "GGUF": "缺省：只下一种量化（%s）" % "、".join(
                sorted({agentlib.gguf_tag(f["path"]) for f in chosen if f["path"].lower().endswith(".gguf") and agentlib.gguf_tag(f["path"])}))
            }.get(info["format"], "缺省：整个仓库")
        else:
            chosen = [f for f in files if f["path"] in self.fileChoice]
            how = "自己选的"
        note = "" if info["format"] in ("safetensors", "PyTorch") else "；%s 格式，vLLM / SGLang 跑不了" % info["format"]
        self.filesText.setText("%s，%d / %d 个文件，%s%s" % (how, len(chosen), len(files), human(sum(f["size"] for f in chosen)), note))

    def _pick_files(self):
        info = self.repoFiles or {}
        if not info.get("files"):
            return
        chosen = self.fileChoice if self.fileChoice is not None else {f["path"] for f in info["files"] if f["default"]}
        dialog = FilesDialog(self, info["key"][1], info["format"], info["files"], chosen)
        if dialog.exec() == QDialog.Accepted:
            picked = dialog.chosen()
            self.fileChoice = None if picked == {f["path"] for f in info["files"] if f["default"]} else picked
            self._show_files()

    def _include_args(self):
        """自己选了文件时传给 agent 的 --include（精确路径）；按缺省为空"""
        if self.fileChoice is None:
            return []
        if not self.fileChoice:
            raise ValueError("一个文件都没选")
        return sorted(self.fileChoice)

    def _find_variants(self):
        """仓库填好后找同一个模型的原版和各种量化版，列进「量化版本」"""
        self._find_files()
        repo = self.repo.text().strip()
        source = self.source.currentData()
        if "/" not in repo:
            self.variant.clear()
            self.variantKey = None
            return
        key = (source, catalog.base_name(repo).lower())
        if key == self.variantKey:
            self._select_variant(repo)
            return
        self.variantKey = key
        self.variant.clear()
        self.variant.addItem("在找 %s 的量化版本…" % catalog.base_name(repo))
        self.variant.setEnabled(False)

        def done(models):
            if self.variantKey == key:
                self._show_variants(models)

        def failed(error):
            if self.variantKey == key:
                self.variantKey = None
                self.variant.clear()
                self.variant.addItem("没找到量化版本（%s）" % (str(error) or error.__class__.__name__))

        background(catalog.variants, source, repo, done=done, failed=failed)

    def _show_variants(self, models):
        self.variant.clear()
        for model in models:
            problem = self._gpu_problem(model)
            self.variant.addItem("%s · %s%s%s" % (model.format or model.quant, model.repo, "（%s）" % human(model.size) if model.size else "",
                                                 " · " + problem if problem else ""), model.repo)
        self.variant.setEnabled(len(models) > 1)
        self.variant.setToolTip("没找到别的版本" if len(models) <= 1 else
                                "同一个模型的原版和各种量化版（按名字找的，微调、蒸馏版不算），选一个就下它；"
                                "同一种量化官方和常做量化版的排前面，再按下载量")
        self._select_variant(self.repo.text().strip())

    def _select_variant(self, repo):
        index = self.variant.findData(repo)
        if index >= 0:
            self.variant.setCurrentIndex(index)

    def _pick_variant(self, index):
        repo = self.variant.itemData(index)
        if repo and repo != self.repo.text().strip():
            self.repo.setText(repo)

    def _pick_result(self):
        rows = self.resultTable.selectionModel().selectedRows()
        if rows:
            self.repo.setText(self.results[rows[0].row()].repo)

    def _local_dir(self):
        return self.settings.value("localModelDir", "") or os.path.join(os.path.expanduser("~"), "Models")

    def _suggest_target(self, repo):
        repo = repo.strip()
        if not repo:
            return
        if self.conn:
            self.target.setText("%s/models/%s" % (self.conn.base.rstrip("/"), repo))
        else:
            self.target.setText(os.path.normpath(os.path.join(self._local_dir(), *repo.split("/"))))

    def _download_place(self):
        """连着服务器：下到服务器上；没连：只能下到这台电脑上你选的目录"""
        connected = bool(self.conn)
        self.downloadBox.setTitle("下载模型（下到服务器上）" if connected else "下载模型（没连服务器：下到这台电脑上）")
        self.target.setPlaceholderText("服务器上的目录" if connected else "这台电脑上的目录")
        self.browseTarget.setVisible(not connected)
        self.viaLabel.setVisible(connected)
        self.downloadVia.setVisible(connected)
        self.uploadButton.setToolTip("选这台电脑上已经下好的模型文件夹，传到服务器上；断了重新传会接着传"
                                     if connected else "连上服务器后才能传")
        self.target.clear()
        self._suggest_target(self.repo.text())
        self._transfer_buttons()

    def _pick_local_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "模型下到这台电脑上的哪个目录", self._local_dir())
        if folder:
            self.settings.setValue("localModelDir", folder)
            self._suggest_target(self.repo.text())
            if not self.repo.text().strip():
                self.target.setText(os.path.normpath(folder))

    def _download(self):
        repo = self.repo.text().strip()
        target = self.target.text().strip()
        if "/" not in repo or not target:
            QMessageBox.information(self, APP, "填仓库（组织/名字）和%s的目录" % ("服务器上" if self.conn else "这台电脑上"))
            return
        try:
            include = self._include_args()
        except ValueError as error:
            QMessageBox.information(self, APP, "%s：在「选文件…」里至少勾一个" % error)
            return
        if not self.conn:
            self._download_local(repo, target, include)
            return
        via = self.downloadVia.currentData()
        self.settings.setValue("downloadVia", via)
        self.offsets["download"] = 0
        self.watch = "download"
        self.download_started = time.time()
        self.downloadBar.setValue(0)
        self.downloadText.setText("开始下载…")
        self.downloadButton.setEnabled(False)
        self.log("下载 %s → %s（%s%s%s）" % (repo, target, self.source.currentText(), "，经本机中转" if via == "relay" else "",
                                           "，选了 %d 个文件" % len(include) if include else ""))
        conn = self.conn

        def work():
            args = ["download", "--repo", repo, "--source", self.source.currentData(), "--dir", target]
            for path in include:
                args += ["--include", path]
            if via == "relay":
                args += ["--proxy", ",".join("http://127.0.0.1:%d" % port for port in conn.open_relay())]
            else:
                conn.close_relay()
            return conn.agent(*args)

        def failed(error):
            self.downloadButton.setEnabled(True)
            self.fail("下载失败")(error)

        background(work, done=self._job_started("下载"), failed=failed)

    def _cancel_download(self):
        if self.transfer:
            self.transfer["stop"].set()
            self.downloadText.setText("正在取消…")
            return
        self._stop_job("download")

    # ── 没连服务器：下到这台电脑上 ──

    def _download_local(self, repo, target, include=()):
        if self.transfer:
            return
        if not os.path.isabs(target):
            QMessageBox.information(self, APP, "「下到」要填这台电脑上的完整路径")
            return
        source = self.source.currentData()
        state = self._start_transfer("local", 0)
        self.log("下载 %s → 这台电脑的 %s（%s）" % (repo, target, self.source.currentText()))

        def sink(event):
            with state["lock"]:
                if event.get("event") == "progress":
                    state["done"], state["total"], state["file"] = event.get("done") or 0, event.get("total") or 0, event.get("file") or ""
                elif event.get("event") == "log":
                    state["lines"].append(event["text"])

        def work():
            # 只用 agent.py 的下载部分（和服务器上一样：多连接分段、断点续传、校验）；输出交给界面
            agentlib.sink = sink
            agentlib.setup_http()
            try:
                return agentlib.download_repo(source, repo, target, "", list(include), state["stop"])
            finally:
                agentlib.sink = None

        def done(directory):
            self.settings.setValue("uploadFrom", os.path.dirname(directory))
            self._finish_transfer("下好了：%s（连上服务器后可以「从本机上传…」传上去）" % directory, True)

        background(work, done=done,
                   failed=lambda error: self._finish_transfer("下载%s：%s" % ("已取消" if state["stop"].is_set() else "失败", error), False))

    # ── 从本机上传 ──

    def _upload_model(self):
        if not self.conn or self.transfer:
            return
        folder = QFileDialog.getExistingDirectory(self, "选本机上的模型文件夹",
                                                  self.settings.value("uploadFrom", ""))
        if not folder:
            return
        self.settings.setValue("uploadFrom", os.path.dirname(folder))
        files = local_model_files(folder)
        if not files:
            QMessageBox.information(self, APP, "%s 里没有要传的文件" % folder)
            return
        kind = agentlib.file_format([rel for _, rel, _ in files])
        if kind not in ("safetensors", "PyTorch") or "config.json" not in {rel for _, rel, _ in files}:
            answer = QMessageBox.question(self, "从本机上传", "%s 里不是 vLLM / SGLang 能直接跑的模型（%s%s），照样传吗？\n\n"
                                          "要传 vLLM 用的模型，选模型本身那层文件夹（Hugging Face 缓存是 models--组织--名字/snapshots/<一串字符>）。" % (
                                              folder, kind, "，没有 config.json" if kind in ("safetensors", "PyTorch") else ""),
                                          QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if answer != QMessageBox.Yes:
                return
        total = sum(size for _, _, size in files)
        repo = guess_repo(folder)
        default = "%s/models/%s" % (self.conn.base.rstrip("/"), repo)
        target, ok = QInputDialog.getText(self, "从本机上传", "%s\n%d 个文件，共 %s\n\n传到服务器上的目录：" % (folder, len(files), human(total)),
                                          QLineEdit.Normal, default)
        target = target.strip().rstrip("/")
        if not ok or not target:
            return
        state = self._start_transfer("upload", total)
        self.log("上传 %s → %s（%d 个文件，%s）" % (folder, target, len(files), human(total)))
        conn = self.conn

        def progress(n, name):
            with state["lock"]:
                state["done"] += n
                state["file"] = name

        def work():
            free = conn.remote_free(target)
            if free is not None and free < total:
                raise RuntimeError("服务器上 %s 所在的盘只剩 %s，模型要 %s" % (target, human(free), human(total)))
            conn.upload(files, target, progress, state["stop"])
            conn.write_file(target + "/.friday-vllm.json",
                            json.dumps({"repo": repo, "source": "upload", "from": folder, "downloaded": time.time()}, ensure_ascii=False))

        def done(_):
            self._finish_transfer("上传好了：%s" % target, True)
            self._refresh_models()

        background(work, done=done,
                   failed=lambda error: self._finish_transfer("上传%s：%s" % ("已取消" if state["stop"].is_set() else "失败", error), False))

    # ── 本机跑的传输：进度、按钮 ──

    def _start_transfer(self, kind, total):
        state = {"kind": kind, "stop": threading.Event(), "done": 0, "total": total, "file": "", "lines": [], "lock": threading.Lock(),
                 "window": [(time.time(), 0)]}
        self.transfer = state
        self.downloadBar.setValue(0)
        self.downloadText.setText("准备上传…" if kind == "upload" else "开始下载…")
        self._transfer_buttons()
        self.transferTimer.start()
        return state

    def _finish_transfer(self, text, ok):
        self._show_transfer()
        self.transfer = None
        self.transferTimer.stop()
        self._transfer_buttons()
        self.downloadText.setText(text)
        self.log(text)
        if ok:
            self.downloadBar.setValue(1000)

    def _transfer_buttons(self):
        busy = self.transfer is not None
        downloading = bool(self.conn) and (self.jobs.get("download") or {}).get("status") == "running"
        self.uploadButton.setEnabled(bool(self.conn) and not busy)
        self.downloadButton.setEnabled(not busy and not downloading)
        self.downloadStop.setEnabled(busy or downloading)
        self.downloadStop.setText("取消上传" if busy and self.transfer["kind"] == "upload" else "取消下载")

    def _show_transfer(self):
        state = self.transfer
        if not state:
            return
        now = time.time()
        with state["lock"]:
            done, total, name = state["done"], state["total"], state["file"]
            lines, state["lines"] = state["lines"], []
        for line in lines:
            self.log(line)
        window = state["window"]
        window.append((now, done))
        state["window"] = window = [w for w in window if now - w[0] <= 10]
        speed = (window[-1][1] - window[0][1]) / max(window[-1][0] - window[0][0], 0.001) if len(window) > 1 else 0
        if not total:
            return
        self.downloadBar.setValue(int(done * 1000 / total))
        left = (total - done) / speed if speed > 0 else 0
        upload = state["kind"] == "upload"
        self.downloadText.setText("%s %s / %s，%s/s%s%s" % ("上传" if upload else "下到这台电脑", human(done), human(total), human(speed),
                                                          "，还要约 " + duration(left) if left else "",
                                                          ("，正在传 " if upload else "，正在下 ") + name if name else ""))

    # ── 运行 ──

    def _serve_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.serveEngine = QComboBox()
        for engine, (title, *_rest) in ENGINES.items():
            self.serveEngine.addItem(title, engine)
        self.serveEngine.currentIndexChanged.connect(self._engine_changed)
        form.addRow("引擎", self.serveEngine)
        self.serveModel = QComboBox()
        self.serveModel.currentIndexChanged.connect(self._model_changed)
        form.addRow("模型", self.serveModel)
        # 多显卡：勾用哪几张；张量并行 TP（一份模型切到几张卡上，大模型一张放不下时用）× 数据并行 DP（开几份副本，小模型多卡时提吞吐）
        self.gpuChecks = []
        # 每行 4 张（8 卡机一行排不下）
        self.gpuGrid = QGridLayout()
        self.gpuGrid.setContentsMargins(0, 0, 0, 0)
        self.gpuEmpty = QLabel("连上服务器后列出它的显卡")
        self.gpuEmpty.setStyleSheet("color: gray")
        self.gpuGrid.addWidget(self.gpuEmpty, 0, 0)
        gpuWidget = QWidget()
        gpuWidget.setLayout(self.gpuGrid)
        form.addRow("显卡", gpuWidget)
        self.tpSize = QSpinBox()
        self.tpSize.setRange(1, 64)
        self.tpSize.setToolTip("张量并行：一份模型切到几张卡上（要能整除模型的注意力头数，一般是 1、2、4、8）")
        self.dpSize = QSpinBox()
        self.dpSize.setRange(1, 64)
        self.dpSize.setToolTip("数据并行：开几份副本，每份占 TP 张卡；并发请求分给各份，吞吐跟着翻倍")
        self.tpSize.valueChanged.connect(self._parallel_summary)
        self.dpSize.valueChanged.connect(self._parallel_summary)
        autoParallel = QPushButton("自动")
        autoParallel.setToolTip("按模型大小和单卡显存算：一张卡放得下就 TP 1、每张卡一份；放不下就切到能放下的最少张数")
        autoParallel.clicked.connect(self._auto_parallel)
        self.parallelText = QLabel()
        self.parallelText.setWordWrap(True)
        parallelRow = QHBoxLayout()
        parallelRow.addWidget(QLabel("张量并行 TP"))
        parallelRow.addWidget(self.tpSize)
        parallelRow.addWidget(QLabel("× 副本 DP"))
        parallelRow.addWidget(self.dpSize)
        parallelRow.addWidget(autoParallel)
        parallelRow.addWidget(self.parallelText, 1)
        form.addRow("并行", parallelRow)
        self.serveName = QLineEdit()
        self.serveName.setPlaceholderText("Friday 里填的模型名")
        form.addRow("模型名", self.serveName)
        self.servePort = QSpinBox()
        self.servePort.setRange(1024, 65535)
        self.servePort.setValue(int(self.settings.value("servePort", 8000)))
        self.serveHost = QComboBox()
        self.serveHost.addItem("所有网卡（0.0.0.0，别的机器能直接连）", "0.0.0.0")
        self.serveHost.addItem("只本机（127.0.0.1，只能走隧道）", "127.0.0.1")
        portRow = QHBoxLayout()
        portRow.addWidget(self.servePort)
        portRow.addWidget(self.serveHost, 1)
        form.addRow("端口", portRow)
        self.apiKey = QLineEdit(self.settings.value("apiKey", "") or "sk-" + secrets.token_urlsafe(24))
        regenerate = QPushButton("换一个")
        regenerate.clicked.connect(lambda: self.apiKey.setText("sk-" + secrets.token_urlsafe(24)))
        keyRow = QHBoxLayout()
        keyRow.addWidget(self.apiKey, 1)
        keyRow.addWidget(regenerate)
        form.addRow("API 密钥", keyRow)
        self.gpuUtil = QDoubleSpinBox()
        self.gpuUtil.setRange(0.3, 0.98)
        self.gpuUtil.setSingleStep(0.05)
        self.gpuUtil.setValue(float(self.settings.value("gpuUtil", 0.9)))
        self.gpuUtil.setToolTip("权重 + KV 缓存最多占每张卡显存的多少。别调到 0.95 以上：CUDA graph、现场编译的内核也要显存，"
                                "SGLang 在 0.98 时只剩 1 GB 多，起不来")
        self.maxLen = QSpinBox()
        self.maxLen.setRange(0, 4_000_000)
        self.maxLen.setSingleStep(8192)
        self.maxLen.setSpecialValueText("按模型")
        self.maxSeqs = QSpinBox()
        self.maxSeqs.setRange(1, 4096)
        self.maxSeqs.setValue(int(self.settings.value("maxSeqs", 256)))
        tuning = QHBoxLayout()
        tuning.addWidget(QLabel("显存占比"))
        tuning.addWidget(self.gpuUtil)
        tuning.addWidget(QLabel("最长上下文"))
        tuning.addWidget(self.maxLen)
        tuning.addWidget(QLabel("最多同时请求"))
        tuning.addWidget(self.maxSeqs)
        tuning.addStretch(1)
        form.addRow("调优", tuning)
        self.toolParser = QComboBox()
        self.reasoningParser = QComboBox()
        parserRow = QHBoxLayout()
        parserRow.addWidget(QLabel("工具调用"))
        parserRow.addWidget(self.toolParser, 1)
        parserRow.addWidget(QLabel("思考"))
        parserRow.addWidget(self.reasoningParser, 1)
        form.addRow("解析", parserRow)
        self.extraArgs = QLineEdit(self.settings.value("extraArgs", ""))
        self.extraArgs.setPlaceholderText("其他启动参数，例如 vLLM 的 --enforce-eager（启动快、跑得慢）、SGLang 的 --enable-torch-compile")
        form.addRow("其他参数", self.extraArgs)
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.startButton = QPushButton("启动")
        self.startButton.clicked.connect(self._start_serve)
        self.launchNote = QLabel("一台机器上可以同时开好几个服务：用不同的显卡、不同的端口；开好的在「服务」页里看、停")
        self.launchNote.setStyleSheet("color: gray")
        self.launchNote.setWordWrap(True)
        buttons.addWidget(self.startButton)
        buttons.addWidget(self.launchNote, 1)
        layout.addLayout(buttons)
        layout.addStretch(1)
        return page

    def _services_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.serviceTable = QTableWidget(0, 6)
        self.serviceTable.setHorizontalHeaderLabels(["服务", "引擎", "模型", "显卡", "端口", "状态"])
        self._table_style(self.serviceTable)
        self.serviceTable.itemSelectionChanged.connect(self._pick_service)
        layout.addWidget(self.serviceTable, 1)
        buttons = QHBoxLayout()
        self.stopButton = QPushButton("停止")
        self.stopButton.clicked.connect(self._stop_service)
        self.forgetButton = QPushButton("删除记录")
        self.forgetButton.setToolTip("删掉停了的服务的记录和日志")
        self.forgetButton.clicked.connect(self._forget_service)
        self.restartButton = QPushButton("重新启动")
        self.restartButton.setToolTip("照上次的设置再起一次（停了、没起来、服务器重启过的服务）")
        self.restartButton.clicked.connect(self._restart_service)
        self.editButton = QPushButton("改设置…")
        self.editButton.setToolTip("把这个服务的设置填进「新建服务」，改了再启动（模型名和端口不变就还是这个服务）")
        self.editButton.clicked.connect(self._edit_service)
        newButton = QPushButton("新建服务…")
        newButton.clicked.connect(lambda: self.tabs.setCurrentWidget(self.servePage))
        self.serveStatus = QLabel()
        self.serveStatus.setWordWrap(True)
        buttons.addWidget(self.restartButton)
        buttons.addWidget(self.stopButton)
        buttons.addWidget(self.editButton)
        buttons.addWidget(self.forgetButton)
        buttons.addWidget(newButton)
        buttons.addWidget(self.serveStatus, 1)
        layout.addLayout(buttons)

        box = QGroupBox("OpenAI 兼容地址（选中的服务）")
        grid = QGridLayout(box)
        self.remoteURL = readonly()
        self.tunnelURL = readonly()
        self.modelName = readonly()
        self.keyText = readonly()
        grid.addWidget(QLabel("远程地址"), 0, 0)
        grid.addWidget(self.remoteURL, 0, 1)
        grid.addWidget(copy_button(self.remoteURL.text), 0, 2)
        self.remoteNote = QLabel()
        self.remoteNote.setStyleSheet("color: gray")
        self.remoteNote.setWordWrap(True)
        grid.addWidget(self.remoteNote, 1, 1)
        grid.addWidget(QLabel("本机地址（隧道）"), 2, 0)
        grid.addWidget(self.tunnelURL, 2, 1)
        grid.addWidget(copy_button(self.tunnelURL.text), 2, 2)
        tunnelRow = QHBoxLayout()
        tunnelRow.addWidget(QLabel("本机端口"))
        self.localPort = QSpinBox()
        self.localPort.setRange(1024, 65535)
        self.localPort.setValue(int(self.settings.value("localPort", 18000)))
        tunnelRow.addWidget(self.localPort)
        self.tunnelButton = QPushButton("开隧道")
        self.tunnelButton.clicked.connect(self._toggle_tunnel)
        tunnelRow.addWidget(self.tunnelButton)
        self.testButton = QPushButton("测一下")
        self.testButton.clicked.connect(self._test)
        tunnelRow.addWidget(self.testButton)
        tunnelRow.addStretch(1)
        grid.addLayout(tunnelRow, 3, 1)
        grid.addWidget(QLabel("模型名"), 4, 0)
        grid.addWidget(self.modelName, 4, 1)
        grid.addWidget(copy_button(self.modelName.text), 4, 2)
        grid.addWidget(QLabel("API 密钥"), 5, 0)
        grid.addWidget(self.keyText, 5, 1)
        grid.addWidget(copy_button(self.keyText.text), 5, 2)
        friday = QLabel("在 Friday 里用：设置 › 服务与密钥 › 添加服务 › 自定义（OpenAI 兼容），地址、密钥、模型名照上面填；"
                        "起步并发可以调到 16 以上（vLLM、SGLang 并发越多越快）。每个服务可以各开一个隧道，隧道要这个窗口开着才通。")
        friday.setWordWrap(True)
        grid.addWidget(friday, 6, 0, 1, 3)
        layout.addWidget(box)
        return page

    def _engine_changed(self, *_):
        """换引擎：解析器的选项换成那家的叫法；没装的不能启动"""
        engine = self.serveEngine.currentData() or "vllm"
        title, _version, tools, reasonings = ENGINES[engine]
        for combo, names, off in ((self.toolParser, tools, "不开工具调用"), (self.reasoningParser, reasonings, "不解析")):
            combo.clear()
            for name in names:
                combo.addItem({"auto": "自动（按模型家族）", "none": off}.get(name, name), name)
        installed = bool((self.probe.get("engines") or {}).get(engine))
        self.startButton.setEnabled(bool(self.conn) and installed)
        self.startButton.setToolTip("" if installed else "还没装 %s（到「安装引擎」里装）" % title)

    def _model_changed(self, *_):
        path = self.serveModel.currentData() or ""
        if path:
            self.serveName.setText(os.path.basename(path.rstrip("/")).lower())
        self._auto_parallel()

    # ── 多显卡 ──

    def _rebuild_gpus(self, cards):
        """运行页的显卡勾选：每张卡一个；别人占着一成以上显存的缺省不勾"""
        for check in self.gpuChecks:
            self.gpuGrid.removeWidget(check)
            check.deleteLater()
        self.gpuChecks = []
        self.gpuEmpty.setVisible(not cards)
        taken = {}
        for service in self.services.values():
            if service.get("status") == "running":
                for index in service.get("gpus") or []:
                    taken[index] = service.get("name") or service.get("id")
        self.gpuUse = tuple(sorted(taken.items()))
        for position, card in enumerate(cards):
            used = card["memory_used_mib"] / max(card["memory_total_mib"], 1)
            name = card["name"].replace("NVIDIA ", "").replace("GeForce ", "").replace(" Blackwell Server Edition", "")
            check = QCheckBox("%s：%s · %.0f GB%s" % (card.get("index", position), name, card["memory_total_mib"] / 1024,
                                                    "（已用 %.1f GB）" % (card["memory_used_mib"] / 1024) if used > 0.02 else ""))
            check.setToolTip(card["name"])
            check.setProperty("index", card.get("index", position))
            check.setProperty("memory", card["memory_total_mib"] * 1024 * 1024)
            owner = taken.get(card.get("index", position))
            check.setChecked(used < 0.1 and owner is None)
            check.setProperty("taken", owner is not None)
            if owner is not None:
                check.setText(check.text().split("（")[0] + "（%s 在用）" % owner)
                check.setEnabled(False)
            check.toggled.connect(self._auto_parallel)
            self.gpuGrid.addWidget(check, position // 4, position % 4)
            self.gpuChecks.append(check)
        self._auto_parallel()

    def _chosen_gpus(self):
        return [check for check in self.gpuChecks if check.isChecked()]

    def _selected_model(self):
        path = self.serveModel.currentData()
        return next((m for m in self.models if m["path"] == path), None)

    def _auto_parallel(self, *_):
        chosen = self._chosen_gpus()
        if not chosen:
            self._parallel_summary()
            return
        model = self._selected_model() or {}
        size = model.get("size") or 0
        heads = model.get("heads") or 0
        memory = min(check.property("memory") for check in chosen)
        # 权重最多占切开后显存的八成（剩下给 KV 缓存）；TP 取 2 的幂、要能整除注意力头数
        tp = 1
        while tp * 2 <= len(chosen) and size > tp * memory * 0.8 and (not heads or heads % (tp * 2) == 0):
            tp *= 2
        for spin, value in ((self.tpSize, tp), (self.dpSize, max(1, len(chosen) // tp))):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._parallel_summary()

    def _parallel_summary(self, *_):
        chosen = self._chosen_gpus()
        tp, dp = self.tpSize.value(), self.dpSize.value()
        if not self.gpuChecks:
            self.parallelText.setText("")
            return
        if not chosen:
            self.parallelText.setText("<span style='color:#c33'>至少勾一张卡</span>")
            return
        model = self._selected_model() or {}
        notes = []
        if tp * dp > len(chosen):
            notes.append("<span style='color:#c33'>TP × DP = %d，只勾了 %d 张卡</span>" % (tp * dp, len(chosen)))
        heads = model.get("heads") or 0
        if heads and heads % tp:
            notes.append("<span style='color:#c33'>TP 要能整除注意力头数 %d</span>" % heads)
        size = model.get("size") or 0
        memory = min(check.property("memory") for check in chosen) * tp
        if size and size > memory * 0.9:
            notes.append("<span style='color:#c33'>模型 %s，%d 张卡共 %s，放不下：加大 TP</span>" % (human(size), tp, human(memory)))
        if not notes:
            if tp == 1 and dp == 1:
                notes.append("用 1 张卡")
            else:
                notes.append("用 %d 张卡：每份模型%s，开 %d 份副本" % (tp * dp, "切到 %d 张卡上" % tp if tp > 1 else "放一张卡", dp))
            if tp * dp < len(chosen):
                notes.append("（勾了 %d 张，空着 %d 张）" % (len(chosen), len(chosen) - tp * dp))
        self.parallelText.setText("".join(notes))

    def _start_serve(self):
        path = self.serveModel.currentData()
        if not path:
            QMessageBox.information(self, APP, "先在「模型」里下一个模型")
            return
        name = self.serveName.text().strip() or os.path.basename(path)
        for key, value in (("servePort", self.servePort.value()), ("apiKey", self.apiKey.text()), ("gpuUtil", self.gpuUtil.value()),
                           ("maxSeqs", self.maxSeqs.value()), ("extraArgs", self.extraArgs.text()), ("localPort", self.localPort.value())):
            self.settings.setValue(key, value)
        engine = self.serveEngine.currentData()
        args = ["serve", "--engine", engine, "--model", path, "--name", name, "--host", self.serveHost.currentData(), "--port", self.servePort.value(),
                "--gpu-util", "%.2f" % self.gpuUtil.value(), "--max-len", self.maxLen.value(), "--max-seqs", self.maxSeqs.value(),
                "--tool-parser", self.toolParser.currentData(), "--reasoning-parser", self.reasoningParser.currentData(),
                "--extra=" + self.extraArgs.text().strip()]
        if self.apiKey.text().strip():
            args += ["--key", self.apiKey.text().strip()]
        chosen = self._chosen_gpus()
        if self.gpuChecks:
            if not chosen:
                QMessageBox.information(self, APP, "至少勾一张显卡")
                return
            args += ["--gpus", ",".join(str(check.property("index")) for check in chosen),
                     "--tp", self.tpSize.value(), "--dp", self.dpSize.value()]
        self.log("启动 %s：%s（端口 %d）" % (ENGINES[engine][0], path, self.servePort.value()))

        background(self.conn.agent, *args, done=self._serve_started, failed=self.fail("启动失败"))

    def _serve_started(self, result):
        if not result.get("ok"):
            QMessageBox.warning(self, "启动失败", result.get("error") or "没能启动")
            self._update_addresses()
            return
        sid = result["id"]
        self.log("服务 %s：工具调用解析 %s，思考解析 %s；加载模型、编译要几分钟" % (
            sid, result.get("tool_parser") or "不开", result.get("reasoning_parser") or "不解析"))
        self.selectedService = sid
        self.watch = "serve:" + sid
        self.offsets[self.watch] = 0
        self.tabs.setCurrentWidget(self.servicesPage)
        self._poll()

    def _restart_service(self):
        service = self._service()
        if not service:
            return
        self.log("重新启动服务 %s" % service["id"])
        self.restartButton.setEnabled(False)
        background(self.conn.agent, "restart", "--id", service["id"], done=self._serve_started, failed=self.fail("启动失败"))

    def _edit_service(self):
        """把选中服务的设置填进「新建服务」页"""
        service = self._service()
        if not service:
            return
        # 早先的记录没存 options：用记下的那几项
        options = dict(service.get("options") or {})
        for name in ("engine", "model", "name", "host", "port", "key", "tp", "dp"):
            options.setdefault(name, service.get(name))
        options.setdefault("gpus", ",".join(str(i) for i in service.get("gpus") or []))
        options.setdefault("tool_parser", service.get("tool_parser") or "none")
        options.setdefault("reasoning_parser", service.get("reasoning_parser") or "none")
        for combo, value in ((self.serveEngine, options.get("engine") or "vllm"), (self.serveModel, options.get("model")),
                             (self.serveHost, options.get("host"))):
            index = combo.findData(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        if self.serveModel.currentData() != options.get("model"):
            self.log("服务器上的模型列表里没有 %s（被删了，或不在列的目录里）" % options.get("model"))
        self.serveName.setText(options.get("name") or "")
        if options.get("port"):
            self.servePort.setValue(int(options["port"]))
        if options.get("key"):
            self.apiKey.setText(options["key"])
        for spin, name in ((self.gpuUtil, "gpu_util"), (self.maxLen, "max_len"), (self.maxSeqs, "max_seqs")):
            if options.get(name) is not None:
                spin.setValue(type(spin.value())(options[name]))
        for combo, name in ((self.toolParser, "tool_parser"), (self.reasoningParser, "reasoning_parser")):
            index = combo.findData(options.get(name) or "none")
            if index >= 0:
                combo.setCurrentIndex(index)
        self.extraArgs.setText(options.get("extra") or "")
        gpus = {int(i) for i in str(options.get("gpus") or "").split(",") if i.strip()}
        if gpus:
            for check in self.gpuChecks:
                check.blockSignals(True)
                check.setChecked(not check.property("taken") and check.property("index") in gpus)
                check.blockSignals(False)
        for spin, name in ((self.tpSize, "tp"), (self.dpSize, "dp")):
            spin.blockSignals(True)
            spin.setValue(int(options.get(name) or 1))
            spin.blockSignals(False)
        self._parallel_summary()
        self.tabs.setCurrentWidget(self.servePage)

    def _service(self):
        return self.services.get(self.selectedService) or {}

    def _pick_service(self):
        rows = self.serviceTable.selectionModel().selectedRows()
        if not rows:
            return
        sid = self.serviceTable.item(rows[0].row(), 0).data(Qt.UserRole)
        if sid == self.selectedService:
            return
        self.selectedService = sid
        # 日志跟着选中的服务看（从末尾 64 KB 起）
        self.watch = "serve:" + sid
        self.offsets[self.watch] = -1
        if sid in self.tunnels:
            self.localPort.setValue(self.tunnels[sid])
        self._update_addresses()
        self._poll()

    def _stop_service(self):
        service = self._service()
        if not service:
            return
        background(self.conn.agent, "stop", "--job", "serve", "--id", service["id"],
                   done=lambda r: (self.log("服务 %s：%s" % (service["id"], r.get("message", ""))), self._poll()), failed=self.fail("停止失败"))

    def _forget_service(self):
        service = self._service()
        if not service:
            return
        def done(result):
            if not result.get("ok"):
                QMessageBox.information(self, APP, result.get("error") or "删不了")
                return
            self.selectedService = None
            self._poll()
        background(self.conn.agent, "forget", "--id", service["id"], done=done, failed=self.fail("删除失败"))

    def _apply_services(self, services):
        before = self.services
        self.services = services
        # 停了的服务：关掉它的隧道
        for sid, port in list(self.tunnels.items()):
            if (services.get(sid) or {}).get("status") != "running":
                self.conn.close_tunnel(port)
                del self.tunnels[sid]
                self.log("服务 %s 不在跑了，隧道已关" % sid)
        if self.selectedService not in services:
            running = [sid for sid, sv in services.items() if sv.get("status") == "running"]
            self.selectedService = (running or list(services) or [None])[0]
            if self.selectedService and not (self.watch or "").startswith(("install", "download")):
                self.watch = "serve:" + self.selectedService
        texts = {"running": "运行中", "failed": "没起来 / 退出了", "stopped": "已停止"}
        self.serviceTable.blockSignals(True)
        self.serviceTable.setRowCount(len(services))
        for row, (sid, service) in enumerate(services.items()):
            status = texts.get(service.get("status"), service.get("status") or "")
            if service.get("status") == "running" and not service.get("healthy"):
                status = "启动中…"
            parallel = ""
            if (service.get("tp") or 1) > 1 or (service.get("dp") or 1) > 1:
                parallel = "（TP %s × DP %s）" % (service.get("tp") or 1, service.get("dp") or 1)
            values = [service.get("name") or sid, ENGINES.get(service.get("engine") or "vllm", ("vLLM",))[0],
                      os.path.basename((service.get("model") or "").rstrip("/")),
                      ",".join(str(i) for i in service.get("gpus") or []) + parallel, str(service.get("port") or ""),
                      status + ("，隧道 %d" % self.tunnels[sid] if sid in self.tunnels else "")]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.UserRole, sid)
                self.serviceTable.setItem(row, column, item)
            if sid == self.selectedService:
                self.serviceTable.selectRow(row)
        self.serviceTable.blockSignals(False)
        # 新建服务页：别的服务占着的显卡、端口
        use = tuple(sorted((i, sv.get("name") or sid) for sid, sv in services.items() if sv.get("status") == "running" for i in sv.get("gpus") or []))
        if use != self.gpuUse:
            self._rebuild_gpus(self.probe.get("gpus") or [])
        ports = {int(sv.get("port") or 0) for sv in services.values() if sv.get("status") == "running"}
        if self.servePort.value() in ports:
            port = 8000
            while port in ports:
                port += 1
            self.servePort.setValue(port)
        if before.keys() != services.keys() or any((before.get(k) or {}).get("status") != v.get("status") for k, v in services.items()):
            self._show_probe_services(services)
        self._update_addresses()

    def _show_probe_services(self, services):
        self.probe["services"] = services
        if self.conn:
            self._show_probe()

    def _toggle_tunnel(self):
        service = self._service()
        sid = service.get("id")
        if sid in self.tunnels:
            self.conn.close_tunnel(self.tunnels.pop(sid))
            self.log("服务 %s 的隧道已关" % sid)
        else:
            if service.get("status") != "running":
                QMessageBox.information(self, APP, "选一个在跑的服务")
                return
            local = self.localPort.value()
            if local in self.tunnels.values():
                local = max(self.tunnels.values()) + 1
            port = free_local_port(local)
            self.localPort.setValue(port)
            try:
                self.conn.open_tunnel(port, service["port"])
                self.tunnels[sid] = port
                self.settings.setValue("localPort", port)
                self.log("服务 %s 的隧道：本机 127.0.0.1:%d → 服务器 127.0.0.1:%s" % (sid, port, service["port"]))
            except OSError as error:
                QMessageBox.warning(self, "开不了隧道", str(error))
        self._apply_services(self.services)

    def _test(self):
        service = self._service()
        url = self.tunnelURL.text() if service.get("id") in self.tunnels else self.remoteURL.text()
        key = self.keyText.text()
        if not url.startswith("http"):
            return

        def work():
            request = urllib.request.Request(url.split("（")[0].rstrip("/") + "/models", headers={"Authorization": "Bearer " + key} if key else {})
            with urllib.request.urlopen(request, timeout=10) as response:
                return [m["id"] for m in json.loads(response.read().decode())["data"]]

        background(work, done=lambda ids: QMessageBox.information(self, "通了", "%s\n能用的模型：%s" % (url, "、".join(ids))),
                   failed=self.fail("连不上 %s" % url))

    def _update_addresses(self):
        service = self._service()
        running = service.get("status") == "running"
        self.stopButton.setEnabled(running)
        self.restartButton.setEnabled(bool(service) and not running and bool(self.conn))
        self.editButton.setEnabled(bool(service))
        self.forgetButton.setEnabled(bool(service) and not running)
        self.tunnelButton.setEnabled(running)
        self.testButton.setEnabled(running)
        if not service:
            self.serveStatus.setText("还没有服务：到「新建服务」里开一个")
        elif running:
            self.serveStatus.setText("%s：%s" % (service["id"], "运行中" if service.get("healthy") else "启动中（加载模型、编译，几分钟），日志在下面"))
        else:
            self.serveStatus.setText("%s：%s" % (service["id"], service.get("message") or {"failed": "没起来，看下面的日志", "stopped": "已停止"}.get(service.get("status"), "")))
        if not running:
            for field in (self.remoteURL, self.tunnelURL, self.modelName, self.keyText):
                field.clear()
            self.remoteNote.clear()
            self.tunnelButton.setText("开隧道")
            return
        port = service.get("port")
        if service.get("host") == "127.0.0.1":
            self.remoteURL.setText("http://127.0.0.1:%s/v1（只在服务器上能用）" % port)
            self.remoteNote.setText("只听本机：别的机器连不上，用下面的隧道")
        else:
            self.remoteURL.setText("http://%s:%s/v1" % (self.conn.host, port))
            ips = "、".join(self.probe.get("ips") or [])
            self.remoteNote.setText("要这个端口能从外面访问（云服务器放行安全组 / 防火墙；AutoDL 这类经网关的机器直接连不上，用隧道或它的自定义服务端口）"
                                    + ("；服务器的内网 IP：%s" % ips if ips else ""))
        tunnel = self.tunnels.get(service["id"])
        self.tunnelURL.setText("http://127.0.0.1:%d/v1" % tunnel if tunnel else "")
        self.tunnelURL.setPlaceholderText("" if tunnel else "点「开隧道」")
        self.tunnelButton.setText("关隧道" if tunnel else "开隧道")
        self.modelName.setText(service.get("name") or "")
        self.keyText.setText(service.get("key") or "")

    # ── 轮询服务器 ──

    def _poll(self):
        if not self.conn or self.polling:
            return
        self.polling = True
        watch = self.watch

        def work():
            status = self.conn.agent("status", timeout=30)
            tail = None
            if watch:
                tail = self.conn.agent("tail", "--job", watch, "--offset", self.offsets.get(watch, -1), timeout=30)
            return status, watch, tail

        def done(result):
            self.polling = False
            status, job, tail = result
            self._apply_jobs(status.get("jobs") or {})
            self._apply_services(status.get("services") or {})
            if tail:
                self.offsets[job] = tail["offset"]
                self._apply_log(job, tail["text"])

        def failed(error):
            self.polling = False
            if not self.conn.connected:
                self.timer.stop()
                self._set_connected(False)
                self.log("和服务器断开了：%s" % error)

        background(work, done=done, failed=failed)

    def _apply_log(self, job, text):
        for line in text.splitlines():
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except ValueError:
                    self.log(line)
                    continue
                if event.get("event") == "progress" and job == "download":
                    self._show_download(event)
                continue
            if line.strip():
                self.log(line)

    def _show_download(self, event):
        total = event.get("total") or 0
        done = event.get("done") or 0
        speed = event.get("speed") or 0
        self.downloadBar.setValue(int(done * 1000 / total) if total else 0)
        left = (total - done) / speed if speed > 0 else 0
        self.downloadText.setText("%s / %s，%s/s%s%s" % (human(done), human(total), human(speed),
                                                         "，还要约 " + duration(left) if left else "",
                                                         "，正在下 " + event["file"] if event.get("file") else ""))

    def _apply_jobs(self, jobs):
        before = self.jobs
        self.jobs = jobs
        install = jobs.get("install") or {}
        download = jobs.get("download") or {}
        texts = {"running": "正在进行", "done": "完成", "failed": "失败", "stopped": "已停止"}
        self.installStatus.setText("%s%s" % (texts.get(install.get("status"), ""), "：" + install["message"] if install.get("message") else ""))
        self.installButton.setEnabled(install.get("status") != "running")
        self.installStop.setEnabled(install.get("status") == "running")
        self._transfer_buttons()
        # 经本机中转的下载：重连后中转没了，在服务器上原来那个端口重新开；下完了就关
        argv = download.get("argv") or []
        relay = argv[argv.index("--proxy") + 1] if "--proxy" in argv[:-1] else ""
        if download.get("status") == "running" and relay and self.conn and not self.conn.relay_ports and not self.relayReopening:
            self.relayReopening = True
            ports = [int(proxy.rsplit(":", 1)[-1]) for proxy in relay.split(",")]

            def reopened(_):
                self.relayReopening = False
                self.log("接上经本机中转的下载（服务器端口 %s）" % "、".join(str(p) for p in ports))

            def reopen_failed(error):
                self.relayReopening = False
                self.log("接不上经本机中转的下载（%s）：取消后重新点下载，会接着下" % error)

            background(self.conn.open_relay, ports, done=reopened, failed=reopen_failed)
        elif download.get("status") != "running" and (before.get("download") or {}).get("status") == "running" and self.conn and self.conn.relay_ports:
            background(self.conn.close_relay)
        if download.get("status") in ("done", "failed", "stopped") and download.get("message") and (before.get("download") or {}).get("status") == "running":
            self.downloadText.setText(download["message"])
            if download.get("status") == "done":
                self.downloadBar.setValue(1000)
            self._refresh_models()
        if install.get("status") in ("done", "failed") and (before.get("install") or {}).get("status") == "running":
            background(self.conn.agent, "probe", timeout=60, done=self._reprobe, failed=self.fail("读服务器信息失败"))
        self.startButton.setEnabled(bool(self.conn) and bool((self.probe.get("engines") or {}).get(self.serveEngine.currentData())))
        # 没在看日志时，跟着在跑的任务看
        if not self.watch:
            for job in ("download", "install"):
                if (jobs.get(job) or {}).get("status") == "running":
                    self.watch = job
                    break

    def _reprobe(self, probe):
        self.probe = probe
        self._show_probe()

    def closeEvent(self, event):
        if self.conn:
            self.conn.close()
        super().closeEvent(event)


def selftest(path):
    """打包后的自检（CI 和本机核对用）：agent.py 在不在、界面建不建得起来，结果写到 path"""
    from pathlib import Path
    agent = Path(__file__).resolve().parent / "agent" / "agent.py"
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    ok = agent.is_file() and b"def cmd_probe" in agent.read_bytes() and window.tabs.count() == 6
    with open(path, "w", encoding="utf-8") as f:
        f.write("ok\n" if ok else "failed: agent=%s\n" % agent)
    return 0 if ok else 1


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    app = QApplication(sys.argv)
    app.setApplicationName(APP)
    app.setOrganizationName("Friday")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
