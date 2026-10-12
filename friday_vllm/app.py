"""Friday vLLM 的主窗口：连上远程服务器 → 装 vLLM / SGLang → 列出 / 下载模型 → 启动，给出 OpenAI 兼容地址（远程地址和本机隧道地址）。

Friday 本身不用改：把这里给的地址、密钥、模型名填进 Friday「设置 › 服务与密钥」的自定义服务（OpenAI 兼容）即可。
"""
import json
import os
import secrets
import shlex
import sys
import time
import urllib.request

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
                               QWidget)

from . import __version__, catalog
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
        self.watch = None            # 正在看日志的任务：install / download / serve
        self.download_started = None

        tabs = QTabWidget()
        tabs.addTab(self._server_tab(), "服务器")
        tabs.addTab(self._install_tab(), "安装引擎")
        tabs.addTab(self._models_tab(), "模型")
        tabs.addTab(self._serve_tab(), "运行")
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
        for index in (1, 2, 3):
            self.tabs.widget(index).setEnabled(connected)
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
        serve = (p.get("jobs") or {}).get("serve") or {}
        if serve.get("status") == "running":
            rows.append(("服务", "在跑：%s（%s），端口 %s%s" % (serve.get("name"), ENGINES.get(serve.get("engine") or "vllm", ("vLLM",))[0],
                                                          serve.get("port"), "" if serve.get("healthy") else "，还在启动")))
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
        self.modelTable = QTableWidget(0, 6)
        self.modelTable.setHorizontalHeaderLabels(["模型", "大小", "架构", "量化", "状态", "目录"])
        self._table_style(self.modelTable)
        self.modelTable.cellDoubleClicked.connect(self._use_model_row)
        localLayout.addWidget(self.modelTable)
        hint = QLabel("双击一行：在「运行」里用它")
        hint.setStyleSheet("color: gray")
        localLayout.addWidget(hint)
        layout.addWidget(local, 2)

        remote = QGroupBox("下载模型（下到服务器上）")
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
        filterRow.addWidget(QLabel("量化"))
        self.quantFilter = QComboBox()
        self.quantFilter.addItem("全部", "")
        for name in catalog.QUANT_NAMES:
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
        filterRow.addStretch(1)
        self.resultCount = QLabel()
        self.resultCount.setStyleSheet("color: gray")
        filterRow.addWidget(self.resultCount)
        remoteLayout.addLayout(filterRow)
        self.resultsAll = []
        self.results = []
        self.resultTable = QTableWidget(0, 7)
        self.resultTable.setHorizontalHeaderLabels(["仓库", "量化", "大小", "下载量", "更新", "架构", "装好的引擎支持吗"])
        self._table_style(self.resultTable)
        self.resultTable.itemSelectionChanged.connect(self._pick_result)
        remoteLayout.addWidget(self.resultTable)
        grid = QGridLayout()
        grid.addWidget(QLabel("仓库"), 0, 0)
        self.repo = QLineEdit()
        self.repo.setPlaceholderText("组织/名字，例如 Qwen/Qwen3.6-35B-A3B-FP8")
        self.repo.textChanged.connect(self._suggest_target)
        grid.addWidget(self.repo, 0, 1)
        grid.addWidget(QLabel("下到"), 1, 0)
        self.target = QLineEdit()
        self.target.setPlaceholderText("服务器上的目录")
        grid.addWidget(self.target, 1, 1)
        self.downloadButton = QPushButton("下载")
        self.downloadButton.clicked.connect(self._download)
        self.downloadStop = QPushButton("取消下载")
        self.downloadStop.clicked.connect(lambda: self._stop_job("download"))
        grid.addWidget(self.downloadButton, 0, 2)
        grid.addWidget(self.downloadStop, 1, 2)
        remoteLayout.addLayout(grid)
        self.downloadBar = QProgressBar()
        self.downloadBar.setRange(0, 1000)
        self.downloadText = QLabel()
        remoteLayout.addWidget(self.downloadBar)
        remoteLayout.addWidget(self.downloadText)
        layout.addWidget(remote, 3)
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
            values = [model["name"], human(model["size"]), "、".join(model["architectures"]), model["quantization"] or "—",
                      "完整" if model["complete"] else "没下完", model["path"]]
            for column, value in enumerate(values):
                self.modelTable.setItem(row, column, QTableWidgetItem(str(value)))
        current = self.serveModel.currentText()
        self.serveModel.blockSignals(True)
        self.serveModel.clear()
        for model in self.models:
            if model["complete"]:
                self.serveModel.addItem("%s（%s）" % (model["name"], human(model["size"])), model["path"])
        self.serveModel.blockSignals(False)
        index = self.serveModel.findText(current)
        if index >= 0:
            self.serveModel.setCurrentIndex(index)
        self._model_changed()

    def _use_model_row(self, row, _column):
        path = self.models[row]["path"]
        index = self.serveModel.findData(path)
        if index >= 0:
            self.serveModel.setCurrentIndex(index)
            self.tabs.setCurrentIndex(3)

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
            self.log("没搜到（GGUF、MLX 这类 vLLM / SGLang 用不了的已经去掉了）")

    def _engine_archs(self):
        """装好的引擎各自支持的架构（没装、没读到的不算）"""
        engines = self.probe.get("engines") or {}
        return {e: set(a) for e, a in (self.probe.get("archs") or {}).items() if a and engines.get(e) and e in ENGINES}

    def _apply_filters(self, *_):
        models = list(self.resultsAll)
        quant = self.quantFilter.currentData()
        if quant:
            models = [m for m in models if m.quant == quant]
        cards = self.probe.get("gpus") or []
        self.fitsGpu.setEnabled(bool(cards))
        if not cards:
            self.fitsGpu.setToolTip("没检测到这台服务器的显卡（没连上，或 AutoDL 无卡模式）")
        if self.fitsGpu.isChecked() and cards:
            memory = sum(c["memory_total_mib"] for c in cards) * 1024 * 1024 * 0.9
            try:
                capability = min(float(c["compute_cap"]) for c in cards)
            except ValueError:
                capability = 0.0
            models = [m for m in models if (not m.size or m.size <= memory) and capability >= catalog.QUANT_MIN_CAP.get(m.quant, 0)]
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
            values = [model.repo, model.quant, human(model.size), str(model.downloads or ""),
                      time.strftime("%Y-%m-%d", time.localtime(model.updated)) if model.updated else "—",
                      "、".join(model.architectures), supported]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column in (2, 3):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.resultTable.setItem(row, column, item)
        self.resultCount.setText("显示 %d / 共 %d 个" % (len(models), len(self.resultsAll)) if self.resultsAll else "")

    def _pick_result(self):
        rows = self.resultTable.selectionModel().selectedRows()
        if rows:
            self.repo.setText(self.results[rows[0].row()].repo)

    def _suggest_target(self, repo):
        if self.conn and repo.strip():
            self.target.setText("%s/models/%s" % (self.conn.base.rstrip("/"), repo.strip()))

    def _download(self):
        repo = self.repo.text().strip()
        target = self.target.text().strip()
        if "/" not in repo or not target:
            QMessageBox.information(self, APP, "填仓库（组织/名字）和服务器上的目录")
            return
        self.offsets["download"] = 0
        self.watch = "download"
        self.download_started = time.time()
        self.downloadBar.setValue(0)
        self.downloadText.setText("开始下载…")
        self.log("下载 %s → %s（%s）" % (repo, target, self.source.currentText()))
        background(self.conn.agent, "download", "--repo", repo, "--source", self.source.currentData(), "--dir", target,
                   done=self._job_started("下载"), failed=self.fail("下载失败"))

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
        self.stopButton = QPushButton("停止")
        self.stopButton.clicked.connect(lambda: self._stop_job("serve"))
        self.serveStatus = QLabel("没在跑")
        buttons.addWidget(self.startButton)
        buttons.addWidget(self.stopButton)
        buttons.addWidget(self.serveStatus, 1)
        layout.addLayout(buttons)

        box = QGroupBox("OpenAI 兼容地址")
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
                        "起步并发可以调到 16 以上（vLLM、SGLang 并发越多越快）。隧道要这个窗口开着才通。")
        friday.setWordWrap(True)
        grid.addWidget(friday, 6, 0, 1, 3)
        layout.addWidget(box)
        layout.addStretch(1)
        return page

    def _engine_changed(self, *_):
        """换引擎：解析器的选项换成那家的叫法；没装的不能启动"""
        engine = self.serveEngine.currentData() or "vllm"
        title, _version, tools, reasonings = ENGINES[engine]
        for combo, names, off in ((self.toolParser, tools, "不开工具调用"), (self.reasoningParser, reasonings, "不解析")):
            combo.clear()
            for name in names:
                combo.addItem({"auto": "自动（按模型家族）", "none": off}.get(name, name), name)
        running = (self.jobs.get("serve") or {}).get("status") == "running"
        installed = bool((self.probe.get("engines") or {}).get(engine))
        self.startButton.setEnabled(bool(self.conn) and not running and installed)
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
        for position, card in enumerate(cards):
            used = card["memory_used_mib"] / max(card["memory_total_mib"], 1)
            name = card["name"].replace("NVIDIA ", "").replace("GeForce ", "").replace(" Blackwell Server Edition", "")
            check = QCheckBox("%s：%s · %.0f GB%s" % (card.get("index", position), name, card["memory_total_mib"] / 1024,
                                                    "（已用 %.1f GB）" % (card["memory_used_mib"] / 1024) if used > 0.02 else ""))
            check.setToolTip(card["name"])
            check.setProperty("index", card.get("index", position))
            check.setProperty("memory", card["memory_total_mib"] * 1024 * 1024)
            check.setChecked(used < 0.1)
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
                "--extra", self.extraArgs.text().strip()]
        if self.apiKey.text().strip():
            args += ["--key", self.apiKey.text().strip()]
        chosen = self._chosen_gpus()
        if self.gpuChecks:
            if not chosen:
                QMessageBox.information(self, APP, "至少勾一张显卡")
                return
            args += ["--gpus", ",".join(str(check.property("index")) for check in chosen),
                     "--tp", self.tpSize.value(), "--dp", self.dpSize.value()]
        self.offsets["serve"] = 0
        self.watch = "serve"
        self.log("启动 %s：%s（端口 %d）" % (ENGINES[engine][0], path, self.servePort.value()))

        def done(result):
            if not result.get("ok"):
                QMessageBox.warning(self, "启动失败", result.get("error") or "没能启动")
                return
            self.log("工具调用解析：%s，思考解析：%s；加载模型、编译要几分钟" % (result.get("tool_parser") or "不开", result.get("reasoning_parser") or "不解析"))
            self._poll()

        background(self.conn.agent, *args, done=done, failed=self.fail("启动失败"))

    def _toggle_tunnel(self):
        serve = self.jobs.get("serve") or {}
        local = self.localPort.value()
        if self.conn.tunnel_open(local):
            self.conn.close_tunnel(local)
            self.log("隧道已关")
        else:
            if not serve.get("port"):
                QMessageBox.information(self, APP, "服务还没启动")
                return
            port = free_local_port(local)
            if port != local:
                self.localPort.setValue(port)
            try:
                self.conn.open_tunnel(port, serve["port"])
                self.log("隧道：本机 127.0.0.1:%d → 服务器 127.0.0.1:%s" % (port, serve["port"]))
            except OSError as error:
                QMessageBox.warning(self, "开不了隧道", str(error))
        self._update_addresses()

    def _test(self):
        url = self.tunnelURL.text() if self.conn.tunnel_open(self.localPort.value()) else self.remoteURL.text()
        key = self.keyText.text()

        def work():
            request = urllib.request.Request(url.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key} if key else {})
            with urllib.request.urlopen(request, timeout=10) as response:
                return [m["id"] for m in json.loads(response.read().decode())["data"]]

        background(work, done=lambda ids: QMessageBox.information(self, "通了", "%s\n能用的模型：%s" % (url, "、".join(ids))),
                   failed=self.fail("连不上 %s" % url))

    def _update_addresses(self):
        serve = self.jobs.get("serve") or {}
        running = serve.get("status") == "running"
        if not running:
            for field in (self.remoteURL, self.tunnelURL, self.modelName, self.keyText):
                field.clear()
            self.remoteNote.clear()
            self.tunnelButton.setText("开隧道")
            return
        port = serve.get("port")
        if serve.get("host") == "127.0.0.1":
            self.remoteURL.setText("http://127.0.0.1:%s/v1（只在服务器上能用）" % port)
            self.remoteNote.setText("只听本机：别的机器连不上，用下面的隧道")
        else:
            self.remoteURL.setText("http://%s:%s/v1" % (self.conn.host, port))
            ips = "、".join(self.probe.get("ips") or [])
            self.remoteNote.setText("要这个端口能从外面访问（云服务器放行安全组 / 防火墙；AutoDL 这类经网关的机器直接连不上，用隧道或它的自定义服务端口）"
                                    + ("；服务器的内网 IP：%s" % ips if ips else ""))
        tunnel = self.conn.tunnel_open(self.localPort.value())
        self.tunnelURL.setText("http://127.0.0.1:%d/v1" % self.localPort.value() if tunnel else "")
        self.tunnelURL.setPlaceholderText("点「开隧道」" if not tunnel else "")
        self.tunnelButton.setText("关隧道" if tunnel else "开隧道")
        self.modelName.setText(serve.get("name") or "")
        self.keyText.setText(serve.get("key") or "")

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
        serve = jobs.get("serve") or {}
        texts = {"running": "正在进行", "done": "完成", "failed": "失败", "stopped": "已停止"}
        self.installStatus.setText("%s%s" % (texts.get(install.get("status"), ""), "：" + install["message"] if install.get("message") else ""))
        self.installButton.setEnabled(install.get("status") != "running")
        self.installStop.setEnabled(install.get("status") == "running")
        self.downloadButton.setEnabled(download.get("status") != "running")
        self.downloadStop.setEnabled(download.get("status") == "running")
        if download.get("status") in ("done", "failed", "stopped") and download.get("message") and (before.get("download") or {}).get("status") == "running":
            self.downloadText.setText(download["message"])
            if download.get("status") == "done":
                self.downloadBar.setValue(1000)
            self._refresh_models()
        if install.get("status") in ("done", "failed") and (before.get("install") or {}).get("status") == "running":
            background(self.conn.agent, "probe", timeout=60, done=self._reprobe, failed=self.fail("读服务器信息失败"))
        running = serve.get("status") == "running"
        if running:
            cards = "，显卡 %s" % ",".join(str(i) for i in serve.get("gpus") or []) if serve.get("gpus") else ""
            if (serve.get("tp") or 1) > 1 or (serve.get("dp") or 1) > 1:
                cards += "（TP %s × DP %s）" % (serve.get("tp") or 1, serve.get("dp") or 1)
            self.serveStatus.setText("运行中：%s%s" % (serve.get("name"), cards) if serve.get("healthy") else "启动中（加载模型、编译，几分钟）…")
        else:
            self.serveStatus.setText({"failed": "没起来，看下面的日志", "stopped": "已停止"}.get(serve.get("status"), "没在跑"))
            if self.conn and self.conn.tunnel_open(self.localPort.value()) and (before.get("serve") or {}).get("status") == "running":
                self.conn.close_tunnel(self.localPort.value())
        self.startButton.setEnabled(not running and bool((self.probe.get("engines") or {}).get(self.serveEngine.currentData())))
        self.stopButton.setEnabled(running)
        self._update_addresses()
        # 没在看日志时，跟着在跑的任务看
        if not self.watch:
            for job in ("download", "install", "serve"):
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
    ok = agent.is_file() and b"def cmd_probe" in agent.read_bytes() and window.tabs.count() == 4
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
