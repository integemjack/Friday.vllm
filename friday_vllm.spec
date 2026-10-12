# PyInstaller：Windows 打成单个 FridayVLLM.exe，Mac 打成 Friday vLLM.app。
#   pyinstaller --noconfirm friday_vllm.spec
import sys

sys.path.insert(0, SPECPATH)  # noqa: F821（PyInstaller 给的：spec 所在目录）
from friday_vllm import __version__  # noqa: E402

block_cipher = None
# agent.py 是传到服务器上跑的脚本：作为数据文件放在包里同样的相对位置（ssh.py 按 __file__ 找它）
datas = [("friday_vllm/agent/agent.py", "friday_vllm/agent")]

a = Analysis(
    ["run.py"],
    pathex=["."],
    datas=datas,
    hiddenimports=["keyring.backends.Windows", "keyring.backends.macOS"],
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if sys.platform == "darwin":
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Friday vLLM", console=False, target_arch=None)
    coll = COLLECT(exe, a.binaries, a.datas, name="Friday vLLM")
    app = BUNDLE(
        coll,
        name="Friday vLLM.app",
        bundle_identifier="com.integem.friday.vllm",
        info_plist={
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
else:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="FridayVLLM", console=False, upx=False)
