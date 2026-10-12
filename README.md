# Friday vLLM

连上一台远程 Linux + NVIDIA 显卡的服务器（AutoDL、云服务器、自己的机器），在上面：

1. **装 vLLM 或 SGLang**：各建一个虚拟环境（两家要的 torch 版本不一样），用 uv 装（真下一个文件比速度挑最快的 pip 镜像、
   并行下载、失败重试）；
2. **列出 / 下载模型**：在魔搭、Hugging Face、hf-mirror 上搜 vLLM 能用的模型（safetensors，GGUF 这类排除），
   下到服务器上你指定的目录（多连接分段下载、断点续传、sha256 校验）；也能列出服务器上已有的模型；
3. **启动**：选引擎，`vllm serve` 或 `python -m sglang.launch_server`，按模型家族自动配工具调用和思考的解析器（两家的叫法不一样，各按各的）；
4. **给出 OpenAI 兼容地址**：远程地址（`http://服务器:端口/v1`）和本机隧道地址（`http://127.0.0.1:端口/v1`，经 SSH 隧道，
   AutoDL 这类不开公网端口的机器也能用），加上模型名、API 密钥。

Friday 不用改：在 Friday「设置 › 服务与密钥 › 添加服务 › 自定义（OpenAI 兼容）」里填这里给的地址、密钥、模型名即可；
起步并发可以调到 16 以上（vLLM 并发越多越快）。

## 用法

- Windows：下载 Release 里的 `FridayVLLM-*-windows-x64.zip`，解压后双击 `FridayVLLM.exe`。
- Mac（Apple 芯片）：下载 `FridayVLLM-*-macos-arm64.zip`，解压后把 `Friday vLLM.app` 拖进「应用程序」。
  没签名：第一次打开要在「系统设置 › 隐私与安全性」里点「仍要打开」。

1. **服务器**：填主机、端口、用户，密码或密钥文件（不填就用 `~/.ssh` 里的密钥和 ssh-agent）。工作目录不填时，
   AutoDL 上用 `/root/autodl-tmp/friday-vllm`（数据盘；系统盘只有 30 GB），其他机器用 `~/friday-vllm`。
   密码可以存进系统的钥匙串 / 凭据管理器。
2. **安装引擎**：选 vLLM（缺省 0.31.0）或 SGLang（缺省最新），和 pip 镜像（缺省自动挑最快的）。各要 8–10 GB 磁盘。
   安装不需要显卡，AutoDL 可以先用无卡模式开机装好，再带卡开机运行。
3. **模型**：搜索 → 选一行 → 下载（下到哪个目录可以改）。「也看这些目录」里填别的目录（比如 `/dev/shm`）也会列出来。
   双击已下好的模型，在「运行」里用它。
4. **运行**：选引擎、模型、端口、API 密钥（自动生成）、显存占比、最长上下文、最多同时请求，点「启动」。
   加载模型、编译要几分钟；好了以后「OpenAI 兼容地址」里给出地址，「开隧道」后本机地址可用，「测一下」核对能不能连上。
   **多显卡**：勾用哪几张（别人占着一成以上显存的缺省不勾）。「张量并行 TP」把一份模型切到几张卡上（一张放不下的大模型用），
   「副本 DP」开几份（小模型多卡时提吞吐），「自动」按模型大小和单卡显存算：一张放得下就每张卡一份，放不下就切到放得下的最少张数。
   TP 要能整除模型的注意力头数（一般 1、2、4、8），启动前会核对。编号按 nvidia-smi 的（启动时设 `CUDA_DEVICE_ORDER=PCI_BUS_ID`）。

安装、下载、服务都在服务器上后台跑（`setsid`）：关掉窗口、断线了照样继续，重新连上会接着显示进度和日志。
隧道要这个窗口开着才通；要长期用，云服务器可以放行端口直接用远程地址，AutoDL 可以把端口设成 6006 用它的「自定义服务」。

## 服务器上的布局

```
<工作目录>/
  agent.py      每次连上时上传的助手脚本（服务器上的 Python 3.8 起就能跑，只用标准库）
  env/          vLLM 的虚拟环境
  env-sglang/   SGLang 的虚拟环境
  uv-cache/     uv 的下载缓存（和 env 同一个盘时用硬链接，不多占空间）
  models/       缺省的下载位置（<组织>/<名字>/）
  vllm-cache/   torch.compile 的缓存（第二次启动快很多）
  logs/         install.log、download.log、serve.log
  state/        各任务的状态（pid、端口、密钥……）
```

不想用界面时，`agent.py` 也能直接在服务器上用：`python3 agent.py --help`。

## 开发

```bash
pip install -r requirements.txt
python run.py
```

打包：`pip install pyinstaller && pyinstaller --noconfirm friday_vllm.spec`（Windows 出 `dist/FridayVLLM.exe`，Mac 出 `dist/Friday vLLM.app`）。
推 `vX.Y.Z` 标签时 GitHub Actions 打两个平台的包并建 Release。

## 要求

- 本机：Windows 10/11 x64，或 macOS 12 起的 Apple 芯片 Mac。
- 服务器：Linux x64 + NVIDIA 显卡；vLLM 0.2x 起默认用 CUDA 13，要驱动 580 起（旧驱动装旧一点的 vLLM，比如 0.11.0）；
  有 Python 3.10–3.13（AutoDL 自带的 Miniconda 就行）。
