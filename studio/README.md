# Studio

```text
studio/
  core/
    core.py                  # 启动服务、监听、等待、叫停、路由、回复
    voiceagent.py            # 组合组件，持有应用状态
    voicemem.py              # 唯一 VoiceMem 初始化与原生流式入口
    utils/
      asr/                   # component.py + initialize.py
      eot/
      llm/                   # 默认 DeepSeek；保留本地回复与缓存
      tts/                   # Breeze、初始化、快速缓存、语气协议
      conversation/          # 会话逻辑与状态初始化
      reply_modes/
      turn_taking/
      models/                # 权重完整性检查与下载清单
      startup/               # 环境检查与严格预热
      ...                    # 捕获、打断、回放、入库、展示等独立组件
  models/                    # 只放权重，不放 Python 实现
  harness/
    persona/policy.py
    speaking_style/policy.py
    reply_modes/policy.py
    turn_taking/policy.py
  web/                       # 浏览器、HTTP/WebSocket、AudioWorklets
  apps/
```

## 启动

Linux/NVIDIA 的完整 Docker 部署及 Mac 原生部署见 [部署说明](../docker/README.md)。
Linux 在 `.env` 配置好 API Key 后可直接执行 `docker compose up -d --build`。

安装好对应的 Python 3.12 环境后，在项目根目录启动。Linux / NVIDIA 使用：

```bash
./scripts/run_studio_cuda.sh
```

macOS / Apple Silicon 使用：

```bash
./scripts/run_studio_mlx.sh
```

脚本自动使用对应虚拟环境、选择后端，并开启详细终端日志，不需要手动激活环境。
两者仍调用同一个 Python 入口；如果已经激活了对应环境，也可以直接运行：

```bash
python -m studio --verbose
```

直接使用 Python 入口且未覆盖配置时，Linux 自动选择 CUDA，Mac 自动选择 MLX。
默认 DeepSeek、中文、
`studio-zh` 记忆空间和 `8787` 端口，CUDA 默认只使用 `cuda:0`。
原来的 `python web/run.py` 仍使用同一入口。需要其他记忆空间时加 `--space 空间名`；
直接使用 Python 入口时，省略 `--verbose` 可切换为精简终端日志。
启动会沿用所选空间已保存的语言，不会重写已有记忆；旧的 `demo-zh` 仍可用
`--space demo-zh` 打开。

打开 `http://localhost:8787`；远程使用需 HTTPS 或本地 SSH 端口转发才能让浏览器使用麦克风。

## 首次安装

### Linux / NVIDIA CUDA

Studio 在进程内加载 Breeze CUDA，通过专用线程流式生成 PCM，不需要单独启动 Breeze
HTTP 服务。使用已有的 Breeze CUDA streaming 源码仓库和权重；两者默认是本仓库旁边的
`breeze-tts/` 和 `breeze-tts-2/`。其他位置在 `.env` 中配置 `BREEZE_CODE_DIR` 和
`BREEZE_MODEL_DIR`。源码必须包含 `models/fast_streaming.py`。

Python 3.12 首次安装（保留原有 Mac/旧 Studio 环境）：

```bash
python3.12 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install -e '.[studio-cuda]'
```

以上为本机 Python 环境；Docker 部署不需要在宿主机安装这些依赖。

### macOS / Apple Silicon MLX

同样使用 Python 3.12，保留原生 MLX 模型和调度方式：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[studio]'
```

## 配置和启动检查

首次创建配置后，在 `.env` 中填写 `DEEPSEEK_API_KEY`；已有文件时直接编辑，不要覆盖：

```bash
cp -n .env.example .env
```

启动依次检查凭据、依赖版本、资源、四份 policy 和模型清单；缺项集中打印并退出。
自动加载仓库根目录 `.env`，已导出的环境变量优先；兼容 `.env.qwen`。
DeepSeek 模式的回复和记忆处理都使用 `DEEPSEEK_API_KEY`，不再要求额外 OpenAI key。
`--llm openai` / `--mode realtime` 使用 `OPENAI_API_KEY`；`--llm qwen` 使用
`DASHSCOPE_API_KEY`。`--llm local` 仅支持 MLX。

下面是可选覆盖项，正常启动无需设置：

| 配置 | 用途 |
| --- | --- |
| `STUDIO_BACKEND` / `--backend` | `cuda` 或 `mlx`；未指定时 Linux 默认 CUDA，Mac 默认 MLX |
| `STUDIO_DEVICE` / `--device` | CUDA ASR/Router 设备，默认 `cuda:0` |
| `STUDIO_TTS_DEVICE` / `--tts-device` | Breeze CUDA 设备，默认与 Studio 相同 |
| `STUDIO_MODELS_DIR` | 模型根目录，默认 `studio/models/` |
| `BREEZE_CODE_DIR` | Breeze CUDA streaming 实现目录 |
| `BREEZE_MODEL_DIR` | Breeze CUDA 权重目录，未指定则复用旁边的 `breeze-tts-2` |

命令行参数优先于环境变量和 `.env`。两种后端分别检查依赖；CUDA 不检查或下载 MLX。
`python -m studio --check` 只检查，不下载、不打开 Memory Space。

检查通过后，完整的原有权重以本地链接复用到 `studio/models/`，缺少的权重自动下载；
中断后再次启动会复用下载缓存。选中的模型预热失败会阻止服务启动。
参考录音和审核后的附和素材保留在 `voice/`，不会自动生成替代声音。

当前入口在同一进程中先初始化 VoiceMem，再开放 Studio 服务；没有独立的记忆 RPC
服务。`core/voicemem.py` 直接调用原生流式接口，保留线程、取消与投机检索边界。
Memory Spaces、录音、日志和原有模型目录保留。旧 provider 导入仅兼容转发。

Studio prompt 只保留一份，不再按语言复制；`--lang` 继续控制 ASR 和 Memory Space。
VoiceMem 库自身的多语言默认 prompt 保留。四秒追问只针对明确未完成的残句；
“不是”“不对”“我有一个问题”等不触发。续说会合并，断开和切换空间会取消或丢弃。

离线回归覆盖状态和协议；真实音色、麦克风与端到端延时需要完整原生模型环境验收。

MLX uses `transformers==5.16.1`, Hub 1.x, and `mlx-audio==0.5.1`.
CUDA uses Torch 2.8.0, `transformers==4.57.3`, Hub 0.x, and `qwen-tts==0.1.1`.
Install the matching extra in its own environment; do not combine both extras.

情绪识别使用共享的 SenseVoiceSmall CPU 实例，保留情绪标签，不再生成多模态情绪原因。

## 回复路由与长垫话

三档仍是 instant、mem、mem+cot，但不再由小模型独自决定是否使用记忆：

| VoiceMem 判断需要记忆 | 小模型判断需要深思 | 最终模式 |
| --- | --- | --- |
| 否 | 否 | instant |
| 是 | 否 | mem |
| 任意 | 是 | mem+cot |

记忆资格沿用 `voicemem/gate.py` 原有判断，复用已经完成的检索或补齐缺失检索。
小模型只根据当前发言和最近 4 条消息（共 320 字符）回答“是否需要深思”，提示在
`harness/reply_modes/policy.py`。Studio 不再额外按日期、问句或关键词硬编码路由。
普通推理或小模型调用失败都不会否决 Gate 已批准的记忆；陌生人声纹仍不能访问主人记忆。
原 Gate 和小模型都可能误判，但记忆与推理各自负责自己的部分，不再重复筛掉记忆。

mem+cot 在正文音频尚未就绪时使用原有长垫话流程，不再被最近闲聊较快的耗时估计挡住。
正文和垫话继续并行生成；正文先准备好就跳过垫话，垫话已经播放则等待它结束再放行正文。
这不改变讲话途中的附和、未完句续话计时、TTS 切句或提前生成。
选择 mem 不代表一定能查到日程：记忆库必须已有相关记录，没有记录时不能编造。

```bash
python -m unittest evals.test_thinking_router evals.test_dialogue_harness.TurnTakingTimingTests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m evals.router_quality --device cuda:0 --assert-quality
```

后者仅测本地模型的深思分类，不访问记忆库；记忆资格的保留由前面的确定性回归验证。
这些检查不能替代实际记忆召回、完整 GPU 负载下的延迟或真实听感验收。

## TTS 切句

正文 TTS 使用两端共用的完整句优先切分：首段通常不再在短逗号分句处提交，
后续优先句末，长句才在逗号等位置分段。首段文字缓冲等待上限为 450ms；
后续为 600ms，浏览器确认正在播放且剩余音频充足时最多放宽到 1200ms。
等待从该段首个正文片段进入缓冲开始计算，不包含 LLM 首字和 TTS 合成耗时。
句末或 LLM 输出结束会立即提交，不固定等待到上限。长度兜底为首段 48、后续 100 字符，
首段达到 28、后续达到 60 字符时允许逗号等软边界。语气指令和音频 chunk 参数未改变。

## 可选：CUDA 性能验证

CUDA 默认预热并使用 depth decoder 的编译加速，首次启动需要等待 CUDA graph 准备。
完成的语音段在日志中报告 `[tts-cuda] RTF`；RTF 小于 1 表示交付速度快于音频播放。
激活 CUDA 环境后，可用以下命令验证单卡供给速度（合成测试，不访问个人记忆或 LLM API）：

```bash
python -m evals.breeze_cuda_latency --device cuda:0 --with-asr --assert-realtime
python -m evals.breeze_cuda_latency --device cuda:0 --with-asr --segmented --assert-realtime
```
